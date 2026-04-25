"""merge_utils.py — Core weight-merging utilities for Tiny Aya Vision.

Contains all merge logic shared between the standalone CLI
(scripts/merge_weights.py) and the post-training merge step called
automatically at the end of SFT training scripts.

Merge strategy
--------------
Only the language-model backbone (keys prefixed with ``language_model.``) is
interpolated via linear interpolation (LERP):

    merged_param = (1 - α) × original_param  +  α × finetuned_param

The multimodal projector (``multi_modal_projector.*``) and vision encoder
(``vision_encoder.*``) weights are kept verbatim from the fine-tuned checkpoint
because they contain no text-only signal.

α = 0.0  →  identical to the original text-only Tiny Aya Base
α = 1.0  →  identical to the multimodal fine-tuned VLM
"""

from __future__ import annotations

import gc
import logging
import re
import sys
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_PREFIX = "language_model."
PROJECTOR_PREFIX = "multi_modal_projector."

_TIED_PAIRS = [
    ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight"),
]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the checkpoint_*.pt file with the highest step number, or None."""
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    def _step(p: Path) -> int:
        m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    return max(checkpoints, key=_step)


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def _restore_tied_weights(state: Dict[str, torch.Tensor]) -> None:
    """Restore weight-tied keys that safetensors deduplication may have dropped."""
    for src_key, tied_key in _TIED_PAIRS:
        if src_key in state and tied_key not in state:
            state[tied_key] = state[src_key]


def lerp_state_dicts(
    original: Dict[str, torch.Tensor],
    finetuned: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Linear interpolation of two state dicts with matching keys.

    Args:
        original:  State dict of the text-only LLM (keys without any prefix).
        finetuned: State dict of the fine-tuned LLM (keys without any prefix).
        alpha:     Merge coefficient in [0, 1]. 0 → original; 1 → finetuned.

    Returns:
        New state dict with merged tensors (detached, on CPU).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    orig_keys = set(original.keys())
    ft_keys = set(finetuned.keys())
    missing_in_ft = orig_keys - ft_keys
    missing_in_orig = ft_keys - orig_keys

    if missing_in_ft or missing_in_orig:
        raise ValueError(
            f"Key mismatch between original and finetuned state dicts.\n"
            f"  Missing in finetuned:  {sorted(missing_in_ft)[:5]!r}{'...' if len(missing_in_ft) > 5 else ''}\n"
            f"  Missing in original:   {sorted(missing_in_orig)[:5]!r}{'...' if len(missing_in_orig) > 5 else ''}"
        )

    merged: Dict[str, torch.Tensor] = {}
    for key in original:
        orig_t = original[key]
        ft_t = finetuned[key]
        if orig_t.shape != ft_t.shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': "
                f"original={tuple(orig_t.shape)}, finetuned={tuple(ft_t.shape)}"
            )
        orig_f = orig_t.float()
        ft_f = ft_t.float()
        merged[key] = ((1.0 - alpha) * orig_f + alpha * ft_f).to(orig_t.dtype).detach().cpu()

    return merged


def extract_llm_state_dict(full_vlm_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract LLM parameters from a full VLM state dict, stripping the prefix."""
    return {
        key[len(LLM_PREFIX):]: val
        for key, val in full_vlm_state.items()
        if key.startswith(LLM_PREFIX)
    }


def extract_non_llm_state_dict(full_vlm_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract non-LLM parameters (projector + vision encoder) from VLM state dict."""
    return {
        key: val
        for key, val in full_vlm_state.items()
        if not key.startswith(LLM_PREFIX)
    }


def build_merged_vlm_state(
    original_llm_state: Dict[str, torch.Tensor],
    finetuned_vlm_state: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Build a complete merged VLM state dict.

    LLM keys are linearly interpolated; projector and vision encoder keys
    are copied from the fine-tuned checkpoint untouched.

    Args:
        original_llm_state:  State dict of ``AutoModelForCausalLM`` (no prefix).
        finetuned_vlm_state: State dict of ``TinyAyaVisionForConditionalGeneration``.
        alpha:               Merge coefficient in [0, 1].
    """
    ft_llm_state = extract_llm_state_dict(finetuned_vlm_state)
    non_llm_state = extract_non_llm_state_dict(finetuned_vlm_state)

    tied_keys = set(original_llm_state) - set(ft_llm_state)
    if tied_keys:
        log.info(
            "Excluding %d key(s) from original absent in finetuned "
            "(weight-tied or not saved — will be restored on model load): %s",
            len(tied_keys), sorted(tied_keys),
        )
        original_llm_state = {k: v for k, v in original_llm_state.items() if k not in tied_keys}

    log.info("Merging %d LLM parameter tensors with α=%.2f …", len(original_llm_state), alpha)

    merged_llm = lerp_state_dicts(original_llm_state, ft_llm_state, alpha)

    merged_vlm: Dict[str, torch.Tensor] = {}
    for key, val in merged_llm.items():
        merged_vlm[f"{LLM_PREFIX}{key}"] = val
    merged_vlm.update(non_llm_state)

    return merged_vlm


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_merge_summary(
    original_llm: Dict[str, torch.Tensor],
    merged_llm: Dict[str, torch.Tensor],
    alpha: float,
    output_path: Path,
) -> None:
    total_params = sum(t.numel() for t in merged_llm.values())
    delta_sq_sum = sum(
        ((merged_llm[k].float() - original_llm[k].float()) ** 2).sum().item()
        for k in original_llm
    )
    norm_delta = delta_sq_sum ** 0.5

    print("\n" + "=" * 60)
    print("  Tiny Aya Vision — Weight Merge Summary")
    print("=" * 60)
    print(f"  α (merge ratio)   : {alpha:.2f}  (0=text-only, 1=full VLM)")
    print(f"  LLM param tensors : {len(original_llm):,}")
    print(f"  Total params (LLM): {total_params:,}")
    print(f"  ‖merged − orig‖₂  : {norm_delta:.4f}")
    print(f"  Output path       : {output_path}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_original_llm(model_name: str, device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Load the original text-only LLM state dict."""
    log.info("Loading original LLM from '%s' …", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def _load_finetuned_vlm(checkpoint_path: str, device: str, token: str = "") -> Dict[str, torch.Tensor]:
    """Load a fine-tuned VLM state dict from a HuggingFace Hub ID, a local HF
    model directory, or a raw ``.pt`` / ``.safetensors`` checkpoint file.

    Loading priority
    ----------------
    1. HF Hub ID or local dir with config.json → AutoModel.from_pretrained
    2. Directory with raw weight files → loads first candidate
    3. Single weight file → loads directly
    """
    from transformers import AutoModel

    p = Path(checkpoint_path)
    is_hf_dir = p.is_dir() and (p / "config.json").exists()
    is_hub_id = not p.exists()

    if is_hf_dir or is_hub_id:
        log.info("Loading fine-tuned VLM via AutoModel.from_pretrained('%s') …", checkpoint_path)
        try:
            kwargs = {"torch_dtype": torch.float32, "trust_remote_code": True}
            if token:
                kwargs["token"] = token
            if device != "cpu":
                kwargs["device_map"] = device
                kwargs["low_cpu_mem_usage"] = True
            model = AutoModel.from_pretrained(checkpoint_path, **kwargs)
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _restore_tied_weights(state)
            return state
        except (ValueError, OSError) as exc:
            if (
                "model_type" not in str(exc)
                and "Unrecognized" not in str(exc)
                and "does not recognize this architecture" not in str(exc)
            ):
                raise
            log.warning(
                "AutoModel could not instantiate '%s' (%s). "
                "Falling back to raw weight download via snapshot_download ...",
                checkpoint_path, exc,
            )

        from huggingface_hub import snapshot_download
        snap_kwargs = {}
        if token:
            snap_kwargs["token"] = token
        local_dir = Path(snapshot_download(checkpoint_path, **snap_kwargs))
        log.info("Snapshot downloaded to '%s'", local_dir)

        safetensor_shards = sorted(local_dir.glob("*.safetensors"))
        if safetensor_shards:
            from safetensors.torch import load_file
            state: Dict[str, torch.Tensor] = {}
            for shard in safetensor_shards:
                log.info("  Loading shard: %s", shard.name)
                state.update(load_file(str(shard), device=device))
        else:
            bin_files = sorted(local_dir.glob("*.bin")) + sorted(local_dir.glob("*.pt"))
            if not bin_files:
                raise FileNotFoundError(
                    f"No .safetensors / .bin / .pt weight files found in '{local_dir}'."
                )
            state = {}
            for bf in bin_files:
                log.info("  Loading: %s", bf.name)
                chunk = torch.load(str(bf), map_location=device, weights_only=True)
                if isinstance(chunk, dict) and "state_dict" in chunk:
                    chunk = chunk["state_dict"]
                if isinstance(chunk, dict) and "model" in chunk:
                    chunk = chunk["model"]
                state.update(chunk)

        state = {k: v.detach().cpu() for k, v in state.items()}
        _restore_tied_weights(state)
        return state

    if p.is_dir():
        candidates = list(p.glob("*.pt")) + list(p.glob("*.pth")) + list(p.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(
                f"No .pt / .pth / .safetensors checkpoint file found in '{checkpoint_path}'."
            )
        checkpoint_path = str(candidates[0])
        log.info("Found checkpoint file: %s", checkpoint_path)

    log.info("Loading fine-tuned VLM state dict from '%s' …", checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(checkpoint_path, device=device)
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    state = {k: v.detach().cpu() for k, v in state.items()}
    _restore_tied_weights(state)
    return state


def _save_outputs(
    merged_state: Dict[str, torch.Tensor],
    output_dir: Path,
    dtype: torch.dtype,
    save_hf: bool,
    original_llm_name: str,
) -> None:
    """Save merged state dict and optionally a HuggingFace model directory.

    Always writes ``merged_state.pt`` (full VLM: LLM + connector + vision encoder).
    When ``save_hf=True``, also writes ``hf_model/`` containing the merged LLM
    backbone via ``save_pretrained`` and ``connector_state.pt``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cast_state = {k: v.to(dtype) for k, v in merged_state.items()}

    pt_path = output_dir / "merged_state.pt"
    torch.save(cast_state, pt_path)
    log.info("Saved full merged VLM state dict → %s", pt_path)

    if save_hf:
        hf_dir = output_dir / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)

        llm_state = {
            key[len(LLM_PREFIX):]: val
            for key, val in cast_state.items()
            if key.startswith(LLM_PREFIX)
        }
        log.info("Building HF LLM model at '%s' …", hf_dir)
        model = AutoModelForCausalLM.from_pretrained(original_llm_name, torch_dtype=dtype)
        missing, unexpected = model.load_state_dict(llm_state, strict=False)
        if missing:
            log.warning(
                "HF save: %d missing LLM keys (expected if vocab was resized): %s …",
                len(missing), missing[:3],
            )
        if unexpected:
            log.warning("HF save: %d unexpected LLM keys: %s …", len(unexpected), unexpected[:3])
        model.save_pretrained(str(hf_dir))
        log.info("Saved HF LLM backbone → %s", hf_dir)
        del model

        connector_state = {
            key: val
            for key, val in cast_state.items()
            if key.startswith(PROJECTOR_PREFIX)
        }
        if connector_state:
            connector_path = hf_dir / "connector_state.pt"
            torch.save(connector_state, connector_path)
            log.info(
                "Saved connector weights (%d tensors) → %s",
                len(connector_state), connector_path,
            )
        else:
            log.warning("No '%s*' keys found in merged state — connector not saved.", PROJECTOR_PREFIX)


# ---------------------------------------------------------------------------
# Post-training merge (called from training scripts with in-memory model)
# ---------------------------------------------------------------------------

def run_post_training_merge(
    raw_model,
    original_llm_name: str,
    alpha: float,
    output_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
    save_hf: bool = True,
) -> None:
    """Merge LoRA-tuned VLM weights with the original text-only LLM and save.

    Called at the end of SFT training with the still-in-memory model.
    LoRA adapters are merged into the base LLM weights before the LERP so
    the interpolation operates on dense parameter tensors.

    Args:
        raw_model:          Unwrapped ``TinyAyaVisionForConditionalGeneration``
                            (no DDP / torch.compile wrappers).
        original_llm_name:  HF Hub ID of the text-only base LLM
                            (e.g. ``"CohereLabs/tiny-aya-base"``).
        alpha:              LERP coefficient in [0, 1].
        output_dir:         Directory to write ``merged_state.pt`` (and optionally
                            ``hf_model/``) into.
        dtype:              Dtype to cast weights to before saving.
        save_hf:            Whether to also save an HF-format model directory.
    """
    log.info("Starting post-training merge (α=%.2f) → %s", alpha, output_dir)

    # Merge LoRA adapters into base weights and get the dense LLM module.
    # merge_and_unload() folds lora_A/B into the base weight tensors and
    # returns a plain nn.Module without PEFT wrappers.
    merged_lm = raw_model.language_model.merge_and_unload()

    # Assemble the full VLM state dict expected by build_merged_vlm_state.
    finetuned_vlm_state: Dict[str, torch.Tensor] = {}
    for k, v in merged_lm.state_dict().items():
        finetuned_vlm_state[f"{LLM_PREFIX}{k}"] = v.detach().cpu()
    for k, v in raw_model.multi_modal_projector.state_dict().items():
        finetuned_vlm_state[f"{PROJECTOR_PREFIX}{k}"] = v.detach().cpu()
    for k, v in raw_model.vision_encoder.state_dict().items():
        finetuned_vlm_state[f"vision_encoder.{k}"] = v.detach().cpu()
    _restore_tied_weights(finetuned_vlm_state)

    del merged_lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    original_llm_state = _load_original_llm(original_llm_name, "cpu", dtype)
    merged_vlm_state = build_merged_vlm_state(original_llm_state, finetuned_vlm_state, alpha)

    merged_llm_state = extract_llm_state_dict(merged_vlm_state)
    _print_merge_summary(original_llm_state, merged_llm_state, alpha, output_dir)

    _save_outputs(merged_vlm_state, output_dir, dtype, save_hf, original_llm_name)
    log.info("Post-training merge complete. ✓")
