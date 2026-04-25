"""merge_weights.py — CLI for interpolating multimodal fine-tuned and text-only Tiny Aya weights.

Usage
-----
python scripts/merge_weights.py \\
    --original  CohereLabs/tiny-aya-base \\
    --finetuned ./checkpoints/tiny-aya-vision-ft \\
    --alpha     0.5 \\
    --output    ./merged/alpha_0.5 \\
    [--save-hf] \\
    [--dtype bfloat16] \\
    [--device cpu]

Merge strategy
--------------
Only the language-model backbone (all keys prefixed with ``language_model.``) is
interpolated via linear interpolation (LERP):

    merged_param = (1 - α) × original_param  +  α × finetuned_param

The multimodal projector (``multi_modal_projector.*``) and vision encoder
(``vision_encoder.*``) weights are kept verbatim from the fine-tuned checkpoint.

α = 0.0  →  identical to the original text-only Tiny Aya Base
α = 1.0  →  identical to the multimodal fine-tuned VLM
Recommended sweep range: {0.3, 0.4, 0.5, 0.6, 0.7}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from pipeline.merge_utils import (
    LLM_PREFIX,
    PROJECTOR_PREFIX,
    _load_original_llm,
    _load_finetuned_vlm,
    _print_merge_summary,
    _restore_tied_weights,
    _save_outputs,
    build_merged_vlm_state,
    extract_llm_state_dict,
    extract_non_llm_state_dict,
    lerp_state_dicts,
)

__all__ = [
    "LLM_PREFIX",
    "PROJECTOR_PREFIX",
    "_restore_tied_weights",
    "build_merged_vlm_state",
    "extract_llm_state_dict",
    "extract_non_llm_state_dict",
    "lerp_state_dicts",
    "parse_args",
]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multimodal fine-tuned Tiny Aya weights with text-only base weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original",
        required=True,
        help="HuggingFace Hub ID or local path for the text-only Tiny Aya Base LLM.",
    )
    parser.add_argument(
        "--finetuned",
        required=True,
        help="Path to fine-tuned VLM checkpoint (.pt file or directory containing one).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Merge ratio α ∈ [0, 1]. 0 = pure original text model; 1 = pure fine-tuned VLM.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. merged_state.pt will be written here.",
    )
    parser.add_argument(
        "--save-hf",
        action="store_true",
        default=False,
        help="Also save the merged LLM backbone as a HuggingFace model directory.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype to cast weights to before saving.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load model weights onto (e.g. 'cpu', 'cuda:0').",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    alpha = args.alpha
    if not 0.0 <= alpha <= 1.0:
        log.error("--alpha must be in [0, 1], got %.3f", alpha)
        sys.exit(1)

    dtype = getattr(torch, args.dtype)
    output_dir = Path(args.output)

    original_llm_state = _load_original_llm(args.original, args.device, dtype)
    finetuned_vlm_state = _load_finetuned_vlm(args.finetuned, args.device)

    merged_vlm_state = build_merged_vlm_state(original_llm_state, finetuned_vlm_state, alpha)

    merged_llm_state = extract_llm_state_dict(merged_vlm_state)
    _print_merge_summary(original_llm_state, merged_llm_state, alpha, output_dir)

    _save_outputs(
        merged_vlm_state,
        output_dir,
        dtype=dtype,
        save_hf=args.save_hf,
        original_llm_name=args.original,
    )

    log.info("Done. ✓")


if __name__ == "__main__":
    main()
