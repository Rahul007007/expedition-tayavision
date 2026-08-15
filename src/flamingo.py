"""Flamingo-style conditioning: gated cross-attention over resampled media.

Implements the two pieces that make Flamingo (Alayrac et al., 2022) different
from a LLaVA-style prefix-token VLM:

1. A **resampler** that maps a variable-size frozen vision grid to a small,
   fixed set of media tokens (Perceiver Resampler, or SoftWhere's multi-foveal
   TokenLearner — see :mod:`src.softwhere`).
2. **Gated cross-attention dense layers** interleaved between the frozen LLM
   decoder layers.  Both the attention and the feed-forward branch are scaled
   by ``tanh(gate)`` with ``gate`` initialised at 0, so the conditioned model
   starts out *exactly* equal to the frozen text-only LLM and learns to use
   vision gradually.

Text tokens attend only to the media of the most recently preceding ``<image>``
marker (``only_attend_immediate_media=True``), matching Flamingo's masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perceiver Resampler (Flamingo baseline)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Pre-norm MLP block used inside the resampler and the xattn blocks."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner, bias=False)
        self.fc2 = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class PerceiverAttention(nn.Module):
    """Latent-query attention over ``[media ; latents]`` (Flamingo eq. 1)."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.dim_head).transpose(1, 2)

    def forward(self, media: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        media = self.norm_media(media)
        latents = self.norm_latents(latents)

        q = self._split(self.to_q(latents)) * self.scale
        kv_input = torch.cat([media, latents], dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)

        attn = torch.matmul(q, k.transpose(-1, -2)).softmax(dim=-1)
        out = torch.matmul(attn, v)
        b, _, n, _ = out.shape
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Flamingo's Perceiver Resampler: N patch tokens -> R latent media tokens."""

    def __init__(self, config):
        super().__init__()
        dim = config.vision_hidden_size
        self.num_latents = config.num_latent_tokens
        self.latents = nn.Parameter(torch.randn(self.num_latents, dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim, dim_head=config.xattn_head_dim, heads=config.xattn_num_heads
                        ),
                        FeedForward(dim, mult=config.xattn_ff_mult),
                    ]
                )
                for _ in range(config.perceiver_depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

        self.norm_media = nn.LayerNorm(dim)
        self.linear_1 = nn.Linear(dim, config.connector_intermediate_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(
            config.connector_intermediate_size // 2, config.llm_hidden_size, bias=True
        )

    @property
    def num_media_tokens(self) -> int:
        return self.num_latents

    def forward(self, vision_features: torch.Tensor) -> dict[str, torch.Tensor]:
        b = vision_features.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1).to(vision_features.dtype)

        for attn, ff in self.layers:
            latents = attn(vision_features, latents) + latents
            latents = ff(latents) + latents

        latents = self.norm(latents)
        hidden = self.linear_1(self.norm_media(latents))
        x, gate = hidden.chunk(2, dim=-1)
        return {
            "media_tokens": self.linear_2(self.act(gate) * x),
            "media_bias": None,
            "attn": None,
            "diversity_loss": vision_features.new_zeros(()),
            "patch_indices": None,
        }


def create_resampler(config) -> nn.Module:
    """Factory: build the resampler named by ``config.resampler_type``."""
    from src.softwhere import SoftWhereResampler

    resamplers = {
        "softwhere": SoftWhereResampler,
        "perceiver": PerceiverResampler,
    }
    if config.resampler_type not in resamplers:
        raise ValueError(
            f"Unknown resampler_type '{config.resampler_type}'. "
            f"Choose from: {list(resamplers)}"
        )
    return resamplers[config.resampler_type](config)


# ---------------------------------------------------------------------------
# Gated cross-attention
# ---------------------------------------------------------------------------


class MaskedCrossAttention(nn.Module):
    """Cross-attention from text hidden states to media tokens.

    ``text_time`` gives, per text position, the 1-indexed number of ``<image>``
    markers seen so far (0 = before any image).  Media token ``j`` belongs to
    image ``media_time[j]``.  With ``only_attend_immediate_media`` a text token
    sees exactly the most recent image; otherwise it sees every preceding one.
    Positions that precede all media produce a zero contribution.
    """

    def __init__(
        self,
        dim: int,
        dim_media: int,
        dim_head: int = 64,
        heads: int = 8,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.only_attend_immediate_media = only_attend_immediate_media
        inner = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_media = nn.LayerNorm(dim_media)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim_media, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        text_time: torch.Tensor,
        media_valid: torch.Tensor,
        media_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
        x: ``(B, T, D)`` text hidden states.
        media: ``(B, M, n, D_media)`` resampled tokens for M images.
        text_time: ``(B, T)`` long — images seen so far at each text position.
        media_valid: ``(B, M)`` bool — which image slots are real.
        media_bias: optional ``(B, M, n)`` additive bias on the attention
            logits, used by the SoftWhere resampler to route gradient into its
            patch selector (see ``SoftWhereResampler.forward``).
        """
        b, m, n, _ = media.shape
        media_flat = media.reshape(b, m * n, -1)

        q = self._split(self.to_q(self.norm(x))) * self.scale
        k, v = self.to_kv(self.norm_media(media_flat)).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)

        sim = torch.matmul(q, k.transpose(-1, -2))  # (B, H, T, M*n)

        if media_bias is not None:
            sim = sim + media_bias.reshape(b, 1, 1, m * n).to(sim.dtype)

        media_time = torch.arange(1, m + 1, device=x.device).repeat_interleave(n)
        if self.only_attend_immediate_media:
            mask = text_time.unsqueeze(-1) == media_time.view(1, 1, -1)
        else:
            mask = text_time.unsqueeze(-1) >= media_time.view(1, 1, -1)
        mask = mask & media_valid.repeat_interleave(n, dim=1).unsqueeze(1)

        # Fully masked rows (text before any image, or text-only batches) are
        # filled with a finite minimum so softmax stays NaN-free, then zeroed.
        sim = sim.masked_fill(~mask.unsqueeze(1), torch.finfo(sim.dtype).min)
        attn = sim.softmax(dim=-1)
        attn = attn * mask.any(dim=-1).unsqueeze(1).unsqueeze(-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, x.shape[1], self.heads * self.dim_head)
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    """Flamingo XATTN-Dense block with ``tanh`` gates initialised at zero."""

    def __init__(
        self,
        dim: int,
        dim_media: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_media=dim_media,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        text_time: torch.Tensor,
        media_valid: torch.Tensor,
        media_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.attn(x, media, text_time, media_valid, media_bias)
        x = x + attended * self.attn_gate.tanh()
        x = x + self.ff(x) * self.ff_gate.tanh()
        return x


class FlamingoLayer(nn.Module):
    """Wraps a frozen decoder layer, running a gated xattn block before it.

    Unknown attribute lookups are forwarded to the wrapped layer so that
    transformers internals (e.g. ``decoder_layer.attention_type`` on Cohere2)
    keep working.
    """

    def __init__(self, gated_xattn: GatedCrossAttentionBlock, decoder_layer: nn.Module):
        super().__init__()
        # `gated_xattn` is deliberately *not* registered as a child module: the
        # blocks are owned by the model's `xattn_blocks` ModuleList, so each
        # tensor appears exactly once in the state dict (safetensors rejects
        # duplicates, and DDP would otherwise reduce the same grad twice).
        object.__setattr__(self, "gated_xattn", gated_xattn)
        self.decoder_layer = decoder_layer
        self.media: torch.Tensor | None = None
        self.text_time: torch.Tensor | None = None
        self.media_valid: torch.Tensor | None = None
        self.media_bias: torch.Tensor | None = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            if "decoder_layer" in modules:
                return getattr(modules["decoder_layer"], name)
            raise

    def condition(self, media, text_time, media_valid, media_bias=None) -> None:
        self.media = media
        self.text_time = text_time
        self.media_valid = media_valid
        self.media_bias = media_bias

    def clear_conditioning(self) -> None:
        self.media = None
        self.text_time = None
        self.media_valid = None
        self.media_bias = None

    def forward(self, hidden_states=None, *args, **kwargs):
        if hidden_states is None:
            hidden_states = kwargs.pop("hidden_states")
        if self.media is not None:
            hidden_states = self.gated_xattn(
                hidden_states,
                self.media,
                self.text_time,
                self.media_valid,
                self.media_bias,
            )
        return self.decoder_layer(hidden_states, *args, **kwargs)


def _get_decoder_layers(language_model: nn.Module) -> nn.ModuleList:
    """Locate the ``nn.ModuleList`` of decoder layers inside an HF causal LM."""
    for attr in ("model", "transformer", "gpt_neox"):
        base = getattr(language_model, attr, None)
        if base is not None:
            layers = getattr(base, "layers", None) or getattr(base, "h", None)
            if isinstance(layers, nn.ModuleList):
                return layers
            inner = getattr(base, "model", None)
            if inner is not None and isinstance(getattr(inner, "layers", None), nn.ModuleList):
                return inner.layers
    layers = getattr(language_model, "layers", None)
    if isinstance(layers, nn.ModuleList):
        return layers
    raise ValueError(
        f"Could not locate decoder layers on {type(language_model).__name__}; "
        "Flamingo conditioning needs an nn.ModuleList of decoder layers."
    )


def add_gated_cross_attention(
    language_model: nn.Module,
    lm_hidden_size: int,
    media_hidden_size: int,
    every_n_layers: int = 4,
    dim_head: int = 64,
    heads: int = 8,
    ff_mult: int = 4,
    only_attend_immediate_media: bool = True,
) -> tuple[nn.ModuleList, list[int]]:
    """Interleave gated xattn blocks into a frozen causal LM, in place.

    A block is inserted before decoder layers ``0, n, 2n, ...`` so the very
    first layer is already vision-conditioned and gradients reach every block.

    Returns:
        ``(xattn_blocks, layer_indices)`` — the trainable blocks and where they
        were inserted.  The caller owns the returned ``ModuleList``; the
        wrapped layers only hold unregistered references to its entries.
    """
    layers = _get_decoder_layers(language_model)
    indices = list(range(0, len(layers), every_n_layers))

    blocks = nn.ModuleList()
    for i in indices:
        block = GatedCrossAttentionBlock(
            dim=lm_hidden_size,
            dim_media=media_hidden_size,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        layers[i] = FlamingoLayer(block, layers[i])
        blocks.append(block)

    return blocks, indices


def condition_layers(
    language_model: nn.Module, media, text_time, media_valid, media_bias=None
) -> None:
    """Attach media conditioning to every :class:`FlamingoLayer` in the LM."""
    for module in language_model.modules():
        if isinstance(module, FlamingoLayer):
            module.condition(media, text_time, media_valid, media_bias)


def clear_layer_conditioning(language_model: nn.Module) -> None:
    """Drop media conditioning from every :class:`FlamingoLayer` in the LM."""
    for module in language_model.modules():
        if isinstance(module, FlamingoLayer):
            module.clear_conditioning()
