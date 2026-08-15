"""SoftWhere multi-foveal visual resampler.

Adapted from the SoftWhere research prototype
(https://github.com/engichang1467/softwhere), which replaces LookWhere's hard,
single-map selector with a sampling-free, multi-foveal TokenLearner selector.
The TokenLearner modules themselves are a PyTorch port of the Scenic reference
implementation (https://github.com/google-research/scenic).

In Flamingo the frozen vision grid is compressed by a Perceiver Resampler with
``R`` learned latent queries.  SoftWhere replaces those latents with ``S``
learned *foveae*: TokenLearner emits S soft spatial attention maps over the
frozen vision grid, each map pools one visual token, and a diversity penalty
keeps the maps from collapsing onto the same region.

On top of the S soft tokens the resampler can also forward ``K`` raw patch
tokens chosen by SoftWhere's best-performing selection policy (per-map top-k
with spatial non-maximum suppression).  Selection is discrete, so the kept
patches carry a straight-through gate derived from the foveal maps; that keeps
the selector trainable from the language-modelling loss exactly as in
SoftWhere's mini end-to-end experiment.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TokenLearner modules (ported from Open-TokenLearner / Scenic)
# ---------------------------------------------------------------------------


class TokenLearnerV10(nn.Module):
    """TokenLearner v1.0 — four 3x3 convs + sigmoid gating over a square grid.

    Produces ``num_tokens`` independent spatial attention maps.  Because the
    maps are sigmoid-gated (not softmax-normalised over space) they stay
    blobby and spatially distinct, which is what makes them usable as foveae.
    """

    def __init__(self, in_channels: int, num_tokens: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(in_channels)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, num_tokens, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """Args: x ``(B, N, C)``; returns attention maps ``(B, S, N)``."""
        b, n, c = x.shape
        h, w = grid_hw
        attn = self.norm(x).transpose(1, 2).reshape(b, c, h, w)
        for i, conv in enumerate(self.convs):
            attn = conv(attn)
            if i < len(self.convs) - 1:
                attn = F.gelu(attn, approximate="tanh")
        attn = torch.sigmoid(attn)
        return attn.reshape(b, self.num_tokens, n)


class TokenLearnerV11(nn.Module):
    """TokenLearner v1.1 — MLP + softmax over space.

    Grid-free: works on any token set, so it is the variant used with
    resolution-adaptive encoders such as MoonViT.
    """

    def __init__(self, in_channels: int, num_tokens: int, bottleneck_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, num_tokens)

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int] | None = None) -> torch.Tensor:
        """Args: x ``(B, N, C)``; returns attention maps ``(B, S, N)``."""
        h = F.gelu(self.fc1(self.norm(x)), approximate="tanh")
        attn = self.fc2(h).transpose(-1, -2)  # (B, S, N)
        return F.softmax(attn, dim=-1)


# ---------------------------------------------------------------------------
# SoftWhere diagnostics / selection policies
# ---------------------------------------------------------------------------


def pairwise_overlap(attn: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal histogram intersection between the S foveal maps.

    0 means the foveae are disjoint, 1 means they are identical.  Used as the
    diversity regulariser that stops multi-foveal selection from collapsing to
    a single map.

    Args:
        attn: ``(B, S, N)`` non-negative maps.
    """
    b, s, n = attn.shape
    if s < 2:
        return attn.new_zeros(())
    p = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    inter = torch.minimum(p.unsqueeze(2), p.unsqueeze(1)).sum(-1)  # (B, S, S)
    eye = torch.eye(s, device=attn.device, dtype=torch.bool)
    return inter.masked_select(~eye).mean()


@torch.no_grad()
def per_map_nms_select(
    maps: torch.Tensor,
    k: int,
    grid_hw: tuple[int, int],
    min_dist: int = 2,
) -> torch.Tensor:
    """SoftWhere's per-map top-k + spatial NMS selection policy.

    Each foveal map gets an equal share of the patch budget and greedily takes
    its highest-scoring patch, after which every patch within Chebyshev
    distance ``min_dist`` is suppressed for *all* maps.  This is the policy
    that gave the best small-object coverage in the SoftWhere ablation.

    Args:
        maps: ``(B, S, N)`` non-negative foveal maps.
        k: number of patches to keep.
        grid_hw: spatial layout of the N patches.
        min_dist: Chebyshev suppression radius in patch units (1 = dedupe only).

    Returns:
        ``(B, k)`` long tensor of patch indices, sorted per row.
    """
    b, s, n = maps.shape
    h, w = grid_hw
    device = maps.device

    ys = torch.arange(h, device=device).repeat_interleave(w)
    xs = torch.arange(w, device=device).repeat(h)

    scores = maps.float()
    blocked = torch.zeros(b, n, dtype=torch.bool, device=device)
    chosen = torch.zeros(b, n, dtype=torch.bool, device=device)
    picks: list[torch.Tensor] = []

    per_map = max(1, math.ceil(k / s))
    neg = torch.finfo(torch.float32).min

    for step in range(k):
        m = min(step // per_map, s - 1)
        # If suppression exhausted a row, fall back to only blocking picks.
        exhausted = blocked.all(dim=-1, keepdim=True)
        mask = torch.where(exhausted, chosen, blocked)
        idx = scores[:, m, :].masked_fill(mask, neg).argmax(dim=-1)  # (B,)
        picks.append(idx)

        dy = (ys[idx].unsqueeze(1) - ys.unsqueeze(0)).abs()
        dx = (xs[idx].unsqueeze(1) - xs.unsqueeze(0)).abs()
        blocked = blocked | (torch.maximum(dy, dx) < min_dist)
        chosen = chosen.scatter(1, idx.unsqueeze(1), True)
        blocked = blocked | chosen

    return torch.stack(picks, dim=1).sort(dim=-1).values


# ---------------------------------------------------------------------------
# Resampler
# ---------------------------------------------------------------------------


class SoftWhereResampler(nn.Module):
    """Multi-foveal TokenLearner resampler for Flamingo-style conditioning.

    Data flow (SigLIP defaults, S=8 foveae, K=32 kept patches)::

        (B, 729, 1152)  frozen vision grid
        -> TokenLearner -> (B, 8, 729) foveal attention maps
        -> attention-weighted pooling  -> (B, 8, 1152)   soft foveal tokens
        -> per-map NMS top-k selection -> (B, 32, 1152)  gated patch tokens
        -> concat                      -> (B, 40, 1152)
        -> SwiGLU MLP projection       -> (B, 40, D_llm) media tokens
    """

    def __init__(self, config):
        super().__init__()
        self.num_foveal_tokens = config.num_foveal_tokens
        self.variant = config.softwhere_variant
        self.agg = config.softwhere_agg
        self.topk_patches = config.softwhere_topk_patches
        self.nms_min_dist = config.softwhere_nms_min_dist
        self.grid_size = config.vision_grid_size

        if self.variant not in ("v10", "v11"):
            raise ValueError(
                f"softwhere_variant must be 'v10' or 'v11', got '{self.variant}'"
            )
        if self.agg not in ("max", "mean", "logsumexp"):
            raise ValueError(
                f"softwhere_agg must be 'max', 'mean' or 'logsumexp', got '{self.agg}'"
            )

        tl_cls = TokenLearnerV10 if self.variant == "v10" else TokenLearnerV11
        self.token_learner = tl_cls(
            in_channels=config.vision_hidden_size,
            num_tokens=self.num_foveal_tokens,
        )

        self.norm_media = nn.LayerNorm(config.vision_hidden_size)
        self.linear_1 = nn.Linear(
            config.vision_hidden_size, config.connector_intermediate_size, bias=True
        )
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(
            config.connector_intermediate_size // 2, config.llm_hidden_size, bias=True
        )

    @property
    def num_media_tokens(self) -> int:
        return self.num_foveal_tokens + self.topk_patches

    def _aggregate(self, attn: torch.Tensor) -> torch.Tensor:
        """Collapse the S foveal maps into one importance map ``(B, N)``."""
        if self.agg == "max":
            return attn.amax(dim=1)
        if self.agg == "mean":
            return attn.mean(dim=1)
        return torch.logsumexp(attn, dim=1)

    def _infer_grid(self, num_patches: int) -> tuple[int, int]:
        if self.grid_size and self.grid_size**2 == num_patches:
            return self.grid_size, self.grid_size
        side = int(math.isqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(
                f"SoftWhere needs a square patch grid to run the v10 selector / NMS "
                f"policy, got {num_patches} patches. Use softwhere_variant='v11' with "
                f"softwhere_topk_patches=0 for non-square encoders."
            )
        return side, side

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.norm_media(tokens)
        hidden = self.linear_1(tokens)
        x, gate = hidden.chunk(2, dim=-1)
        return self.linear_2(self.act(gate) * x)

    def forward(self, vision_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Resample a frozen vision grid into a compact set of media tokens.

        Args:
            vision_features: ``(B, N, vision_hidden_size)`` patch features.

        Returns:
            Dict with ``media_tokens`` ``(B, S + K, llm_hidden_size)``,
            ``media_bias`` ``(B, S + K)`` straight-through attention bias,
            ``attn`` ``(B, S, N)`` foveal maps, ``diversity_loss`` scalar and
            ``patch_indices`` ``(B, K)`` (or ``None`` when K == 0).
        """
        b, n, _ = vision_features.shape

        grid_hw = None
        if self.variant == "v10" or self.topk_patches > 0:
            grid_hw = self._infer_grid(n)

        attn = self.token_learner(vision_features, grid_hw)  # (B, S, N)

        # Soft foveal tokens: attention-weighted average pooling.  Averaging
        # (rather than SoftWhere's sum pooling) keeps token norms in the range
        # the frozen LLM expects, which matters because the media tokens are
        # consumed by cross-attention rather than a classifier head.
        weights = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        foveal_tokens = torch.einsum("bsn,bnd->bsd", weights, vision_features)

        tokens = foveal_tokens
        patch_indices = None
        media_bias = None
        if self.topk_patches > 0:
            importance = self._aggregate(attn)  # (B, N)
            patch_indices = per_map_nms_select(
                attn.detach(), self.topk_patches, grid_hw, self.nms_min_dist
            )
            batch_range = torch.arange(b, device=vision_features.device).unsqueeze(1)
            patch_tokens = vision_features[batch_range, patch_indices]  # (B, K, D)
            tokens = torch.cat([foveal_tokens, patch_tokens], dim=1)

            # Straight-through keep-gate (SoftWhere's trick), expressed as an
            # additive bias on the cross-attention logits rather than a scale
            # on the token itself.  A multiplicative gate is *not* usable here:
            # both this module's `norm_media` and the LayerNorm in
            # MaskedCrossAttention are scale-invariant, so they would absorb it
            # exactly and the selector would receive no gradient at all.
            # Biasing the logits by log(gate) is equivalent to multiplying the
            # attention mass of a patch by its importance, so the LM loss can
            # say "attend more/less here" and the gradient reaches the
            # TokenLearner that chose the patch.  `softplus` keeps the gate
            # positive for every aggregation (logsumexp can be negative), and
            # subtracting the detached copy makes the bias exactly 0.0 in the
            # forward pass — selection stays hard, only the gradient flows.
            gate = importance[batch_range, patch_indices]
            log_gate = torch.log(F.softplus(gate) + 1e-6)
            patch_bias = log_gate - log_gate.detach()
            media_bias = torch.cat(
                [patch_bias.new_zeros(b, self.num_foveal_tokens), patch_bias], dim=1
            )

        return {
            "media_tokens": self._project(tokens),
            "media_bias": media_bias,
            "attn": attn,
            "diversity_loss": pairwise_overlap(attn),
            "patch_indices": patch_indices,
        }
