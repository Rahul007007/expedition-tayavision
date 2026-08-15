"""Tests for the SoftWhere multi-foveal resampler."""

import pytest
import torch

from config.model_config import TinyAyaFlamingoConfig
from src.flamingo import PerceiverResampler
from src.softwhere import (
    SoftWhereResampler,
    pairwise_overlap,
    per_map_nms_select,
)


@pytest.fixture
def tiny_config():
    return TinyAyaFlamingoConfig(
        vision_hidden_size=32,
        vision_grid_size=8,
        num_vision_tokens=64,
        llm_hidden_size=16,
        connector_intermediate_size=32,
        num_foveal_tokens=4,
        softwhere_topk_patches=8,
        softwhere_nms_min_dist=2,
        num_latent_tokens=6,
        perceiver_depth=1,
        xattn_head_dim=8,
        xattn_num_heads=2,
        torch_dtype="float32",
    )


@pytest.fixture
def vision_features():
    torch.manual_seed(0)
    return torch.randn(3, 64, 32)


class TestPairwiseOverlap:
    def test_identical_maps_overlap_fully(self):
        m = torch.rand(2, 4, 25).abs() + 1e-3
        attn = m[:, :1].expand(-1, 4, -1).contiguous()
        assert pairwise_overlap(attn).item() == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_maps_have_zero_overlap(self):
        attn = torch.zeros(1, 2, 8)
        attn[0, 0, :4] = 1.0
        attn[0, 1, 4:] = 1.0
        assert pairwise_overlap(attn).item() == pytest.approx(0.0, abs=1e-6)

    def test_single_map_is_zero(self):
        assert pairwise_overlap(torch.rand(1, 1, 8)).item() == 0.0


class TestPerMapNMS:
    def test_indices_are_unique_and_in_range(self):
        torch.manual_seed(0)
        maps = torch.rand(4, 4, 64)
        idx = per_map_nms_select(maps, k=8, grid_hw=(8, 8), min_dist=2)

        assert idx.shape == (4, 8)
        assert idx.min() >= 0 and idx.max() < 64
        for row in idx:
            assert len(set(row.tolist())) == 8

    def test_selection_respects_min_distance(self):
        torch.manual_seed(0)
        maps = torch.rand(2, 2, 64)
        min_dist = 3
        idx = per_map_nms_select(maps, k=6, grid_hw=(8, 8), min_dist=min_dist)

        for row in idx:
            coords = [(int(i) // 8, int(i) % 8) for i in row]
            for a in range(len(coords)):
                for b in range(a + 1, len(coords)):
                    cheb = max(
                        abs(coords[a][0] - coords[b][0]),
                        abs(coords[a][1] - coords[b][1]),
                    )
                    assert cheb >= min_dist

    def test_picks_the_global_peak_first(self):
        maps = torch.zeros(1, 1, 16)
        maps[0, 0, 9] = 1.0
        idx = per_map_nms_select(maps, k=1, grid_hw=(4, 4), min_dist=1)
        assert idx[0, 0].item() == 9

    def test_budget_is_shared_across_maps(self):
        """Each foveal map should claim a patch inside its own peak region."""
        maps = torch.zeros(1, 2, 64)
        maps[0, 0, 0] = 1.0   # map 0 peaks top-left
        maps[0, 1, 63] = 1.0  # map 1 peaks bottom-right
        idx = per_map_nms_select(maps, k=2, grid_hw=(8, 8), min_dist=2)
        assert set(idx[0].tolist()) == {0, 63}


class TestSoftWhereResampler:
    def test_output_shapes(self, tiny_config, vision_features):
        resampler = SoftWhereResampler(tiny_config)
        out = resampler(vision_features)

        assert resampler.num_media_tokens == 12  # 4 foveal + 8 patches
        assert out["media_tokens"].shape == (3, 12, 16)
        assert out["attn"].shape == (3, 4, 64)
        assert out["patch_indices"].shape == (3, 8)
        assert out["diversity_loss"].ndim == 0

    def test_foveal_only_mode(self, tiny_config, vision_features):
        tiny_config.softwhere_topk_patches = 0
        resampler = SoftWhereResampler(tiny_config)
        out = resampler(vision_features)

        assert resampler.num_media_tokens == 4
        assert out["media_tokens"].shape == (3, 4, 16)
        assert out["patch_indices"] is None

    def test_v11_variant_is_grid_free(self, tiny_config):
        tiny_config.softwhere_variant = "v11"
        tiny_config.softwhere_topk_patches = 0
        resampler = SoftWhereResampler(tiny_config)

        out = resampler(torch.randn(2, 37, 32))  # non-square token count
        assert out["media_tokens"].shape == (2, 4, 16)
        # v11 maps are softmax-normalised over space.
        assert torch.allclose(out["attn"].sum(-1), torch.ones(2, 4), atol=1e-5)

    def test_v10_requires_square_grid(self, tiny_config):
        resampler = SoftWhereResampler(tiny_config)
        with pytest.raises(ValueError, match="square patch grid"):
            resampler(torch.randn(1, 37, 32))

    def test_rejects_unknown_variant(self, tiny_config):
        tiny_config.softwhere_variant = "v99"
        with pytest.raises(ValueError, match="softwhere_variant"):
            SoftWhereResampler(tiny_config)

    def test_straight_through_gate_is_identity_in_forward(self, tiny_config, vision_features):
        """Kept patches must keep their exact features (gate ratio == 1)."""
        tiny_config.num_foveal_tokens = 4
        resampler = SoftWhereResampler(tiny_config)
        out = resampler(vision_features)

        idx = out["patch_indices"]
        batch_range = torch.arange(3).unsqueeze(1)
        expected = resampler._project(
            torch.cat(
                [
                    torch.einsum(
                        "bsn,bnd->bsd",
                        out["attn"] / (out["attn"].sum(-1, keepdim=True) + 1e-6),
                        vision_features,
                    ),
                    vision_features[batch_range, idx],
                ],
                dim=1,
            )
        )
        assert torch.allclose(out["media_tokens"], expected, atol=1e-5)

    def test_gradients_reach_the_selector(self, tiny_config, vision_features):
        """The LM-side loss must train the TokenLearner, including via the
        straight-through gate on hard-selected patches."""
        resampler = SoftWhereResampler(tiny_config)
        resampler(vision_features)["media_tokens"].sum().backward()

        grads = [
            p.grad for p in resampler.token_learner.parameters() if p.grad is not None
        ]
        assert grads, "TokenLearner received no gradient"
        assert any(g.abs().sum() > 0 for g in grads)

    def test_keep_gate_bias_is_zero_in_forward(self, tiny_config, vision_features):
        """Selection stays hard: the straight-through bias must not perturb
        the forward pass at all."""
        resampler = SoftWhereResampler(tiny_config)
        bias = resampler(vision_features)["media_bias"]

        assert bias.shape == (3, resampler.num_media_tokens)
        assert torch.equal(bias, torch.zeros_like(bias))

    def test_keep_gate_bias_carries_selector_gradient_at_any_feature_scale(
        self, tiny_config, vision_features
    ):
        """Regression test for the keep-gate.

        The gate used to be a multiplicative scale on the kept patch tokens,
        which the LayerNorms in `_project` and in MaskedCrossAttention absorb
        exactly (LayerNorm is scale-invariant), leaving the selector with a
        gradient of ~1e-6 that shrank as 1/var(features).  Expressed as a bias
        on the attention logits it survives, and is scale-independent.
        """
        resampler = SoftWhereResampler(tiny_config).double()
        n_foveal = resampler.num_foveal_tokens

        magnitudes = []
        for scale in (1.0, 20.0):
            out = resampler(vision_features.double() * scale)
            # Backprop from the patch branch alone, so the foveal path (which
            # has always been differentiable) cannot mask the result.  The
            # readout must be linear: the bias is exactly 0.0, so a quadratic
            # one would have zero derivative there for trivial reasons.
            out["media_bias"][:, n_foveal:].sum().backward()
            magnitudes.append(
                sum(
                    p.grad.abs().sum().item()
                    for p in resampler.token_learner.parameters()
                    if p.grad is not None
                )
            )
            resampler.zero_grad(set_to_none=True)

        assert all(m > 1e-3 for m in magnitudes), magnitudes
        # Scale-invariance: a LayerNorm-absorbed gate decays as 1/var instead.
        assert magnitudes[1] > 0.1 * magnitudes[0]

    @pytest.mark.parametrize("agg", ["max", "mean", "logsumexp"])
    def test_every_aggregation_keeps_the_gate_differentiable(self, tiny_config, vision_features, agg):
        """logsumexp importance can be negative, which a clamp would silently
        turn into a zero gradient; softplus keeps every aggregation usable."""
        tiny_config.softwhere_agg = agg
        resampler = SoftWhereResampler(tiny_config)
        out = resampler(vision_features)
        out["media_bias"][:, resampler.num_foveal_tokens :].sum().backward()

        total = sum(
            p.grad.abs().sum().item()
            for p in resampler.token_learner.parameters()
            if p.grad is not None
        )
        assert torch.isfinite(torch.tensor(total)) and total > 0

    def test_diversity_loss_penalises_collapsed_maps(self, tiny_config, vision_features):
        resampler = SoftWhereResampler(tiny_config)
        loss = resampler(vision_features)["diversity_loss"]
        assert 0.0 <= loss.item() <= 1.0


class TestPerceiverResampler:
    def test_output_shape(self, tiny_config, vision_features):
        tiny_config.resampler_type = "perceiver"
        resampler = PerceiverResampler(tiny_config)
        out = resampler(vision_features)

        assert resampler.num_media_tokens == 6
        assert out["media_tokens"].shape == (3, 6, 16)
        assert out["diversity_loss"].item() == 0.0

    def test_handles_variable_token_counts(self, tiny_config):
        tiny_config.resampler_type = "perceiver"
        resampler = PerceiverResampler(tiny_config)
        assert resampler(torch.randn(1, 17, 32))["media_tokens"].shape == (1, 6, 16)
