"""Tests for the Flamingo-style Tiny Aya Vision model.

All tests build a tiny randomly-initialised model (no HF downloads) so they
run on CPU in a few seconds.
"""

import pytest
import torch
from transformers import Cohere2Config, SiglipVisionConfig

from config.model_config import TinyAyaFlamingoConfig
from models.tiny_aya_flamingo import TinyAyaFlamingoForConditionalGeneration
from src.flamingo import FlamingoLayer
from src.processing import TinyAyaFlamingoProcessor

IMAGE_TOKEN_ID = 199
VOCAB_SIZE = 200


def build_config(**overrides) -> TinyAyaFlamingoConfig:
    text_config = Cohere2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128,
        sliding_window=32,
    ).to_dict()
    vision_tower_config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=64,
        patch_size=8,
    ).to_dict()

    kwargs = dict(
        vision_hidden_size=32,
        image_size=64,
        patch_size=8,
        vision_grid_size=8,
        num_vision_tokens=64,
        llm_hidden_size=64,
        connector_intermediate_size=64,
        num_foveal_tokens=4,
        softwhere_topk_patches=8,
        xattn_head_dim=8,
        xattn_num_heads=2,
        cross_attn_every_n_layers=2,
        image_token_id=IMAGE_TOKEN_ID,
        torch_dtype="float32",
        text_config=text_config,
        vision_tower_config=vision_tower_config,
    )
    kwargs.update(overrides)
    return TinyAyaFlamingoConfig(**kwargs)


def build_model(open_gates: bool = False, **overrides):
    torch.manual_seed(0)
    model = TinyAyaFlamingoForConditionalGeneration(build_config(**overrides)).eval()
    if open_gates:
        with torch.no_grad():
            for block in model.xattn_blocks:
                block.attn_gate.fill_(1.0)
                block.ff_gate.fill_(1.0)
    return model


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture(scope="module")
def open_gate_model():
    return build_model(open_gates=True)


class TestAssembly:
    def test_xattn_blocks_are_interleaved(self, model):
        assert model.xattn_layer_indices == [0, 2]
        layers = model.language_model.model.layers
        assert isinstance(layers[0], FlamingoLayer)
        assert not isinstance(layers[1], FlamingoLayer)

    def test_wrapper_delegates_unknown_attributes(self, model):
        """transformers reads e.g. ``layer.attention_type`` off the layer."""
        layer = model.language_model.model.layers[0]
        assert layer.attention_type == layer.decoder_layer.attention_type

    def test_media_token_count(self, model):
        assert model.num_media_tokens == 12  # 4 foveal + 8 NMS-selected patches
        assert model.config.num_media_tokens == 12

    def test_perceiver_variant_assembles(self):
        m = build_model(resampler_type="perceiver", num_latent_tokens=6, perceiver_depth=1)
        out = m(
            input_ids=torch.tensor([[5, IMAGE_TOKEN_ID, 7]]),
            pixel_values=torch.randn(1, 3, 64, 64),
        )
        assert m.num_media_tokens == 6
        assert out.image_hidden_states.shape == (1, 6, 64)


class TestGating:
    def test_zero_gates_reproduce_the_frozen_llm(self, model):
        """Flamingo's defining property: at init the VLM *is* the frozen LLM."""
        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8, 9]])
        pixel_values = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            vlm_logits = model(input_ids=ids, pixel_values=pixel_values).logits
            lm_logits = model.language_model(
                inputs_embeds=model.get_input_embeddings()(ids)
            ).logits

        assert torch.allclose(vlm_logits, lm_logits, atol=1e-5)

    def test_open_gates_change_the_output(self, open_gate_model):
        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8, 9]])
        pixel_values = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            vlm_logits = open_gate_model(input_ids=ids, pixel_values=pixel_values).logits
            lm_logits = open_gate_model.language_model(
                inputs_embeds=open_gate_model.get_input_embeddings()(ids)
            ).logits

        assert not torch.allclose(vlm_logits, lm_logits, atol=1e-4)

    def test_gate_values_report_zero_at_init(self, model):
        values = model.gate_values()
        assert set(values) == {
            "gates/attn_layer_0", "gates/ff_layer_0",
            "gates/attn_layer_2", "gates/ff_layer_2",
        }
        assert all(v == 0.0 for v in values.values())


class TestMediaMasking:
    def _attn_only_model(self):
        model = build_model()
        with torch.no_grad():
            for block in model.xattn_blocks:
                block.attn_gate.fill_(1.0)  # ff_gate stays 0 to isolate xattn
        return model

    def test_text_before_the_marker_is_untouched(self):
        model = self._attn_only_model()
        ids = torch.tensor([[5, 6, IMAGE_TOKEN_ID, 7, 8]])
        pixel_values = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            vlm_logits = model(input_ids=ids, pixel_values=pixel_values).logits
            lm_logits = model.language_model(
                inputs_embeds=model.get_input_embeddings()(ids)
            ).logits

        delta = (vlm_logits - lm_logits).abs().amax(-1)[0]
        assert torch.allclose(delta[:2], torch.zeros(2), atol=1e-7)
        assert delta[2:].min() > 0

    def test_only_attends_the_most_recent_image(self):
        model = self._attn_only_model()
        ids = torch.tensor([[IMAGE_TOKEN_ID, 5, 6, IMAGE_TOKEN_ID, 7, 8]])
        torch.manual_seed(1)
        pixels = torch.randn(2, 3, 64, 64)
        other = pixels.clone()
        other[1] = torch.randn(3, 64, 64)  # change only the *second* image

        with torch.no_grad():
            a = model(input_ids=ids, pixel_values=pixels).logits
            b = model(input_ids=ids, pixel_values=other).logits

        delta = (a - b).abs().amax(-1)[0]
        # Positions 0-2 follow image 1 only, so swapping image 2 cannot move them.
        assert torch.allclose(delta[:3], torch.zeros(3), atol=1e-7)
        assert delta[3:].min() > 0

    def test_attend_all_previous_media_mode(self):
        model = build_model(only_attend_immediate_media=False)
        with torch.no_grad():
            for block in model.xattn_blocks:
                block.attn_gate.fill_(1.0)

        ids = torch.tensor([[IMAGE_TOKEN_ID, 5, 6, IMAGE_TOKEN_ID, 7, 8]])
        torch.manual_seed(1)
        pixels = torch.randn(2, 3, 64, 64)
        other = pixels.clone()
        other[0] = torch.randn(3, 64, 64)  # change the *first* image

        with torch.no_grad():
            a = model(input_ids=ids, pixel_values=pixels).logits
            b = model(input_ids=ids, pixel_values=other).logits

        # With cumulative attention, later text still sees the first image.
        assert (a - b).abs().amax(-1)[0][-1] > 0

    def test_marker_count_must_match_image_count(self, model):
        with pytest.raises(ValueError, match="marker"):
            model(
                input_ids=torch.tensor([[5, IMAGE_TOKEN_ID, 7]]),
                pixel_values=torch.randn(2, 3, 64, 64),
            )

    def test_images_are_routed_to_the_right_sample(self):
        """Sample 0 has two images, sample 1 has one — slots must not cross."""
        model = self._attn_only_model()
        ids = torch.tensor(
            [
                [IMAGE_TOKEN_ID, 5, IMAGE_TOKEN_ID, 6],
                [7, IMAGE_TOKEN_ID, 8, 9],
            ]
        )
        torch.manual_seed(2)
        pixels = torch.randn(3, 3, 64, 64)
        other = pixels.clone()
        other[2] = torch.randn(3, 64, 64)  # sample 1's image

        with torch.no_grad():
            a = model(input_ids=ids, pixel_values=pixels).logits
            b = model(input_ids=ids, pixel_values=other).logits

        delta = (a - b).abs().amax(-1)
        assert torch.allclose(delta[0], torch.zeros(4), atol=1e-7)  # sample 0 untouched
        assert delta[1][1:].min() > 0  # sample 1 changed after its marker


class TestTextOnly:
    def test_text_only_forward(self, open_gate_model):
        out = open_gate_model(input_ids=torch.tensor([[5, 6, 7]]))
        assert out.logits.shape == (1, 3, VOCAB_SIZE)

    def test_text_only_equals_the_bare_llm_attention_branch(self):
        """No media -> the cross-attention contribution must be exactly zero."""
        model = build_model()
        with torch.no_grad():
            for block in model.xattn_blocks:
                block.attn_gate.fill_(1.0)

        ids = torch.tensor([[5, 6, 7]])
        with torch.no_grad():
            vlm_logits = model(input_ids=ids).logits
            lm_logits = model.language_model(
                inputs_embeds=model.get_input_embeddings()(ids)
            ).logits
        assert torch.allclose(vlm_logits, lm_logits, atol=1e-6)

    def test_text_only_training_step_keeps_resampler_in_the_graph(self):
        """DDP safety: every rank must produce resampler gradients."""
        model = build_model()
        model.configure_trainable_parameters()
        model.train()

        out = model(input_ids=torch.tensor([[5, 6, 7]]), labels=torch.tensor([[5, 6, 7]]))
        out.loss.backward()

        assert all(p.grad is not None for p in model.resampler.parameters())


class TestTraining:
    def test_only_flamingo_parameters_are_trainable(self):
        model = build_model()
        params = model.configure_trainable_parameters()

        assert all(not p.requires_grad for p in model.vision_encoder.parameters())
        assert all(p.requires_grad for p in model.resampler.parameters())
        assert all(p.requires_grad for p in model.xattn_blocks.parameters())

        assert all(not p.requires_grad for p in model.language_model.parameters())
        assert model.media_marker_delta.requires_grad
        assert any(p is model.media_marker_delta for p in params)

        # The LLM's embedding matrix must never enter the optimizer: it is
        # 262144 x 2048 for tiny-aya-global, so training it to update one row
        # would all-reduce a 1 GiB gradient per micro-step.
        embed = model.get_input_embeddings().weight
        assert not embed.requires_grad
        assert not any(p is embed for p in params)

    def test_marker_delta_trains_while_the_vocabulary_stays_frozen(self):
        model = build_model()
        model.configure_trainable_parameters()
        model.train()

        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7]])
        out = model(input_ids=ids, pixel_values=torch.randn(1, 3, 64, 64), labels=ids)
        out.loss.backward()

        assert model.media_marker_delta.grad.abs().sum() > 0
        assert model.get_input_embeddings().weight.grad is None

    def test_marker_delta_gets_a_gradient_on_text_only_batches(self):
        """DDP would hang if a trainable parameter were skipped by some ranks."""
        model = build_model()
        model.configure_trainable_parameters()
        model.train()

        ids = torch.tensor([[5, 6, 7]])
        model(input_ids=ids, labels=ids).loss.backward()

        assert model.media_marker_delta.grad is not None
        assert model.media_marker_delta.grad.abs().sum() == 0

    def test_marker_delta_shifts_the_marker_embedding_only(self):
        model = build_model(open_gates=True)
        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7]])
        px = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            before = model(input_ids=ids, pixel_values=px).logits
            # Must not be a constant vector: the LLM is pre-norm, so a pure DC
            # offset is mean-subtracted away by the first LayerNorm.
            torch.manual_seed(3)
            model.media_marker_delta.normal_(std=0.5)
            after = model(input_ids=ids, pixel_values=px).logits
            text_only_before = model(input_ids=torch.tensor([[5, 6, 7]])).logits
            model.media_marker_delta.zero_()
            text_only_after = model(input_ids=torch.tensor([[5, 6, 7]])).logits

        assert not torch.allclose(before, after, atol=1e-4)
        assert torch.allclose(text_only_before, text_only_after, atol=1e-6)

    def test_backward_produces_flamingo_gradients(self):
        model = build_model()
        model.configure_trainable_parameters()
        model.train()

        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8]])
        out = model(input_ids=ids, pixel_values=torch.randn(1, 3, 64, 64), labels=ids)
        (out.loss + 0.1 * out.diversity_loss).backward()

        assert all(p.grad is not None for p in model.xattn_blocks.parameters())
        assert all(p.grad is not None for p in model.resampler.parameters())
        assert all(p.grad is None for p in model.vision_encoder.parameters())

    def test_diversity_loss_is_returned(self, model):
        out = model(
            input_ids=torch.tensor([[5, IMAGE_TOKEN_ID, 7]]),
            pixel_values=torch.randn(1, 3, 64, 64),
        )
        assert 0.0 <= out.diversity_loss.item() <= 1.0
        assert out.foveal_maps.shape == (1, 4, 64)


class TestGeneration:
    def test_incremental_decoding_matches_a_full_forward(self, open_gate_model):
        from transformers import DynamicCache

        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8, 9, 11]])
        pixel_values = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            full = open_gate_model(input_ids=ids, pixel_values=pixel_values).logits
            cache = DynamicCache(
                config=open_gate_model.config.get_text_config(decoder=True)
            )
            prefill = open_gate_model(
                input_ids=ids[:, :-1],
                pixel_values=pixel_values,
                past_key_values=cache,
                use_cache=True,
            )
            step = open_gate_model(
                input_ids=ids[:, -1:],
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )

        assert torch.allclose(full[:, -1], step.logits[:, -1], atol=1e-5)

    def test_generate_with_and_without_cache_agree(self, open_gate_model):
        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8, 9]])
        pixel_values = torch.randn(1, 3, 64, 64)

        cached = open_gate_model.generate(
            input_ids=ids, pixel_values=pixel_values,
            max_new_tokens=6, do_sample=False, use_cache=True,
        )
        uncached = open_gate_model.generate(
            input_ids=ids, pixel_values=pixel_values,
            max_new_tokens=6, do_sample=False, use_cache=False,
        )

        assert cached.shape == (1, ids.shape[1] + 6)
        assert torch.equal(cached, uncached)

    def test_media_state_is_cleared_after_generate(self, open_gate_model):
        open_gate_model.generate(
            input_ids=torch.tensor([[5, IMAGE_TOKEN_ID, 7]]),
            pixel_values=torch.randn(1, 3, 64, 64),
            max_new_tokens=2,
            do_sample=False,
        )
        assert open_gate_model._media_state is None
        assert all(
            layer.media is None
            for layer in open_gate_model.language_model.modules()
            if isinstance(layer, FlamingoLayer)
        )


class TestProcessor:
    def test_one_marker_token_per_image(self):
        assert TinyAyaFlamingoProcessor._tokens_per_image(None, None, 3) == [1, 1, 1]

    def test_placeholder_is_not_expanded(self):
        processor = TinyAyaFlamingoProcessor.__new__(TinyAyaFlamingoProcessor)
        processor.image_token = "<image>"
        assert processor.image_placeholder == "<image>"


class TestSerialization:
    def test_save_and_reload_round_trip(self, tmp_path):
        model = build_model(open_gates=True)
        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8]])
        pixel_values = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            before = model(input_ids=ids, pixel_values=pixel_values).logits

        model.save_pretrained(tmp_path)
        reloaded = TinyAyaFlamingoForConditionalGeneration.from_pretrained(tmp_path).eval()

        with torch.no_grad():
            after = reloaded(input_ids=ids, pixel_values=pixel_values).logits

        assert reloaded.num_media_tokens == model.num_media_tokens
        assert reloaded.xattn_layer_indices == model.xattn_layer_indices
        assert torch.allclose(before, after, atol=1e-5)

    def test_pipeline_checkpoint_round_trip(self, tmp_path):
        from pipeline.utils import load_flamingo_checkpoint, save_flamingo_checkpoint

        trained = build_model(open_gates=True)
        trained.configure_trainable_parameters()
        with torch.no_grad():
            for p in trained.xattn_blocks.parameters():
                p.add_(torch.randn_like(p) * 0.02)
            for p in trained.resampler.parameters():
                p.add_(torch.randn_like(p) * 0.02)
            trained.media_marker_delta.fill_(0.123)

        params = [p for p in trained.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        save_flamingo_checkpoint(tmp_path, 7, trained, optimizer, scheduler)

        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8]])
        pixel_values = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = trained(input_ids=ids, pixel_values=pixel_values).logits

        fresh = build_model(open_gates=True)
        with torch.no_grad():
            baseline = fresh(input_ids=ids, pixel_values=pixel_values).logits
        assert not torch.allclose(baseline, expected, atol=1e-4)

        ckpt = torch.load(tmp_path / "checkpoint_7.pt", weights_only=False)
        load_flamingo_checkpoint(fresh, ckpt)
        with torch.no_grad():
            restored = fresh(input_ids=ids, pixel_values=pixel_values).logits

        # The wrapped decoder layers hold unregistered references to the blocks,
        # so an in-place load must be visible through them too.
        assert torch.allclose(restored, expected, atol=1e-5)
        assert ckpt["step"] == 7


class TestMediaStateHygiene:
    def test_images_without_marker_raise(self):
        model = build_model()
        with pytest.raises(ValueError, match="no <image> marker"):
            model(input_ids=torch.tensor([[5, 6, 7]]), pixel_values=torch.randn(1, 3, 64, 64))

    def test_stale_media_is_not_reused_by_a_later_decode(self):
        model = build_model(open_gates=True)
        img_ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7]])
        with torch.no_grad():
            model(input_ids=img_ids, pixel_values=torch.randn(1, 3, 64, 64))

        # A new text-only prefill must clear the previous sequence's media...
        text_ids = torch.tensor([[5, 6, 7]])
        with torch.no_grad():
            out = model(input_ids=text_ids, use_cache=True)
        assert model._media_state is None

        # ...so the following cached decode step sees no media, and matches a
        # plain text-only forward of the same sequence.
        step = torch.tensor([[9]])
        with torch.no_grad():
            cached = model(
                input_ids=step,
                past_key_values=out.past_key_values,
                use_cache=True,
                cache_position=torch.tensor([3]),
            ).logits[:, -1]
            full = model(input_ids=torch.cat([text_ids, step], dim=1)).logits[:, -1]
        assert torch.allclose(cached, full, atol=1e-4)


class TestBatchInvariance:
    """A mixed batch must give the same answer as running each sample alone."""

    def _samples(self):
        # (input_ids, n_images): zero, one and two images, in one batch.
        return [
            ([5, 6, 7, 8, 9], 0),
            ([5, IMAGE_TOKEN_ID, 7, 8, 9], 1),
            ([IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID, 8, 9], 2),
        ]

    def test_mixed_image_counts_match_individual_forwards(self):
        torch.manual_seed(0)
        model = build_model(open_gates=True)
        samples = self._samples()
        images = torch.randn(sum(n for _, n in samples), 3, 64, 64)

        batch_ids = torch.tensor([ids for ids, _ in samples])
        with torch.no_grad():
            batched = model(input_ids=batch_ids, pixel_values=images).logits

        cursor = 0
        for row, (ids, n_img) in enumerate(samples):
            px = images[cursor : cursor + n_img] if n_img else None
            cursor += n_img
            with torch.no_grad():
                single = model(input_ids=torch.tensor([ids]), pixel_values=px).logits
            assert torch.allclose(batched[row : row + 1], single, atol=1e-5), (
                f"row {row} ({n_img} image(s)) differs when batched"
            )

    def test_second_image_does_not_leak_backwards(self):
        """Tokens before the 2nd marker must be unaffected by the 2nd image."""
        torch.manual_seed(0)
        model = build_model(open_gates=True)
        ids = torch.tensor([[IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID, 8, 9]])
        img_a = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            first = model(
                input_ids=ids, pixel_values=torch.cat([img_a, torch.randn(1, 3, 64, 64)])
            ).logits
            second = model(
                input_ids=ids, pixel_values=torch.cat([img_a, torch.randn(1, 3, 64, 64)])
            ).logits

        # Positions 0 and 1 precede the second marker -> identical.
        assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6)
        # Position 2 onwards sees the second image -> must differ.
        assert not torch.allclose(first[:, 2:], second[:, 2:], atol=1e-4)


class TestSelectorLearningSignal:
    """The headline claim: the LM loss must train *which patches* are kept."""

    def _selector_grad(self, model, ids, pixel_values, labels, disable_bias):
        model.zero_grad(set_to_none=True)
        if disable_bias:
            # Drop the straight-through keep-gate, leaving only the foveal path.
            original = model.resampler.forward

            def no_bias(*args, **kwargs):
                out = original(*args, **kwargs)
                return {**out, "media_bias": None}

            model.resampler.forward = no_bias
        try:
            model(input_ids=ids, pixel_values=pixel_values, labels=labels).loss.backward()
        finally:
            if disable_bias:
                del model.resampler.forward
        return sum(
            p.grad.abs().sum().item()
            for p in model.resampler.token_learner.parameters()
            if p.grad is not None
        )

    def test_lm_loss_reaches_the_patch_selector(self):
        torch.manual_seed(0)
        model = build_model(open_gates=True).double()
        ids = torch.tensor([[5, IMAGE_TOKEN_ID, 7, 8, 9]])
        labels = ids.clone()
        pixel_values = torch.randn(1, 3, 64, 64).double()

        with_gate = self._selector_grad(model, ids, pixel_values, labels, False)
        without_gate = self._selector_grad(model, ids, pixel_values, labels, True)

        assert with_gate > 0
        # The keep-gate must contribute a *material* share of the selector's
        # gradient. When it was a multiplicative scale in front of a LayerNorm
        # the two were identical to ~3e-5 relative, i.e. it did nothing.
        rel = abs(with_gate - without_gate) / max(without_gate, 1e-12)
        assert rel > 0.01, f"keep-gate changes selector grad by only {rel:.2e}"
