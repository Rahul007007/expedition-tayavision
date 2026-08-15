"""Tiny Aya Vision — Flamingo-style variant with a SoftWhere resampler.

Architecture::

    VisionEncoder (frozen)
        -> SoftWhere multi-foveal resampler  (or Perceiver Resampler)
        -> gated cross-attention layers interleaved in a frozen Cohere2 LLM

Unlike :class:`~models.tiny_aya_vision.TinyAyaVisionForConditionalGeneration`,
image features are **not** spliced into the text sequence.  The text keeps a
single ``<image>`` marker per image which records *where* the image occurs;
media tokens are injected through cross-attention whose ``tanh`` gates start at
zero, so an untrained model reproduces the frozen LLM exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from config.model_config import TinyAyaFlamingoConfig
from src.flamingo import (
    FlamingoLayer,
    add_gated_cross_attention,
    clear_layer_conditioning,
    condition_layers,
    create_resampler,
)
from src.vision_encoders import create_vision_encoder


@dataclass
class TinyAyaFlamingoOutput(ModelOutput):
    """Output type for TinyAyaFlamingoForConditionalGeneration."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: tuple | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None
    diversity_loss: torch.FloatTensor | None = None
    foveal_maps: torch.FloatTensor | None = None


class TinyAyaFlamingoForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Flamingo-style multilingual VLM on top of the Tiny Aya backbones."""

    config_class = TinyAyaFlamingoConfig
    main_input_name = "input_ids"
    _supports_flash_attn_2 = False
    _no_split_modules = ["SigLIPVisionEncoder", "MoonViTVisionEncoder"]
    _tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}

    def __init__(self, config: TinyAyaFlamingoConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.vision_encoder = create_vision_encoder(config)
        if config.vision_tower_config is None:
            config.vision_tower_config = self.vision_encoder.vision_model.config.to_dict()

        self.resampler = create_resampler(config).to(config.torch_dtype)

        if config.text_config is not None:
            from transformers import CONFIG_MAPPING

            text_config_cls = CONFIG_MAPPING[config.text_config["model_type"]]
            text_cfg = text_config_cls.from_dict(config.text_config)
            self.language_model = AutoModelForCausalLM.from_config(text_cfg)
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                torch_dtype=config.torch_dtype,
                cache_dir=config.cache_dir,
            )
            config.text_config = self.language_model.config.to_dict()
            config._text_config_obj = None  # invalidate cached config

        self.xattn_blocks, self.xattn_layer_indices = add_gated_cross_attention(
            self.language_model,
            lm_hidden_size=config.llm_hidden_size,
            media_hidden_size=config.llm_hidden_size,
            every_n_layers=config.cross_attn_every_n_layers,
            dim_head=config.xattn_head_dim,
            heads=config.xattn_num_heads,
            ff_mult=config.xattn_ff_mult,
            only_attend_immediate_media=config.only_attend_immediate_media,
        )
        self.xattn_blocks.to(config.torch_dtype)

        # Learned offset added to the frozen `<image>` marker embedding.  A
        # dedicated (hidden,) parameter rather than the full embedding matrix:
        # for tiny-aya-global that tensor is 262144 x 2048 = 537M values, so
        # unfreezing it to train one row would put 1 GiB of gradient into every
        # DDP all-reduce (and 2 GiB of AdamW state on the GPU) to update 2048
        # numbers -- and decoupled weight decay would quietly pull on all the
        # frozen rows, and on the tied lm_head with them.
        self.media_marker_delta = (
            nn.Parameter(torch.zeros(config.llm_hidden_size, dtype=config.torch_dtype))
            if config.train_media_token_embedding
            else None
        )

        self.generation_config = self.language_model.generation_config
        self._image_token_id: int | None = config.image_token_id
        self._media_state: dict | None = None

        self.post_init()

    def _init_weights(self, module):
        pass

    # ------------------------------------------------------------------
    # Tokenizer / embeddings
    # ------------------------------------------------------------------

    @property
    def image_token_id(self) -> int:
        if self._image_token_id is None:
            raise ValueError("image_token_id not set. Call setup_tokenizer() first.")
        return self._image_token_id

    @property
    def num_media_tokens(self) -> int:
        return self.resampler.num_media_tokens

    def setup_tokenizer(self, tokenizer) -> None:
        """Register the ``<image>`` marker token and resize the embeddings.

        The freshly added row is initialised to the mean of the pretrained
        embedding matrix so the frozen LLM sees an in-distribution vector even
        before the marker embedding is trained.
        """
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.config.image_token]}
        )
        self._image_token_id = tokenizer.convert_tokens_to_ids(self.config.image_token)
        self.config.image_token_id = self._image_token_id

        if num_added > 0:
            old_embeddings = self.language_model.get_input_embeddings().weight.data
            mean_embedding = old_embeddings.mean(dim=0, keepdim=True).clone()
            self.language_model.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                self.language_model.get_input_embeddings().weight.data[
                    self._image_token_id
                ] = mean_embedding
            self.config.text_config = self.language_model.config.to_dict()
            self.config._text_config_obj = None  # invalidate cached config

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # ------------------------------------------------------------------
    # Trainable-parameter policy
    # ------------------------------------------------------------------

    def configure_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Freeze everything except the Flamingo stack, Flamingo-style.

        Trainable: the resampler and the gated cross-attention blocks.  When
        ``config.train_media_token_embedding`` is set, the ``<image>`` marker
        also gets a learned embedding offset (``media_marker_delta``).  The
        LLM's own embedding matrix stays frozen, so the rest of the vocabulary
        -- and the tied ``lm_head`` -- cannot drift.
        """
        self.requires_grad_(False)
        self.resampler.requires_grad_(True)
        self.xattn_blocks.requires_grad_(True)

        params = list(self.resampler.parameters()) + list(self.xattn_blocks.parameters())

        if self.media_marker_delta is not None:
            self.media_marker_delta.requires_grad_(True)
            params.append(self.media_marker_delta)

        return params

    # ------------------------------------------------------------------
    # Vision -> media tokens
    # ------------------------------------------------------------------

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode and resample images into media tokens.

        Returns the resampler output dict; ``media_tokens`` has shape
        ``(num_images, num_media_tokens, llm_hidden_size)``.
        """
        if self.config.vision_encoder_type == "moonvit":
            raw_features = self.vision_encoder(pixel_values, image_grid_hws=image_grid_hws)
            outputs = [
                self.resampler(feat.view(1, -1, feat.shape[-1])) for feat in raw_features
            ]
            merged = {
                "media_tokens": torch.cat([o["media_tokens"] for o in outputs], dim=0),
                "attn": None,
                "diversity_loss": torch.stack([o["diversity_loss"] for o in outputs]).mean(),
                "patch_indices": None,
            }
            return merged

        vision_features = self.vision_encoder(pixel_values)
        return self.resampler(vision_features)

    def _build_media(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_hws: torch.Tensor | None,
    ) -> dict:
        """Group per-image media tokens into ``(B, M, n, D)`` batch slots.

        Images arrive flat (one row per image in the batch); each sample claims
        as many as it has ``<image>`` markers, in order.
        """
        resampled = self.get_image_features(pixel_values, image_grid_hws)
        media_tokens = resampled["media_tokens"]  # (num_images, n, D)
        num_images, n_tok, dim = media_tokens.shape

        counts = (input_ids == self.image_token_id).sum(dim=1)  # (B,)
        batch_size = input_ids.shape[0]
        total = int(counts.sum().item())
        if total != num_images:
            raise ValueError(
                f"Got {num_images} image(s) but {total} <image> marker(s) in the batch. "
                "Each image needs exactly one <image> marker."
            )

        max_images = max(1, int(counts.max().item()))
        media = media_tokens.new_zeros(batch_size, max_images, n_tok, dim)
        media_valid = torch.zeros(
            batch_size, max_images, dtype=torch.bool, device=media_tokens.device
        )
        token_bias = resampled.get("media_bias")
        media_bias = (
            None
            if token_bias is None
            else token_bias.new_zeros(batch_size, max_images, n_tok)
        )

        if num_images > 0:
            sample_idx = torch.repeat_interleave(
                torch.arange(batch_size, device=counts.device), counts
            )
            offsets = torch.cumsum(counts, dim=0) - counts
            slot_idx = torch.arange(num_images, device=counts.device) - offsets[sample_idx]
            media[sample_idx, slot_idx] = media_tokens
            media_valid[sample_idx, slot_idx] = True
            if media_bias is not None:
                media_bias[sample_idx, slot_idx] = token_bias

        return {
            "media": media,
            "media_bias": media_bias,
            "media_valid": media_valid,
            "counts": counts,
            "media_tokens": media_tokens,
            "attn": resampled["attn"],
            "diversity_loss": resampled["diversity_loss"],
        }

    def _empty_media(self, input_ids: torch.LongTensor) -> dict:
        """Zero media with an all-false validity mask, for text-only batches.

        A dummy image is still pushed through the vision encoder and resampler
        so that their parameters take part in the autograd graph (keeping DDP
        gradient reduction in sync across ranks).
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        dtype = self.get_input_embeddings().weight.dtype

        if self.training:
            img_size = self.config.image_size
            dummy_pixel = torch.zeros(1, 3, img_size, img_size, device=device, dtype=dtype)
            resampled = self.get_image_features(dummy_pixel)
            media_tokens = resampled["media_tokens"]
            media = media_tokens[:1].unsqueeze(0).expand(batch_size, -1, -1, -1) * 0.0
            diversity_loss = resampled["diversity_loss"] * 0.0
        else:
            media = torch.zeros(
                batch_size, 1, self.num_media_tokens, self.config.llm_hidden_size,
                device=device, dtype=dtype,
            )
            diversity_loss = torch.zeros((), device=device, dtype=dtype)

        return {
            "media": media,
            "media_bias": None,
            "media_valid": torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            "counts": torch.zeros(batch_size, dtype=torch.long, device=device),
            "media_tokens": None,
            "attn": None,
            "diversity_loss": diversity_loss,
        }

    # ------------------------------------------------------------------
    # Forward / generation
    # ------------------------------------------------------------------

    @staticmethod
    def _past_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return 0

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> TinyAyaFlamingoOutput:
        if input_ids is None:
            raise ValueError(
                "TinyAyaFlamingoForConditionalGeneration needs input_ids to locate "
                "<image> markers; pass input_ids rather than inputs_embeds."
            )

        has_markers = bool((input_ids == self.image_token_id).any())
        is_decode_step = self._past_length(past_key_values) > 0

        if pixel_values is not None and has_markers:
            state = self._build_media(input_ids, pixel_values, image_grid_hws)
            self._media_state = state
            offset = torch.zeros_like(state["counts"])
        elif is_decode_step and self._media_state is not None:
            state = self._media_state
            offset = state["counts"]
        else:
            if pixel_values is not None and pixel_values.shape[0] > 0 and not is_decode_step:
                raise ValueError(
                    f"Got {pixel_values.shape[0]} image(s) but no <image> marker in "
                    "input_ids, so they would be silently ignored. Each image needs "
                    "exactly one <image> marker."
                )
            # A fresh sequence with no images: drop any media left over from a
            # previous call so a later decode step cannot pick up stale state.
            if not is_decode_step:
                self._media_state = None
            state = self._empty_media(input_ids)
            offset = state["counts"]

        # text_time[b, t] = number of <image> markers at or before position t.
        text_time = offset.unsqueeze(1) + (input_ids == self.image_token_id).cumsum(dim=1)

        media = state["media"].to(dtype=self.get_input_embeddings().weight.dtype)
        condition_layers(
            self.language_model,
            media,
            text_time,
            state["media_valid"],
            state.get("media_bias"),
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if self.media_marker_delta is not None:
            # Multiply by a 0/1 mask rather than indexing, so the parameter
            # still takes part in the graph on text-only batches (a missing
            # gradient would desynchronise DDP's reducer).
            marker = (input_ids == self.image_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds + marker.to(
                inputs_embeds.dtype
            ) * self.media_marker_delta.to(inputs_embeds.dtype)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            cache_position=cache_position,
            **kwargs,
        )

        clear_layer_conditioning(self.language_model)

        return TinyAyaFlamingoOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            image_hidden_states=state["media_tokens"],
            diversity_loss=state["diversity_loss"],
            foveal_maps=state["attn"],
        )

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        # Force DynamicCache instead of the HybridCache that Cohere2 normally
        # uses: the static-cache compilation path inside generate() is
        # incompatible with per-step media conditioning.
        from transformers import DynamicCache

        if not generation_config.use_cache:
            # Leave past_key_values unset — a non-None cache object makes
            # generate() trim input_ids as if a cache were being filled.
            return model_kwargs

        model_kwargs["past_key_values"] = DynamicCache(
            config=self.config.get_text_config(decoder=True)
        )
        return model_kwargs

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Generate text, resetting any media cached by a previous call."""
        self._media_state = None
        clear_layer_conditioning(self.language_model)
        try:
            return super().generate(*args, **kwargs)
        finally:
            self._media_state = None
            clear_layer_conditioning(self.language_model)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_hws=None,
        attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        """Prepare model inputs for autoregressive generation.

        ``input_ids`` are always forwarded (the marker positions are needed to
        build ``text_time``); pixel values are only needed on the first step,
        after which the resampled media stay cached on the model.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
        model_inputs.pop("inputs_embeds", None)
        model_inputs["input_ids"] = input_ids if past_key_values is None else model_inputs.get(
            "input_ids", input_ids
        )

        if self._past_length(past_key_values) == 0:
            model_inputs["pixel_values"] = pixel_values
            if image_grid_hws is not None:
                model_inputs["image_grid_hws"] = image_grid_hws

        return model_inputs

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def gate_values(self) -> dict[str, float]:
        """Current ``tanh`` gate magnitudes, one entry per xattn block.

        Values start at exactly 0 (the model is then equivalent to the frozen
        text-only LLM) and grow as the model learns to use vision.
        """
        values = {}
        for layer_idx, block in zip(self.xattn_layer_indices, self.xattn_blocks):
            values[f"gates/attn_layer_{layer_idx}"] = block.attn_gate.tanh().abs().item()
            values[f"gates/ff_layer_{layer_idx}"] = block.ff_gate.tanh().abs().item()
        return values


__all__ = [
    "FlamingoLayer",
    "TinyAyaFlamingoForConditionalGeneration",
    "TinyAyaFlamingoOutput",
]
