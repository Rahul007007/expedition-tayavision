from __future__ import annotations

import inspect
from pathlib import Path

import yaml
from transformers import PretrainedConfig


class TinyAyaVisionConfig(PretrainedConfig):
    """Central configuration for the Tiny Aya Vision model."""

    model_type = "tiny_aya_vision"

    def __init__(
        self,
        vision_encoder_type: str = "siglip",
        vision_model_name: str = "google/siglip2-so400m-patch14-384",
        vision_hidden_size: int = 1152,
        image_size: int = 384,
        patch_size: int = 14,
        vision_grid_size: int = 27,
        num_vision_tokens: int = 729,
        trust_remote_code: bool = False,
        connector_type: str = "pixel_shuffle",
        downsample_factor: int = 2,
        padded_grid_size: int = 28,
        num_tokens_after_shuffle: int = 196,
        pixel_shuffle_embed_dim: int = 4608,
        tokens_per_tile: int = 4,
        in_token_limit: int = 1024,
        connector_intermediate_size: int = 2048,
        adapter_layer_norm_eps: float = 1e-6,
        post_projector_rms_norm: bool = False,
        llm_model_name: str = "CohereLabs/tiny-aya-base",
        llm_hidden_size: int = 2048,
        llm_vocab_size: int = 262144,
        num_llm_layers: int = 36,
        image_token: str = "<image>",
        image_token_id: int | None = None,
        torch_dtype: str = "bfloat16",
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "full",
        cache_dir: str | None = None,
        text_config: dict | None = None,
        vision_tower_config: dict | None = None,
        **kwargs,
    ):
        self.vision_encoder_type = vision_encoder_type
        self.vision_model_name = vision_model_name
        self.vision_hidden_size = vision_hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_grid_size = vision_grid_size
        self.num_vision_tokens = num_vision_tokens
        self.trust_remote_code = trust_remote_code
        self.connector_type = connector_type
        self.downsample_factor = downsample_factor
        self.padded_grid_size = padded_grid_size
        self.num_tokens_after_shuffle = num_tokens_after_shuffle
        self.pixel_shuffle_embed_dim = pixel_shuffle_embed_dim
        self.tokens_per_tile = tokens_per_tile
        self.in_token_limit = in_token_limit
        self.connector_intermediate_size = connector_intermediate_size
        self.adapter_layer_norm_eps = adapter_layer_norm_eps
        self.post_projector_rms_norm = post_projector_rms_norm
        self.llm_model_name = llm_model_name
        self.llm_hidden_size = llm_hidden_size
        self.llm_vocab_size = llm_vocab_size
        self.num_llm_layers = num_llm_layers
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        # Treat empty strings from older serialized configs as "use the
        # Hugging Face default cache" rather than "cache in cwd".
        self.cache_dir = cache_dir or None
        self.text_config = text_config
        self.vision_tower_config = vision_tower_config
        self._text_config_obj = None
        super().__init__(torch_dtype=torch_dtype, **kwargs)

    def get_text_config(self, decoder: bool = False, **kwargs) -> "PretrainedConfig":
        """Return a proper PretrainedConfig for the LLM sub-model.

        Required by transformers >=4.49 for GenerationConfig and DynamicCache
        initialization during generate(). Must never return a raw dict —
        newer transformers calls .to_dict() on the result.
        """
        if not isinstance(self._text_config_obj, PretrainedConfig):
            # _text_config_obj may be None or a stale dict deserialized from
            # a config.json that was saved before this guard was in place.
            if self.text_config is not None:
                from transformers import CONFIG_MAPPING
                text_cls = CONFIG_MAPPING[self.text_config["model_type"]]
                self._text_config_obj = text_cls.from_dict(self.text_config)
        if isinstance(self._text_config_obj, PretrainedConfig):
            return self._text_config_obj
        return self

    def to_dict(self):
        output = super().to_dict()
        # _text_config_obj is a derived/cached value; exclude it from the
        # serialized config so it is never loaded back as a stale raw dict.
        output.pop("_text_config_obj", None)
        return output

    @classmethod
    def for_base(cls) -> TinyAyaVisionConfig:
        """Config for CohereLabs/tiny-aya-base (pretrained base model)."""
        return cls(llm_model_name="CohereLabs/tiny-aya-base")

    @classmethod
    def for_global(cls) -> TinyAyaVisionConfig:
        """Config for CohereLabs/tiny-aya-global (instruction-tuned, best multilingual balance)."""
        return cls(llm_model_name="CohereLabs/tiny-aya-global")

    @classmethod
    def for_encoder(cls, encoder: str, llm: str = "base") -> TinyAyaVisionConfig:
        """Load config from config/vision/<encoder>.yaml and merge with defaults.

        Args:
            encoder: Vision encoder name — "siglip" or "moonvit".
            llm: LLM variant — "base" or "global".

        Example:
            config = TinyAyaVisionConfig.for_encoder("moonvit")
            config = TinyAyaVisionConfig.for_encoder("siglip", llm="global")
        """
        yaml_path = Path(__file__).parent / "vision" / f"{encoder}.yaml"
        if not yaml_path.exists():
            available = [p.stem for p in yaml_path.parent.glob("*.yaml")]
            raise FileNotFoundError(
                f"No vision config for '{encoder}' at {yaml_path}. "
                f"Available: {available}"
            )

        with open(yaml_path) as f:
            overrides = yaml.safe_load(f)

        # Collect accepted fields across the MRO so subclasses that only
        # declare their own extra fields still accept the base ones.
        valid_fields: set[str] = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            valid_fields |= set(inspect.signature(init).parameters) - {"self", "kwargs"}
        filtered = {k: v for k, v in overrides.items() if k in valid_fields}

        llm_names = {
            "base": "CohereLabs/tiny-aya-base",
            "global": "CohereLabs/tiny-aya-global",
        }
        if llm not in llm_names:
            raise ValueError(f"llm must be 'base' or 'global', got '{llm}'")

        return cls(**filtered, llm_model_name=llm_names[llm])


class TinyAyaFlamingoConfig(TinyAyaVisionConfig):
    """Config for the Flamingo-style Tiny Aya Vision variant.

    Keeps every vision-encoder and LLM setting of :class:`TinyAyaVisionConfig`
    (same SigLIP tower, same ``CohereLabs/tiny-aya-*`` backbone) and adds the
    Flamingo conditioning stack: a resampler that compresses the vision grid
    into a few media tokens, plus gated cross-attention layers interleaved into
    the frozen LLM.

    The LLaVA-style ``connector_*`` / pixel-shuffle fields are inherited but
    unused: media tokens are consumed by cross-attention, not spliced into the
    text sequence, so a single ``<image>`` marker token is emitted per image.
    """

    model_type = "tiny_aya_flamingo"

    def __init__(
        self,
        resampler_type: str = "softwhere",
        # --- SoftWhere multi-foveal resampler ---
        num_foveal_tokens: int = 8,
        softwhere_variant: str = "v10",
        softwhere_agg: str = "max",
        softwhere_topk_patches: int = 32,
        softwhere_nms_min_dist: int = 2,
        # --- Perceiver resampler (Flamingo baseline) ---
        num_latent_tokens: int = 64,
        perceiver_depth: int = 2,
        # --- Gated cross-attention ---
        cross_attn_every_n_layers: int = 4,
        xattn_num_heads: int = 8,
        xattn_head_dim: int = 64,
        xattn_ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
        train_media_token_embedding: bool = True,
        **kwargs,
    ):
        self.resampler_type = resampler_type
        self.num_foveal_tokens = num_foveal_tokens
        self.softwhere_variant = softwhere_variant
        self.softwhere_agg = softwhere_agg
        self.softwhere_topk_patches = softwhere_topk_patches
        self.softwhere_nms_min_dist = softwhere_nms_min_dist
        self.num_latent_tokens = num_latent_tokens
        self.perceiver_depth = perceiver_depth
        self.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.xattn_num_heads = xattn_num_heads
        self.xattn_head_dim = xattn_head_dim
        self.xattn_ff_mult = xattn_ff_mult
        self.only_attend_immediate_media = only_attend_immediate_media
        self.train_media_token_embedding = train_media_token_embedding
        super().__init__(**kwargs)

    @property
    def num_media_tokens(self) -> int:
        """Media tokens produced per image by the configured resampler."""
        if self.resampler_type == "softwhere":
            return self.num_foveal_tokens + self.softwhere_topk_patches
        return self.num_latent_tokens
