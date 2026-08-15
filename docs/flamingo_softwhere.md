# Flamingo-style Tiny Aya Vision with a SoftWhere resampler

This branch adds a second VLM architecture next to the existing LLaVA-style
model. Same vision tower (`google/siglip2-so400m-patch14-384`), same LLM
backbones (`CohereLabs/tiny-aya-base` / `tiny-aya-global`) — different way of
getting vision into the language model.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              TinyAyaFlamingoForConditionalGeneration                          │
│                                                                               │
│  ┌────────────┐   ┌───────────────────────────┐   ┌────────────────────────┐ │
│  │ SigLIP2    │   │  SoftWhere resampler      │   │ Tiny Aya (Cohere2)     │ │
│  │ so400m     │──>│  TokenLearner foveae      │──>│ frozen decoder layers  │ │
│  │ (frozen)   │   │  + per-map NMS patches    │   │ + gated XATTN-Dense    │ │
│  └────────────┘   └───────────────────────────┘   └────────────────────────┘ │
│    729 tokens          40 media tokens              cross-attention, not      │
│                                                     sequence splicing         │
└──────────────────────────────────────────────────────────────────────────────┘
```

## What changed vs. the LLaVA-style model

| | LLaVA-style (`tiny_aya_vision`) | Flamingo-style (`tiny_aya_flamingo`) |
|---|---|---|
| Vision → LLM | pixel-shuffle MLP, features **spliced into the text sequence** | resampler + **gated cross-attention** inside the LLM |
| `<image>` tokens in text | 196 placeholders per image | **1 marker** per image |
| Text sequence cost | +196 tokens/image | +1 token/image |
| Trainable | connector (~11.5M) | resampler + xattn blocks + `<image>` marker offset |
| Init behaviour | random projector → garbage image tokens | `tanh` gates at 0 → **exactly the frozen LLM** |
| Multi-image | one image per sequence in practice | native, with per-image attention masking |

## Components

### 1. SoftWhere resampler — `src/softwhere.py`

Flamingo compresses the frozen vision grid with a Perceiver Resampler (learned
latent queries). SoftWhere replaces those latents with **learned foveae**:

1. **TokenLearner** emits `S` soft spatial attention maps over the 27×27 SigLIP
   grid (`v10` = 4 convs + sigmoid, spatially blobby; `v11` = MLP + softmax,
   grid-free).
2. Each map pools one **soft foveal token** by attention-weighted averaging.
3. The maps are aggregated (`max` / `mean` / `logsumexp`) into one importance
   map, and **`K` raw patch tokens** are picked with SoftWhere's best-performing
   selection policy — per-map top-k with Chebyshev **non-maximum suppression**,
   so the budget is spread across foveae instead of piling onto one blob.
4. Kept patches carry a **straight-through keep-gate** built from the importance
   map (SoftWhere's `keep_gate` trick): exactly identity in the forward pass, but
   gradient flows from the language-modelling loss back into the selector, so the
   LM learns *which patches to keep* and not just how to read them.

   The gate is applied as an additive bias `log(softplus(importance))` on the
   **cross-attention logits**, not as a scale on the token. This matters: a
   multiplicative gate would sit directly in front of `norm_media`, and
   LayerNorm is scale-invariant (`LN(a·x) == LN(x)`), so it would be absorbed
   exactly — leaving the selector with a gradient ~5 orders of magnitude too
   small, decaying as `1/var(features)`. Biasing the logits is equivalent to
   scaling a patch's share of the attention mass, which no normalisation can
   undo. Subtracting the detached copy keeps the forward bias at exactly `0.0`,
   so selection stays hard.
5. A **diversity penalty** (`pairwise_overlap`, mean off-diagonal histogram
   intersection of the maps) stops the foveae from collapsing onto each other.

Defaults: `S = 8`, `K = 32` → **40 media tokens per image** instead of 196.

`resampler_type: perceiver` swaps in a faithful Perceiver Resampler, so the
SoftWhere contribution can be ablated with one config flag.

### 2. Gated cross-attention — `src/flamingo.py`

`GatedCrossAttentionBlock` = masked cross-attention + FFN, each scaled by
`tanh(gate)` with `gate` initialised to **0**. Blocks are inserted before
decoder layers `0, n, 2n, …` (`cross_attn_every_n_layers`, default 4) by
wrapping them in `FlamingoLayer`, which forwards unknown attribute lookups
(e.g. Cohere2's `layer.attention_type`) to the wrapped layer.

Masking follows Flamingo: `text_time[b, t]` counts the `<image>` markers at or
before position `t`, and with `only_attend_immediate_media=True` a text token
attends to exactly the most recent image. Text before any image gets a
guaranteed-zero cross-attention contribution (verified by a test), and
fully-masked rows are handled without producing NaNs.

Because the gates start at zero, an untrained model reproduces the frozen LLM
logit-for-logit — training starts from the LLM's caption perplexity, not from
noise.

### 3. Processor — `src/processing.py`

`TinyAyaFlamingoProcessor` is `TinyAyaVisionProcessor` with one change: an
image contributes a **single** `<image>` marker token. The marker's embedding
row is initialised to the mean of the pretrained embedding matrix and is the
only vocabulary row that trains (enforced by a gradient mask).

## Training

```bash
# SoftWhere resampler (default)
python pipeline/train_flamingo.py

# Flamingo Perceiver-Resampler ablation
python pipeline/train_flamingo.py flamingo=perceiver

# Multi-GPU
torchrun --nproc_per_node=8 pipeline/train_flamingo.py training.batch_size=64

# Knobs
python pipeline/train_flamingo.py \
  flamingo.num_foveal_tokens=16 \
  flamingo.softwhere_topk_patches=0 \
  flamingo.cross_attn_every_n_layers=2 \
  training.diversity_loss_weight=0.2
```

Loss: `CE + diversity_loss_weight * pairwise_overlap`. Frozen: vision tower and
LLM. Trainable: resampler, xattn blocks, and `media_marker_delta` — a single
`(hidden,)` offset added to the frozen `<image>` embedding. (Unfreezing the
embedding matrix itself to train one row would put 537M parameters into the
optimizer for tiny-aya-global: a 1 GiB gradient all-reduced every micro-step,
2 GiB of AdamW state, and weight decay silently pulling on the frozen
vocabulary and the tied `lm_head`.) Checkpoints
store only that stack (`pipeline/utils.py::save_flamingo_checkpoint`).

W&B logs `train/diversity_loss` plus a `gates/…` entry per block — the gates
leaving zero is the signal that the LLM has started using vision.

Config groups: `config/config_flamingo.yaml` (root),
`config/flamingo/{softwhere,perceiver}.yaml` (architecture),
`config/training/flamingo_alignment.yaml` (schedule).

> `torch.compile` is intentionally not applied: media conditioning is attached
> to the wrapped layers as Python state between forward passes.

## Cost

Per xattn block: `4 · d · d_inner` (attention) + `2 · d · 4d` (FFN). With
`d = 2048` and 9 blocks that is ~340M trainable parameters — Flamingo's
XATTN-Dense stack is deliberately heavy. Reduce with
`cross_attn_every_n_layers` (fewer blocks) or `xattn_ff_mult` (thinner FFN).

Verified end-to-end on an A100 with the real SigLIP2 tower: 40 media tokens per
image, zero-gate output identical to the frozen LLM, and a 25-step overfit run
driving CE 7.22 → 0.38 while the foveal overlap fell 0.996 → 0.044.

## Tests

```bash
pytest tests/test_softwhere.py tests/test_flamingo.py -v
```

CPU-only, no downloads: resampler shapes, NMS spacing/uniqueness, diversity
loss bounds, straight-through identity, zero-gate equivalence to the frozen
LLM, immediate-vs-cumulative media masking, per-sample image routing,
embedding-gradient masking, and cached-vs-uncached generation agreement.
