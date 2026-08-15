"""Flamingo-style alignment pre-training for Tiny Aya Vision (DDP).

Phase 1 training for the cross-attention VLM.  Only the Flamingo stack is
trained; everything it is bolted onto stays frozen:

  - Vision encoder:                 frozen
  - Resampler (SoftWhere/Perceiver): trainable
  - Gated cross-attention blocks:    trainable
  - ``<image>`` marker offset:       trainable (one hidden-size vector)
  - LLM backbone:                    frozen

Because the ``tanh`` gates start at zero, step 0 is numerically identical to
the frozen text-only LLM, and the loss curve starts from the LLM's caption
perplexity rather than from random.

Launch:
  Single GPU:  python pipeline/train_flamingo.py
  Multi GPU:   torchrun --nproc_per_node=NUM_GPUS pipeline/train_flamingo.py
  Ablation:    python pipeline/train_flamingo.py flamingo=perceiver
"""

import json
import sys
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

from config.training_config import FlamingoAlignmentConfig
from config.model_config import TinyAyaFlamingoConfig
from models.tiny_aya_flamingo import TinyAyaFlamingoForConditionalGeneration
from pipeline.data import AlignmentDataset, collate_fn
from pipeline.utils import (
    is_torchrun,
    setup_ddp,
    cleanup_ddp,
    _unwrap_model,
    save_flamingo_checkpoint,
    load_flamingo_checkpoint,
    find_latest_checkpoint,
    build_lr_scheduler,
)
from src.processing import TinyAyaFlamingoProcessor


def build_model_config(cfg: DictConfig) -> TinyAyaFlamingoConfig:
    """Build the model config from the vision/flamingo Hydra groups."""
    model_config = TinyAyaFlamingoConfig.for_encoder(
        cfg.vision.vision_encoder_type, llm=cfg.llm
    )
    for key, value in OmegaConf.to_container(cfg.flamingo, resolve=True).items():
        setattr(model_config, key, value)
    return model_config


def train(
    model,
    dataloader: torch.utils.data.DataLoader,
    sampler: DistributedSampler | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    trainable_params: list[torch.nn.Parameter],
    training_config: FlamingoAlignmentConfig,
    checkpoint_dir: Path,
    compute_dtype: torch.dtype,
    device: torch.device,
    step_offset: int = 0,
):
    model.train()
    accumulated_loss = 0.0
    accumulated_ce_loss = 0.0
    accumulated_div_loss = 0.0
    use_ddp = dist.is_initialized()
    is_main = (not use_ddp) or dist.get_rank() == 0
    raw = _unwrap_model(model)

    for epoch in range(training_config.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{training_config.num_epochs}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for step, batch in enumerate(pbar, start=step_offset):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"]
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)
            image_grid_hws = batch.get("image_grid_hws")
            if image_grid_hws is not None:
                image_grid_hws = image_grid_hws.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=compute_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_hws=image_grid_hws,
                    labels=labels,
                    use_cache=False,
                )
                ce_loss = outputs.loss / training_config.grad_acc_steps
                div_loss = outputs.diversity_loss.float() / training_config.grad_acc_steps

            loss = ce_loss + training_config.diversity_loss_weight * div_loss
            loss.backward()

            accumulated_loss += loss.item()
            accumulated_ce_loss += ce_loss.item()
            accumulated_div_loss += div_loss.item()

            if (step + 1) % training_config.grad_acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, training_config.max_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step = (step + 1) // training_config.grad_acc_steps

                if is_main:
                    log_dict = {
                        "train/loss": accumulated_loss,
                        "train/ce_loss": accumulated_ce_loss,
                        "train/diversity_loss": accumulated_div_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        **raw.gate_values(),
                    }

                    pbar.set_postfix(
                        loss=f"{accumulated_loss:.4f}",
                        div=f"{accumulated_div_loss:.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )

                    if opt_step % training_config.logging_steps == 0:
                        tqdm.write(
                            f"Epoch {epoch}, Opt Step {opt_step}, "
                            f"Loss {accumulated_loss:.4f}, "
                            f"LR {lr_scheduler.get_last_lr()[0]}"
                        )

                    if opt_step % training_config.save_steps == 0:
                        save_flamingo_checkpoint(
                            checkpoint_dir, step + 1, model, optimizer, lr_scheduler
                        )

                    wandb.log(log_dict, step=opt_step)

                if use_ddp:
                    dist.barrier()

                accumulated_loss = 0.0
                accumulated_ce_loss = 0.0
                accumulated_div_loss = 0.0

    if is_main:
        save_flamingo_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)
    if use_ddp:
        dist.barrier()
    if is_main:
        print("Training complete")


def run(cfg: DictConfig):
    """Core training logic with DDP support."""
    use_ddp = is_torchrun()
    if use_ddp:
        local_rank = setup_ddp()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_config = FlamingoAlignmentConfig(**training_dict)

    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed_all(training_config.seed)

    model_config = build_model_config(cfg)

    assert training_config.batch_size % world_size == 0, (
        f"batch_size ({training_config.batch_size}) must be "
        f"divisible by world_size ({world_size})"
    )
    per_gpu_batch_size = training_config.batch_size // world_size

    if is_main:
        print(f"{'DDP' if use_ddp else 'Single-GPU'}: world_size={world_size}, "
              f"global_batch_size={training_config.batch_size}, "
              f"per_gpu_batch_size={per_gpu_batch_size}")

    resume_run_id = cfg.get("resume", None)
    run_id = resume_run_id if resume_run_id else str(uuid.uuid4())

    checkpoint_dir = Path(training_config.models_dir) / run_id
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Checkpoint dir: {checkpoint_dir}")
    if use_ddp:
        dist.barrier()

    config_path = checkpoint_dir / "config.json"
    if is_main and not config_path.exists():
        with open(config_path, "w") as f:
            json.dump({
                "training_config": asdict(training_config),
                "model_config": model_config.to_dict(),
            }, f, indent=2)

    if is_main:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=run_id,
            id=run_id.replace("-", ""),
            resume="allow",
            config={**asdict(training_config), **model_config.to_dict()},
        )

    model = TinyAyaFlamingoForConditionalGeneration(config=model_config)
    processor = TinyAyaFlamingoProcessor(config=model_config)
    model.setup_tokenizer(processor.tokenizer)

    trainable_params = model.configure_trainable_parameters()

    model.to(device, non_blocking=True)

    compute_dtype = getattr(torch, training_config.torch_dtype)
    model.vision_encoder.to(dtype=compute_dtype, non_blocking=True)
    model.language_model.to(dtype=compute_dtype, non_blocking=True)

    model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if is_main:
        n_resampler = sum(p.numel() for p in model.resampler.parameters())
        n_xattn = sum(p.numel() for p in model.xattn_blocks.parameters())
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"Resampler: {model_config.resampler_type} "
            f"({model.num_media_tokens} media tokens/image); "
            f"xattn at layers {model.xattn_layer_indices}"
        )
        print(
            f"Trainable: resampler {n_resampler / 1e6:.1f}M + "
            f"xattn {n_xattn / 1e6:.1f}M "
            f"(+ <image> marker offset) / {n_total / 1e6:.1f}M total"
        )

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    resume_step = 0
    ckpt = None
    if resume_run_id:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            if is_main:
                print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            load_flamingo_checkpoint(model, ckpt)
            resume_step = ckpt["step"]
        elif is_main:
            print(f"No checkpoints found in {checkpoint_dir}, starting from scratch")

    dataset = AlignmentDataset(
        config=model_config,
        dataset_name=training_config.dataset_name,
        data_dir=training_config.data_dir,
        processor_cls=TinyAyaFlamingoProcessor,
    )

    full_dataset_len = len(dataset)

    samples_to_skip = resume_step * per_gpu_batch_size
    if 0 < samples_to_skip < len(dataset):
        dataset = torch.utils.data.Subset(
            dataset, list(range(samples_to_skip, len(dataset)))
        )
        if is_main:
            print(f"Skipped {samples_to_skip} samples, {len(dataset)} remaining")

    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=training_config.seed,
        )
        if use_ddp
        else None
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=training_config.num_workers > 0,
        prefetch_factor=2 if training_config.num_workers > 0 else None,
        drop_last=False,
    )

    opt = torch.optim.AdamW(
        trainable_params,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    lr_scheduler = build_lr_scheduler(
        opt, training_config, full_dataset_len, per_gpu_batch_size, world_size
    )

    if resume_step > 0 and ckpt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    train(
        model=model,
        dataloader=loader,
        sampler=sampler,
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        trainable_params=trainable_params,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        compute_dtype=compute_dtype,
        device=device,
        step_offset=resume_step,
    )

    if is_main:
        wandb.finish()
    if use_ddp:
        cleanup_ddp()


@hydra.main(version_base="1.3", config_path="../config", config_name="config_flamingo")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
