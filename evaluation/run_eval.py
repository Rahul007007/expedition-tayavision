"""
uv run python evaluation/run_eval.py

Args:
    [--task]: cvqa, cvqa_blind, xmmmu, ...
    [--backend]: vllm | hf | hf-multimodal
    [--batch-size]: auto | 1 (vllm | hf)
    [--limit]: int  (num samples for quick tests)
    [--output-dir]: str
    [--model-name]: str (HF repo id OR local path to save_pretrained output)

"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="lm-eval runner for Tiny Aya Vision benchmarks.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="hf-multimodal", choices=["vllm", "hf", "hf-multimodal"])
    parser.add_argument("--batch-size", type=str, default="1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    parser.add_argument("--log-samples", action="store_true", help="Log per-question results")
    parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template")
    args = parser.parse_args()

    logger.info(f"Starting evaluation for task: {args.task}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"=========================================")

    # Register TinyAyaVision with HuggingFace Auto classes so lm-eval can
    # load it via AutoModelForCausalLM.from_pretrained / AutoConfig.
    import models  # noqa: F401 — triggers Auto class registration

    import lm_eval

    model_args = f"pretrained={args.model_name},dtype=bfloat16,trust_remote_code=True"
    if args.backend == "vllm":
        model_args += ",tensor_parallel_size=1"

    eval_kwargs = dict(
        model=args.backend,
        model_args=model_args,
        tasks=[args.task],
        batch_size=args.batch_size,
        task_manager=lm_eval.tasks.TaskManager(include_path="evaluation/tasks"),
        log_samples=args.log_samples,
    )

    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    if args.apply_chat_template:
        eval_kwargs["apply_chat_template"] = True

    results = lm_eval.simple_evaluate(**eval_kwargs)

    from lm_eval.utils import make_table
    logger.info(make_table(results))

    if args.output_dir:
        import json
        from pathlib import Path
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"{args.task}_results.json", "w") as f:
            json.dump(results.get("results", {}), f, indent=2)
        logger.info(f"Results saved to {output_path / f'{args.task}_results.json'}")


if __name__ == "__main__":
    main()
