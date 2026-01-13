"""
Evaluation script for decoder-only LLMs on reasoning tasks.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer
from DecoderLLM_model import (
    DecoderLLMWithPET,
    DecoderLLMConfig,
    ChatTemplateHandler,
    create_reasoning_dataloader,
    DecoderLLMTrainer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate decoder-only LLM on reasoning tasks")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint to evaluate")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k", "math"],
                       help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test"],
                       help="Dataset split")
    parser.add_argument("--use_cot", action="store_true", default=True,
                       help="Use Chain-of-Thought prompting")

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum input length")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0 for greedy)")
    parser.add_argument("--do_sample", action="store_true", default=False,
                       help="Use sampling instead of greedy")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Number of sequences to generate (for voting)")

    # PET settings (should match training)
    parser.add_argument("--tune_method", type=str, default="lora",
                       choices=["adapter", "lora", "model"],
                       help="Tuning method used in training")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--adapter_size", type=int, default=64,
                       help="Adapter size")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use BF16")

    # Output
    parser.add_argument("--output_file", type=str, default=None,
                       help="Optional output file for results")

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("DECODER LLM EVALUATION ON REASONING TASKS")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"Tuning method: {args.tune_method}")
    print("="*60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create chat template handler
    print("Creating chat template handler...")
    chat_handler = ChatTemplateHandler(args.model_name, tokenizer)

    # Create model config
    print("\nCreating model configuration...")
    model_config = DecoderLLMConfig(
        model_name_or_path=args.model_name,
        tune_method=args.tune_method,
        apply_lora=(args.tune_method == "lora"),
        apply_adapter=(args.tune_method == "adapter"),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_size=args.adapter_size,
        use_bf16=args.bf16,
        device_map_auto=False,
    )

    # Create model
    print("Loading model...")
    model = DecoderLLMWithPET(model_config)

    # Create dataloader
    print(f"\nLoading {args.dataset} dataset ({args.split} split)...")
    dataloader = create_reasoning_dataloader(
        dataset_name=args.dataset,
        split=args.split,
        tokenizer=tokenizer,
        chat_template_handler=chat_handler,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_cot=args.use_cot,
        shuffle=False,
        cache_dir=args.cache_dir,
    )

    print(f"Evaluation batches: {len(dataloader)}")

    # Create trainer (for evaluation)
    print("\nInitializing trainer...")
    trainer = DecoderLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        chat_template_handler=chat_handler,
        device=args.device,
        output_dir="./temp_eval",
        bf16=args.bf16,
    )

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_path}...")
    trainer.load_checkpoint(args.checkpoint_path)

    # Evaluate
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60 + "\n")

    metrics = trainer.evaluate(
        dataloader,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        num_return_sequences=args.num_return_sequences,
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("="*60)

    # Save results if requested
    if args.output_file:
        import json
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "checkpoint": args.checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "metrics": metrics,
            "config": {
                "model_name": args.model_name,
                "tune_method": args.tune_method,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
            }
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
