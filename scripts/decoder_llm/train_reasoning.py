"""
Training script for decoder-only LLMs on reasoning tasks.
Supports Qwen, Llama with Parameter-Efficient Tuning.
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
    parser = argparse.ArgumentParser(description="Train decoder-only LLM on reasoning tasks")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model name or path (Qwen, Llama, etc.)")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for model and data")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k", "math"],
                       help="Dataset to use")
    parser.add_argument("--use_cot", action="store_true", default=True,
                       help="Use Chain-of-Thought prompting")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/decoder_llm",
                       help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--max_steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")

    # Logging and evaluation
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")

    # Parameter-Efficient Tuning
    parser.add_argument("--tune_method", type=str, default="lora",
                       choices=["adapter", "lora", "model"],
                       help="Tuning method")
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
                       help="Use BF16 training")
    parser.add_argument("--fp16", action="store_true", default=False,
                       help="Use FP16 training")

    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    print("="*60)
    print("DECODER LLM TRAINING ON REASONING TASKS")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Tuning method: {args.tune_method}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("="*60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    # Ensure pad token is set
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

    # Create dataloaders
    print(f"\nLoading {args.dataset} dataset...")

    train_dataloader = create_reasoning_dataloader(
        dataset_name=args.dataset,
        split="train",
        tokenizer=tokenizer,
        chat_template_handler=chat_handler,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_cot=args.use_cot,
        shuffle=True,
        cache_dir=args.cache_dir,
    )

    eval_dataloader = create_reasoning_dataloader(
        dataset_name=args.dataset,
        split="test",
        tokenizer=tokenizer,
        chat_template_handler=chat_handler,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        use_cot=args.use_cot,
        shuffle=False,
        cache_dir=args.cache_dir,
    )

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Eval batches: {len(eval_dataloader)}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DecoderLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        chat_template_handler=chat_handler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    trainer.train()

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)

    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = trainer.evaluate(eval_dataloader)
    print(f"\nFinal Test Metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
