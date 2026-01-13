"""
Mode connectivity analysis script for decoder-only LLMs.
Analyzes the loss landscape between two trained models.
"""

import os
import sys
import torch
import argparse
import json
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Mode connectivity analysis for decoder LLMs")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--checkpoint_1", type=str, required=True,
                       help="Path to first checkpoint")
    parser.add_argument("--checkpoint_2", type=str, required=True,
                       help="Path to second checkpoint")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k", "math"],
                       help="Dataset for evaluation")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test"],
                       help="Dataset split")
    parser.add_argument("--use_cot", action="store_true", default=True,
                       help="Use Chain-of-Thought prompting")

    # Interpolation arguments
    parser.add_argument("--num_points", type=int, default=11,
                       help="Number of interpolation points (including endpoints)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum input length")

    # PET settings
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
                       help="Use BF16")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/mode_connectivity",
                       help="Output directory for results")
    parser.add_argument("--plot", action="store_true", default=True,
                       help="Generate plots")

    return parser.parse_args()


def plot_interpolation_results(results: dict, output_dir: Path):
    """Plot mode connectivity results"""
    import numpy as np

    alphas = np.array(results["alphas"])
    accuracies = np.array(results["accuracies"])

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot accuracy curve
    plt.plot(alphas, accuracies, 'o-', linewidth=2, markersize=8, label="Accuracy")

    # Mark endpoints
    plt.scatter([alphas[0], alphas[-1]], [accuracies[0], accuracies[-1]],
               s=150, c='red', marker='*', zorder=5, label="Endpoints")

    # Formatting
    plt.xlabel("Interpolation coefficient α", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Mode Connectivity: Linear Interpolation", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add statistics
    max_acc = max(accuracies)
    min_acc = min(accuracies)
    mean_acc = np.mean(accuracies)

    stats_text = f"Max: {max_acc:.4f}\nMin: {min_acc:.4f}\nMean: {mean_acc:.4f}"
    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    # Save figure
    plot_path = output_dir / "mode_connectivity_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # Also save as PDF
    plot_path_pdf = output_dir / "mode_connectivity_plot.pdf"
    plt.savefig(plot_path_pdf, bbox_inches='tight')

    plt.close()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MODE CONNECTIVITY ANALYSIS FOR DECODER-ONLY LLMS")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint 1: {args.checkpoint_1}")
    print(f"Checkpoint 2: {args.checkpoint_2}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"Interpolation points: {args.num_points}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

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

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DecoderLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        chat_template_handler=chat_handler,
        device=args.device,
        output_dir=args.output_dir,
        bf16=args.bf16,
    )

    # Perform mode connectivity analysis
    print("\n" + "="*80)
    print("PERFORMING MODE CONNECTIVITY ANALYSIS")
    print("="*80 + "\n")

    results = trainer.interpolate_evaluate(
        checkpoint_path_1=args.checkpoint_1,
        checkpoint_path_2=args.checkpoint_2,
        dataloader=dataloader,
        num_points=args.num_points,
        save_results=True,
    )

    # Save detailed results
    results_file = output_dir / "mode_connectivity_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "checkpoint_1": args.checkpoint_1,
            "checkpoint_2": args.checkpoint_2,
            "dataset": args.dataset,
            "split": args.split,
            "num_points": args.num_points,
            "results": results,
            "config": {
                "model_name": args.model_name,
                "tune_method": args.tune_method,
                "use_cot": args.use_cot,
            }
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        plot_interpolation_results(results, output_dir)

    print("\n" + "="*80)
    print("MODE CONNECTIVITY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
