"""
Mode Connectivity Analysis for Decoder-Only LLMs
Analyzes the loss landscape between two trained models on reasoning tasks

Usage:
  python decoder_interpolation.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset gsm8k \
    --tune_method lora \
    --load_PET_path_1 ./model1/checkpoint.pt \
    --load_PET_path_2 ./model2/checkpoint.pt \
    --itpl_points 11 \
    --output_dir ./outputs/connectivity
"""

import os
import logging
import numpy as np
import pandas as pd
import torch

from DecoderLLM_model.modeling_decoder import load_decoder_llm_with_pet
from DecoderLLM_model.chat_template import ChatTemplateHandler
from DecoderLLM_model.decoder_evaluator import DecoderLLMEvaluator
from utils.options import option


def model_provider(args):
    """Load base model from HuggingFace"""
    model, config, tokenizer = load_decoder_llm_with_pet(args)
    return model, config, tokenizer


def load_checkpoint(path, tune_method=None):
    """
    Load checkpoint from path
    Handles different checkpoint formats:
    - Full model state dict (most common for full parameter training)
    - LoRA weights: {'lora': {...}}
    - Adapter weights: {'adapter': {...}}
    - Wrapped model: {'model': {...}} or {'state_dict': {...}}
    """
    logger = logging.getLogger(__name__)
    checkpoint = torch.load(path, map_location='cpu')

    # Priority order for full model checkpoints
    # 1. Direct state dict (most common)
    if all(not key.startswith(('lora', 'adapter', 'model', 'state_dict', 'optimizer', 'scheduler', 'epoch'))
           for key in checkpoint.keys()):
        logger.info(f"Loading direct state dict from {path}")
        return checkpoint

    # 2. Wrapped in 'model' or 'state_dict' key
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        logger.info(f"Loading from 'model' key in {path}")
        return checkpoint['model']

    if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
        logger.info(f"Loading from 'state_dict' key in {path}")
        return checkpoint['state_dict']

    # 3. Check for PET methods if tune_method is specified
    if tune_method:
        if tune_method in checkpoint:
            logger.info(f"Loading {tune_method} weights from {path}")
            return checkpoint[tune_method]

    # 4. Try to detect PET automatically
    if 'lora' in checkpoint:
        logger.info(f"Detected LoRA weights in {path}")
        return checkpoint['lora']

    if 'adapter' in checkpoint:
        logger.info(f"Detected adapter weights in {path}")
        return checkpoint['adapter']

    # 5. Fallback: assume the whole checkpoint is the state dict
    logger.warning(f"Could not determine checkpoint format, using entire checkpoint as state dict")
    return checkpoint


def main():
    args = option().parse()

    # Setup output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "interpolation_log.txt")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Set device
    args.n_gpu = torch.cuda.device_count()
    logger.info(f"Using {args.n_gpu} gpus")

    logger.info("="*60)
    logger.info("MODE CONNECTIVITY ANALYSIS FOR DECODER-ONLY LLMS")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint 1: {args.load_PET_path_1}")
    logger.info(f"Checkpoint 2: {args.load_PET_path_2}")
    logger.info(f"Interpolation points: {args.itpl_points}")
    logger.info("="*60)

    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = DecoderLLMEvaluator(args, logger, model_provider)

    # Get base model state dict
    model_dict = {k: v for k, v in evaluator.model.state_dict().items()}

    # Load checkpoints
    logger.info(f"\nLoading checkpoint 1 from: {args.load_PET_path_1}")
    PET_state_dict_1 = load_checkpoint(args.load_PET_path_1, args.tune_method)

    logger.info(f"Loading checkpoint 2 from: {args.load_PET_path_2}")
    PET_state_dict_2 = load_checkpoint(args.load_PET_path_2, args.tune_method)

    # Ensure keys match
    common_keys = set(PET_state_dict_1.keys()) & set(PET_state_dict_2.keys())
    if len(common_keys) == 0:
        logger.error("ERROR: No common keys found between checkpoints!")
        logger.error(f"Checkpoint 1 keys (first 5): {list(PET_state_dict_1.keys())[:5]}")
        logger.error(f"Checkpoint 2 keys (first 5): {list(PET_state_dict_2.keys())[:5]}")
        return

    logger.info(f"Found {len(common_keys)} common parameters between checkpoints")

    # Filter to common keys only
    PET_state_dict_1 = {k: v for k, v in PET_state_dict_1.items() if k in common_keys}
    PET_state_dict_2 = {k: v for k, v in PET_state_dict_2.items() if k in common_keys}

    # Evaluate endpoints first
    logger.info("\n" + "="*60)
    logger.info("Evaluating Endpoint 1 (x=0.0)...")
    logger.info("="*60)

    x = 0.0
    model_dict_to_update = {
        key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
        for key in PET_state_dict_1.keys()
    }
    model_dict.update(model_dict_to_update)
    evaluator.model.load_state_dict(model_dict, strict=False)

    metric, left_perf, _, _ = evaluator.itp_valid(x=x)
    logger.info(f"Endpoint 1: {metric}={left_perf:.4f}")

    logger.info("\n" + "="*60)
    logger.info("Evaluating Endpoint 2 (x=1.0)...")
    logger.info("="*60)

    x = 1.0
    model_dict_to_update = {
        key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
        for key in PET_state_dict_1.keys()
    }
    model_dict.update(model_dict_to_update)
    evaluator.model.load_state_dict(model_dict, strict=False)

    metric, right_perf, _, _ = evaluator.itp_valid(x=x)
    logger.info(f"Endpoint 2: {metric}={right_perf:.4f}")

    # Start interpolation
    logger.info("\n" + "="*60)
    logger.info(f"Starting Interpolation ({args.itpl_points} points)...")
    logger.info("="*60)

    results = []

    for x in np.linspace(0, 1, args.itpl_points):
        x = round(x, 3)
        logger.info(f"\nInterpolation point: x={x:.3f}")

        # Interpolate parameters: (1-x)*checkpoint1 + x*checkpoint2
        model_dict_to_update = {
            key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
            for key in PET_state_dict_1.keys()
        }
        model_dict.update(model_dict_to_update)
        evaluator.model.load_state_dict(model_dict, strict=False)

        # Evaluate
        metric, performance, loss, raw_scores = evaluator.itp_valid(x=x)

        logger.info(f"  {metric}={performance:.4f}")

        results.append({
            'x': x,
            'metric': metric,
            'performance': performance,
            'loss': loss
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"interpolation_results_{args.dataset}.csv")
    df.to_csv(csv_path, index=False)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("INTERPOLATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Endpoint 1 (x=0.0): {metric}={left_perf:.4f}")
    logger.info(f"Endpoint 2 (x=1.0): {metric}={right_perf:.4f}")
    logger.info(f"Best performance: {metric}={max(df['performance']):.4f}")
    logger.info(f"Min performance: {metric}={min(df['performance']):.4f}")
    logger.info(f"Mean performance: {metric}={df['performance'].mean():.4f}")
    logger.info(f"\nResults saved to: {csv_path}")
    logger.info("="*60)

    # Optional: Create visualization
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(df['x'], df['performance'], 'o-', linewidth=2, markersize=8)
        plt.scatter([0, 1], [left_perf, right_perf], s=150, c='red', marker='*', zorder=5)
        plt.xlabel('Interpolation coefficient (x)', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Mode Connectivity: {args.model} on {args.dataset}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(args.output_dir, f"interpolation_plot_{args.dataset}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")


if __name__ == "__main__":
    main()
