"""
Mode Connectivity Analysis for Decoder-Only LLMs
Analyzes the loss landscape between two trained models on reasoning tasks

Supports two types of model sources:
1. Local checkpoint files: ./checkpoint.pt, /path/to/model.pth
2. HuggingFace model IDs: username/model-name, Qwen/Qwen2.5-0.5B-Instruct

Usage examples:
  # Two local checkpoints
  python decoder_interpolation.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --model_path_1 ./checkpoint1.pt \
    --model_path_2 ./checkpoint2.pt \
    --dataset gsm8k \
    --itpl_points 11

  # Two HuggingFace models
  python decoder_interpolation.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --model_path_1 username/model1 \
    --model_path_2 username/model2 \
    --dataset gsm8k \
    --itpl_points 11

  # Mixed: local and HuggingFace
  python decoder_interpolation.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --model_path_1 ./local_checkpoint.pt \
    --model_path_2 username/hf-model \
    --dataset math \
    --itpl_points 21
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from DecoderLLM_model.modeling_decoder import load_decoder_llm_with_pet
from DecoderLLM_model.chat_template import ChatTemplateHandler
from DecoderLLM_model.decoder_evaluator import DecoderLLMEvaluator
from utils.options import option


def model_provider(args):
    """Load base model from HuggingFace"""
    model, config, tokenizer = load_decoder_llm_with_pet(args)
    return model, config, tokenizer


def load_model_state_dict(path_or_id, logger, base_model_name=None, cache_dir=None):
    """
    Load full parameter model state dict from either:
    1. Local checkpoint file (.pt, .pth, .bin)
    2. HuggingFace model ID (username/model-name)

    Args:
        path_or_id: Local file path or HuggingFace model ID
        logger: Logger instance
        base_model_name: Base model name for HuggingFace loading
        cache_dir: Cache directory for HuggingFace models

    Returns:
        state_dict: Model state dictionary
    """
    # Check if it's a local file
    if os.path.exists(path_or_id):
        logger.info(f"Loading from local checkpoint: {path_or_id}")
        checkpoint = torch.load(path_or_id, map_location='cpu')

        # Handle different checkpoint formats
        # 1. Direct state dict (most common for full parameter models)
        if all(not key.startswith(('lora', 'adapter', 'model', 'state_dict', 'optimizer', 'scheduler', 'epoch'))
               for key in checkpoint.keys()):
            logger.info("  Format: Direct state dict")
            return checkpoint

        # 2. Wrapped in 'model' key
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            logger.info("  Format: Wrapped in 'model' key")
            return checkpoint['model']

        # 3. Wrapped in 'state_dict' key
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            logger.info("  Format: Wrapped in 'state_dict' key")
            return checkpoint['state_dict']

        # 4. Fallback: use entire checkpoint
        logger.warning("  Format: Unknown, using entire checkpoint")
        return checkpoint

    else:
        # Assume it's a HuggingFace model ID
        logger.info(f"Loading from HuggingFace: {path_or_id}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path_or_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            state_dict = model.state_dict()
            # Move to CPU to save GPU memory
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
            del model
            torch.cuda.empty_cache()
            logger.info(f"  Loaded {len(state_dict)} parameters from HuggingFace")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load from HuggingFace: {e}")
            raise


def main():
    args = option().parse()

    # Backward compatibility: support old argument names
    if hasattr(args, 'load_PET_path_1') and args.load_PET_path_1:
        args.model_path_1 = args.load_PET_path_1
    if hasattr(args, 'load_PET_path_2') and args.load_PET_path_2:
        args.model_path_2 = args.load_PET_path_2

    # Check required arguments
    if not hasattr(args, 'model_path_1') or not args.model_path_1:
        raise ValueError("--model_path_1 is required")
    if not hasattr(args, 'model_path_2') or not args.model_path_2:
        raise ValueError("--model_path_2 is required")

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
    logger.info(f"Using {args.n_gpu} GPUs")

    logger.info("="*60)
    logger.info("MODE CONNECTIVITY ANALYSIS FOR DECODER-ONLY LLMS")
    logger.info("="*60)
    logger.info(f"Base model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model 1: {args.model_path_1}")
    logger.info(f"Model 2: {args.model_path_2}")
    logger.info(f"Interpolation points: {args.itpl_points}")
    logger.info("="*60)

    # Create evaluator with base model
    logger.info("\nCreating evaluator with base model...")
    evaluator = DecoderLLMEvaluator(args, logger, model_provider)

    # Get base model state dict structure
    model_dict = {k: v for k, v in evaluator.model.state_dict().items()}
    logger.info(f"Base model has {len(model_dict)} parameters")

    # Load model state dicts
    logger.info("\n" + "="*60)
    logger.info("Loading Model 1...")
    logger.info("="*60)
    state_dict_1 = load_model_state_dict(
        args.model_path_1,
        logger,
        base_model_name=args.model,
        cache_dir=getattr(args, 'cache_dir', None)
    )

    logger.info("\n" + "="*60)
    logger.info("Loading Model 2...")
    logger.info("="*60)
    state_dict_2 = load_model_state_dict(
        args.model_path_2,
        logger,
        base_model_name=args.model,
        cache_dir=getattr(args, 'cache_dir', None)
    )

    # Find common keys between the two models
    common_keys = set(state_dict_1.keys()) & set(state_dict_2.keys())
    if len(common_keys) == 0:
        logger.error("\nERROR: No common keys found between models!")
        logger.error(f"Model 1 keys (first 5): {list(state_dict_1.keys())[:5]}")
        logger.error(f"Model 2 keys (first 5): {list(state_dict_2.keys())[:5]}")
        raise ValueError("Models have incompatible state dicts")

    logger.info(f"\nFound {len(common_keys)} common parameters")

    # Filter to common keys only
    state_dict_1 = {k: v for k, v in state_dict_1.items() if k in common_keys}
    state_dict_2 = {k: v for k, v in state_dict_2.items() if k in common_keys}

    # Evaluate endpoint 1 (x=0.0)
    logger.info("\n" + "="*60)
    logger.info("Evaluating Endpoint 1 (x=0.0) - Model 1")
    logger.info("="*60)

    x = 0.0
    model_dict_to_update = {
        key: ((1-x)*state_dict_1[key].cuda() + x*state_dict_2[key].cuda())
        for key in state_dict_1.keys()
    }
    model_dict.update(model_dict_to_update)
    evaluator.model.load_state_dict(model_dict, strict=False)

    metric, left_perf, left_loss, _ = evaluator.itp_valid(x=x)
    logger.info(f"Model 1 performance: {metric}={left_perf:.4f}, loss={left_loss:.4f}")

    # Evaluate endpoint 2 (x=1.0)
    logger.info("\n" + "="*60)
    logger.info("Evaluating Endpoint 2 (x=1.0) - Model 2")
    logger.info("="*60)

    x = 1.0
    model_dict_to_update = {
        key: ((1-x)*state_dict_1[key].cuda() + x*state_dict_2[key].cuda())
        for key in state_dict_1.keys()
    }
    model_dict.update(model_dict_to_update)
    evaluator.model.load_state_dict(model_dict, strict=False)

    metric, right_perf, right_loss, _ = evaluator.itp_valid(x=x)
    logger.info(f"Model 2 performance: {metric}={right_perf:.4f}, loss={right_loss:.4f}")

    # Start interpolation
    logger.info("\n" + "="*60)
    logger.info(f"Starting Interpolation ({args.itpl_points} points)")
    logger.info("="*60)

    results = []

    for x in np.linspace(0, 1, args.itpl_points):
        x = round(x, 3)
        logger.info(f"\nInterpolation point x={x:.3f}...")

        # Interpolate: (1-x)*model1 + x*model2
        model_dict_to_update = {
            key: ((1-x)*state_dict_1[key].cuda() + x*state_dict_2[key].cuda())
            for key in state_dict_1.keys()
        }
        model_dict.update(model_dict_to_update)
        evaluator.model.load_state_dict(model_dict, strict=False)

        # Evaluate at this interpolation point
        metric, performance, loss, raw_scores = evaluator.itp_valid(x=x)

        logger.info(f"  {metric}={performance:.4f}, loss={loss:.4f}")

        results.append({
            'x': x,
            'metric': metric,
            'performance': performance,
            'loss': loss
        })

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"interpolation_results_{args.dataset}.csv")
    df.to_csv(csv_path, index=False)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("INTERPOLATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Model 1 (x=0.0): {metric}={left_perf:.4f}")
    logger.info(f"Model 2 (x=1.0): {metric}={right_perf:.4f}")
    logger.info(f"Best performance: {metric}={max(df['performance']):.4f} at x={df.loc[df['performance'].idxmax(), 'x']:.3f}")
    logger.info(f"Worst performance: {metric}={min(df['performance']):.4f} at x={df.loc[df['performance'].idxmin(), 'x']:.3f}")
    logger.info(f"Mean performance: {metric}={df['performance'].mean():.4f}")
    logger.info(f"\nResults saved to: {csv_path}")

    # Create visualization
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(df['x'], df['performance'], 'o-', linewidth=2, markersize=8, label='Interpolation')
        plt.scatter([0, 1], [left_perf, right_perf], s=200, c='red', marker='*', zorder=5, label='Endpoints')
        plt.xlabel('Interpolation coefficient (x)', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Mode Connectivity: {args.model} on {args.dataset}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(args.output_dir, f"interpolation_plot_{args.dataset}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")

    logger.info("="*60)


if __name__ == "__main__":
    main()
