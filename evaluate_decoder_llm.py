"""
Evaluate a decoder-only LLM on reasoning tasks
Usage:
  python evaluate_decoder_llm.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset gsm8k \
    --checkpoint_path ./path/to/checkpoint.pt \
    --output_dir ./outputs/eval
"""

import os
import logging
import torch

from DecoderLLM_model.modeling_decoder import load_decoder_llm_with_pet
from DecoderLLM_model.decoder_evaluator import DecoderLLMEvaluator
from utils.options import option


def model_provider(args):
    """Load model from HuggingFace or local path"""
    model, config, tokenizer = load_decoder_llm_with_pet(args)
    return model, config, tokenizer


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
            logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Set device
    args.n_gpu = torch.cuda.device_count()
    logger.info(f"Using {args.n_gpu} gpus")

    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = DecoderLLMEvaluator(args, logger, model_provider)

    # Load checkpoint if provided
    if hasattr(args, 'checkpoint') and args.checkpoint:
        evaluator.load_checkpoint(args.checkpoint)

    # Evaluate
    logger.info("="*60)
    logger.info("Starting evaluation...")
    logger.info("="*60)

    metrics = evaluator.evaluate()

    logger.info("="*60)
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("="*60)

    # Save results
    import json
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint if hasattr(args, 'checkpoint') else None,
        "metrics": metrics
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
