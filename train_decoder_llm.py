"""
Training script for decoder-only LLMs on reasoning tasks
Usage:
  python train_decoder_llm.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct \\
    --dataset gsm8k \\
    --tune_method lora \\
    --output_dir ./outputs/qwen_gsm8k \\
    --seed 42
"""

import os
import logging
import random
import numpy as np
import torch

from DecoderLLM_model.modeling_decoder import load_decoder_llm_with_pet
from DecoderLLM_model.chat_template import ChatTemplateHandler
from DecoderLLM_model.decoder_trainer import DecoderLLMTrainer, set_seed
from utils.options import option


def model_provider(args):
    """Load model, config, tokenizer"""
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
            logging.FileHandler(os.path.join(args.output_dir, "train_log.txt")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Set seed
    set_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Using {args.n_gpu} gpus")

    # Create chat template handler (will be initialized in trainer with tokenizer)
    chat_handler = None

    # Create trainer
    trainer = DecoderLLMTrainer(args, logger, model_provider, chat_handler)

    # Initialize chat handler with tokenizer
    trainer.chat_template_handler = ChatTemplateHandler(args.model, trainer.tokenizer)

    # Train
    logger.info("Starting training...")
    best_metric = trainer.train()

    logger.info(f"Training completed! Best metric: {best_metric:.4f}")

    # Test on best checkpoint
    logger.info("Evaluating best checkpoint...")
    trainer.load_checkpoint(f"{args.output_dir}/checkpoint-best.pt")
    test_metrics = trainer.valid()

    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
