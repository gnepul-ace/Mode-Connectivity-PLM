"""
Mode Connectivity Analysis for Decoder-Only LLMs
Based on task_interpolation.py structure, adapted for Qwen/Llama on reasoning tasks
"""

import os
import logging
import random
import numpy as np
import json
import pandas as pd
import torch

from DecoderLLM_model.modeling_decoder import load_decoder_llm_with_pet
from DecoderLLM_model.chat_template import ChatTemplateHandler
from DecoderLLM_model.decoder_trainer import DecoderLLMTrainer, set_seed
from utils.options import option


def model_provider(args):
    """
    Model provider for decoder-only LLMs
    Similar to T5 model_provider but for causal models
    """
    model, config, tokenizer = load_decoder_llm_with_pet(args)
    return model, config, tokenizer


def write_result(output_dir, result_name, dev_performance, best_dev_performance, valid_loss,
                test_performance, test_loss, args, df, prefix, metric, x):
    """Write results to CSV (same as task_interpolation.py)"""
    best_config = None

    if os.path.exists(os.path.join(output_dir, result_name)):
        df_load = pd.read_csv(os.path.join(output_dir, result_name), sep=',')
        if 'best' in df_load.prefix[len(df_load)-1]:
            best_dev_performance = df_load.dev_performance.iloc[-1]
            best_config = df_load.tail(1).values.tolist()[0]
            df_load.drop(len(df_load)-1, inplace=True)
        else:
            max_iloc = df_load['dev_performance'].argmax()
            best_config = df_load.iloc[[max_iloc]].values.tolist()[0]
            best_dev_performance = max(df_load.dev_performance)
        df = df_load

    if dev_performance > best_dev_performance:
        best_dev_performance = dev_performance
        best_valid_loss = valid_loss
        best_test_performance = test_performance
        best_test_loss = test_loss
        best_config = [prefix, metric, x, best_dev_performance,
                      best_valid_loss, best_test_performance, best_test_loss]

    df.loc[len(df.index)] = [prefix, metric, x, dev_performance,
                             valid_loss, test_performance, test_loss]
    df.to_csv(os.path.join(output_dir, result_name), sep=',',
              index=False, header=True, float_format='%.4f')

    return best_config, best_dev_performance, df


def main():
    args = option().parse()

    # Setup output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir

    # Setup logging
    log_filename = "{}log.txt".format("" if args.do_train else "eval_")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_filename)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    # Set seed
    args.seed = int(args.seed)
    set_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Using {args.n_gpu} gpus")

    # For reasoning tasks, we use dataset name as prefix
    prefix = args.dataset

    logger.info(f"Analyzing mode connectivity for: {prefix}")

    # Create results DataFrame
    df = pd.DataFrame(columns=["prefix", "metric", "x",
                              "dev_performance", "dev_loss", "test_performance", "test_loss"])

    best_dev_performance = -1.0
    best_config = None

    # Create chat template handler
    # We need tokenizer first to create handler, so we'll create it during trainer init
    logger.info("Creating trainer and loading checkpoints...")

    # Create trainer
    chat_handler = None  # Will be created with tokenizer

    trainer = DecoderLLMTrainer(args, logger, model_provider, chat_handler)

    # Now create chat handler with tokenizer
    trainer.chat_template_handler = ChatTemplateHandler(args.model, trainer.tokenizer)
    trainer.chat_handler = trainer.chat_template_handler  # For compatibility

    # Load checkpoints for interpolation
    model_dict = {k: v for (k, v) in trainer.model.state_dict().items()}

    def load_PET_from_path(path):
        """Load checkpoint from path"""
        load_PET = torch.load(path)
        if args.tune_method in load_PET:
            PET_state_dict = load_PET[args.tune_method]
        else:
            PET_state_dict = load_PET
        return PET_state_dict

    logger.info(f"Loading checkpoint 1 from: {args.load_PET_path_1}")
    PET_state_dict_1 = load_PET_from_path(args.load_PET_path_1)

    logger.info(f"Loading checkpoint 2 from: {args.load_PET_path_2}")
    PET_state_dict_2 = load_PET_from_path(args.load_PET_path_2)

    assert PET_state_dict_1.keys() == PET_state_dict_2.keys(), "Checkpoint keys must match!"

    logger.info("Checkpoints loaded successfully. Starting mode connectivity analysis...")

    # Evaluate endpoints first (x=0 and x=1)
    logger.info("="*60)
    logger.info("Evaluating endpoint 1 (x=0)...")
    x = 0.0
    model_dict_to_update = {
        key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
        for key in PET_state_dict_1.keys()
    }
    model_dict.update(model_dict_to_update)
    trainer.model.load_state_dict(model_dict)

    metric, left_dev, left_dev_loss, _ = trainer.itp_valid(x=x)
    _, left_test, left_test_loss, _ = trainer.itp_test(args=args, model=trainer.model, x=x)

    logger.info(f"Endpoint 1 (x=0): {metric}={left_dev:.4f}")

    logger.info("="*60)
    logger.info("Evaluating endpoint 2 (x=1)...")
    x = 1.0
    model_dict_to_update = {
        key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
        for key in PET_state_dict_1.keys()
    }
    model_dict.update(model_dict_to_update)
    trainer.model.load_state_dict(model_dict)

    metric, right_dev, right_dev_loss, _ = trainer.itp_valid(x=x)
    _, right_test, right_test_loss, _ = trainer.itp_test(args=args, model=trainer.model, x=x)

    logger.info(f"Endpoint 2 (x=1): {metric}={right_dev:.4f}")

    # Start interpolation
    logger.info("="*60)
    logger.info(f"Starting interpolation with {args.itpl_points} points...")
    logger.info("="*60)

    result_name = f"interpolation_results_{prefix}.csv"

    for x in np.linspace(0, 1, args.itpl_points):
        x = round(x, 3)
        logger.info(f"\nInterpolation point: x={x:.3f}")

        # Interpolate parameters: (1-x)*checkpoint1 + x*checkpoint2
        model_dict_to_update = {
            key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
            for key in PET_state_dict_1.keys()
        }
        model_dict.update(model_dict_to_update)
        trainer.model.load_state_dict(model_dict)

        # Evaluate on validation/test
        metric, dev_performance, valid_loss, raw_score = trainer.itp_valid(x=x)
        _, test_performance, test_loss, _ = trainer.itp_test(
            args=trainer.args, model=trainer.model, x=x
        )

        logger.info(f"  Dev: {metric}={dev_performance:.4f}, Loss={valid_loss:.4f}")
        logger.info(f"  Test: {metric}={test_performance:.4f}, Loss={test_loss:.4f}")

        # Write results
        best_config, best_dev_performance, df = write_result(
            output_dir, result_name, dev_performance, best_dev_performance,
            valid_loss, test_performance, test_loss, args, df, prefix, metric, x
        )

    # Summary
    logger.info("="*60)
    logger.info("INTERPOLATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Endpoint 1 (x=0.0): {metric}={left_dev:.4f}")
    logger.info(f"Endpoint 2 (x=1.0): {metric}={right_dev:.4f}")
    logger.info(f"Best performance: {metric}={best_dev_performance:.4f}")
    logger.info(f"Results saved to: {os.path.join(output_dir, result_name)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
