"""
Simplified Decoder LLM Evaluator for Mode Connectivity Analysis
Only evaluation and interpolation - NO training
"""

import os
import numpy as np
import torch
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")


class DecoderLLMEvaluator:
    """
    Evaluator for decoder-only LLMs on reasoning tasks.
    Only for evaluation and mode connectivity analysis - no training.
    """

    def __init__(self, args, logger, model_provider, chat_template_handler=None):
        self.args = args
        self.logger = logger

        logger.info("Loading model ...")
        self.model, self.config, self.tokenizer = model_provider(args)

        # Create chat template handler if not provided
        if chat_template_handler is None:
            from .chat_template import ChatTemplateHandler
            chat_template_handler = ChatTemplateHandler(args.model, self.tokenizer)

        self.chat_template_handler = chat_template_handler

        logger.info("Loading Dataset ...")
        from dataloader.reasoning.reasoning_loader import ReasoningData

        # Only load test data for evaluation
        self.test_data = ReasoningData(
            logger, args, args.dataset, split="test",
            tokenizer=self.tokenizer, chat_handler=chat_template_handler
        )
        self.test_data.load_dataset()
        self.test_data.load_dataloader()

        self.device = self.init_device()
        self.model = self.model.to(self.device)

    def init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('No GPU available, using CPU')
            return torch.device('cpu')
        return torch.device('cuda')

    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        my_predictions = []
        references = []

        self.logger.info(f"Evaluating on {len(self.test_data.dataset)} samples ...")

        with torch.no_grad():
            for batch in self.test_data.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Generate
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.args.max_output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode
                gen_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                my_predictions.extend(gen_text)
                references.extend(batch['answer'])

        # Evaluate
        metrics = self.test_data.evaluate(my_predictions, references)

        self.logger.info(f"Evaluation results: {metrics}")
        return metrics

    def itp_valid(self, x=0.0):
        """
        Interpolation evaluation - for mode connectivity analysis
        Returns: metric_name, performance, loss, raw_scores
        """
        self.model.eval()
        my_predictions = []
        references = []

        with torch.no_grad():
            for batch in self.test_data.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.args.max_output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                gen_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                my_predictions.extend(gen_text)
                references.extend(batch['answer'])

        # Evaluate with per-sample scores
        metrics, raw_scores = self.test_data.itp_evaluate(my_predictions, references)

        # Get main metric
        metric_name = list(metrics.keys())[0]
        performance = metrics[metric_name]

        return metric_name, performance, 0.0, raw_scores

    def itp_test(self, args, model, x=0.0):
        """Interpolation test - same as itp_valid"""
        return self.itp_valid(x=x)

    def load_checkpoint(self, checkpoint_path, key=None):
        """
        Load checkpoint from path
        Supports: LoRA weights, adapter weights, or full model

        Args:
            checkpoint_path: Path to checkpoint file
            key: Optional key to extract from checkpoint (e.g., 'lora', 'adapter')
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Determine what type of checkpoint this is
        if key and key in checkpoint:
            state_dict = checkpoint[key]
        elif self.args.tune_method in checkpoint:
            state_dict = checkpoint[self.args.tune_method]
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint

        # Load state dict
        if hasattr(self.model, 'load_state_dict'):
            try:
                self.model.load_state_dict(state_dict, strict=False)
                self.logger.info("Loaded checkpoint successfully")
            except Exception as e:
                self.logger.warning(f"Loading with strict=False: {e}")
                # Try to update only matching keys
                model_dict = self.model.state_dict()
                model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
                self.model.load_state_dict(model_dict)
        else:
            self.logger.error("Model doesn't have load_state_dict method")
