"""
Decoder LLM Trainer for Mode Connectivity Analysis
Based on T5_trainer.py structure, adapted for causal LLMs (Qwen, Llama)
"""

import os
import numpy as np
import torch
import random
import warnings
import json
from collections import OrderedDict
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    is_torch_available,
)
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class DecoderLLMTrainer:
    """
    Trainer for decoder-only LLMs on reasoning tasks.
    Follows the same structure as T5 Trainer for mode connectivity analysis.
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
        # Import reasoning dataset loaders
        from dataloader.reasoning.reasoning_loader import ReasoningData

        self.train_data = ReasoningData(
            logger, args, args.dataset, split="train",
            tokenizer=self.tokenizer, chat_handler=chat_template_handler
        )
        self.train_data.load_dataset()
        self.train_data.load_dataloader()

        self.dev_data = ReasoningData(
            logger, args, args.dataset, split="test",
            tokenizer=self.tokenizer, chat_handler=chat_template_handler
        )
        self.dev_data.load_dataset()
        self.dev_data.load_dataloader()

        self.test_data = self.dev_data  # For reasoning tasks, test = dev

        self.device = self.init_device()
        self.model = self.model.to(self.device)
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.init_tensorboard(args)

        if args.seed is not None:
            set_seed(args.seed)

    def init_device(self):
        if not torch.cuda.is_available():
            print('No GPU available, using CPU')
            return torch.device('cpu')
        return torch.device('cuda')

    def init_tensorboard(self, args):
        if args.tensorboard_dir:
            self.summary_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        else:
            self.summary_writer = None

    def train(self):
        """Training loop following T5 trainer structure"""
        args = self.args
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_data.dataset)}")
        self.logger.info(f"  Num Epochs = {args.train_epochs}")
        self.logger.info(f"  Batch size = {args.train_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

        # Setup optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.train_iters
        )

        self.model.train()
        num_updates = 0
        best_metric = -1.0
        early_stop = 0

        for epoch in range(args.train_epochs):
            for batch_idx, batch in enumerate(self.train_data.dataloader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    num_updates += 1

                    # Logging
                    if num_updates % args.log_interval == 0:
                        self.logger.info(f"Epoch {epoch}, Step {num_updates}, Loss: {loss.item():.4f}")

                    # Validation
                    if num_updates % args.valid_interval == 0:
                        metrics = self.valid(epoch, num_updates)
                        current_metric = sum(metrics.values()) / len(metrics)

                        if current_metric > best_metric:
                            best_metric = current_metric
                            early_stop = 0
                            self.save_checkpoint(f"{args.output_dir}/checkpoint-best.pt", epoch, num_updates)
                        else:
                            early_stop += 1

                        if args.early_stop > 0 and early_stop >= args.early_stop:
                            self.logger.info(f"Early stopping at step {num_updates}")
                            return best_metric

                    # Save checkpoint
                    if args.output_interval and num_updates % args.output_interval == 0:
                        self.save_checkpoint(f"{args.output_dir}/checkpoint@{num_updates}.pt", epoch, num_updates)

                    if num_updates >= args.train_iters:
                        break

            if num_updates >= args.train_iters:
                break

        # Save final checkpoint
        self.save_checkpoint(f"{args.output_dir}/checkpoint-last.pt", epoch, num_updates)
        return best_metric

    def valid(self, epoch=0, num_updates=0):
        """Validation following T5 trainer structure"""
        self.model.eval()
        my_predictions = []
        references = []

        self.logger.info(f"Begin validation on {len(self.dev_data.dataset)} samples ...")

        with torch.no_grad():
            for batch in self.dev_data.dataloader:
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
        metrics = self.dev_data.evaluate(my_predictions, references)

        self.logger.info(f"Validation at epoch {epoch}, step {num_updates}: {metrics}")
        self.model.train()
        return metrics

    def itp_valid(self, x=0.0):
        """
        Interpolation validation - called during mode connectivity analysis
        Similar to valid() but returns raw scores for analysis
        """
        self.model.eval()
        my_predictions = []
        references = []

        with torch.no_grad():
            for batch in self.dev_data.dataloader:
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

        # Evaluate
        metrics, raw_scores = self.dev_data.itp_evaluate(my_predictions, references)

        # Calculate performance (main metric)
        metric_name = list(metrics.keys())[0]
        performance = metrics[metric_name]

        return metric_name, performance, 0.0, raw_scores  # metric, perf, loss, scores

    def itp_test(self, args, model, x=0.0):
        """Interpolation test - similar to itp_valid"""
        return self.itp_valid(x=x)

    def save_checkpoint(self, save_path, epoch, num_updates):
        """Save checkpoint following T5 trainer structure"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # For PET methods, only save trainable parameters
        if self.args.tune_method in ['lora', 'adapter']:
            state_dict = {k: v for k, v in self.model.state_dict().items() if v.requires_grad}
            checkpoint = {
                self.args.tune_method: state_dict,
                'epoch': epoch,
                'num_updates': num_updates
            }
        else:
            checkpoint = {
                'model': self.model.state_dict(),
                'epoch': epoch,
                'num_updates': num_updates
            }

        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path):
        """Load checkpoint"""
        self.logger.info(f"Loading checkpoint from {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device)

        if self.args.tune_method in checkpoint:
            state_dict = checkpoint[self.args.tune_method]
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        model_dict = self.model.state_dict()
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)

    def add_logging(self, log_dict, key, value):
        """Add value to logging dict"""
        if key not in log_dict:
            log_dict[key] = []
        log_dict[key].append(value)

    def log_step(self, log_dict, suffix="", tensorboard_suffix='', epoch=0, num_updates=0, **kwargs):
        """Log step information"""
        loss = np.mean(log_dict.get('loss', [0]))
        msg = f"{suffix} Epoch {epoch}, Step {num_updates}, Loss: {loss:.4f}"

        for k, v in kwargs.items():
            msg += f", {k}: {v:.4f}"

        self.logger.info(msg)
        return loss
