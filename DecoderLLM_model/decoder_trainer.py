"""
Trainer for decoder-only LLMs with reasoning benchmarks.
Supports:
- Supervised fine-tuning (SFT)
- Mode connectivity analysis
- Reasoning evaluation (GSM8K, MATH)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import json
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from .modeling_decoder_llm import DecoderLLMWithPET
from .chat_template import ChatTemplateHandler
from .reasoning_metrics import ReasoningMetrics


class DecoderLLMTrainer:
    """
    Trainer for decoder-only LLMs on reasoning tasks.
    """

    def __init__(
        self,
        model: DecoderLLMWithPET,
        tokenizer,
        chat_template_handler: ChatTemplateHandler,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        output_dir: str = "./outputs",
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        max_steps: int = 10000,
        gradient_accumulation_steps: int = 1,
        fp16: bool = False,
        bf16: bool = True,
    ):
        """
        Args:
            model: DecoderLLMWithPET instance
            tokenizer: HuggingFace tokenizer
            chat_template_handler: ChatTemplateHandler
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            test_dataloader: Test data loader
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_grad_norm: Max gradient norm for clipping
            device: Device to train on
            output_dir: Directory to save outputs
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            max_steps: Maximum training steps
            gradient_accumulation_steps: Gradient accumulation steps
            fp16: Use FP16 training
            bf16: Use BF16 training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template_handler = chat_template_handler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.output_dir = Path(output_dir)
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.bf16 = bf16

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        if train_dataloader:
            self._setup_optimizer()

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None

        # Training state
        self.global_step = 0
        self.best_eval_metric = 0.0

        print(f"Trainer initialized. Output directory: {self.output_dir}")
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Get trainable parameters
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )

    def train(self):
        """Train the model"""
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is required for training")

        print(f"Starting training for {self.max_steps} steps...")

        self.model.train()
        train_iterator = iter(self.train_dataloader)
        total_loss = 0.0
        log_loss = 0.0

        progress_bar = tqdm(range(self.max_steps), desc="Training")

        for step in progress_bar:
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Restart iterator
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            # Forward pass with mixed precision
            if self.bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
            elif self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.gradient_accumulation_steps

            # Backward pass
            if self.fp16 and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()
            log_loss += loss.item()

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.fp16 and self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                if self.fp16 and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Logging
            if (step + 1) % self.logging_steps == 0:
                avg_loss = log_loss / self.logging_steps
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                log_loss = 0.0

            # Evaluation
            if (step + 1) % self.eval_steps == 0 and self.eval_dataloader:
                print(f"\nEvaluating at step {self.global_step}...")
                eval_metrics = self.evaluate(self.eval_dataloader)
                print(f"Eval metrics: {eval_metrics}")

                # Save best model
                if eval_metrics.get("accuracy", 0) > self.best_eval_metric:
                    self.best_eval_metric = eval_metrics["accuracy"]
                    self.save_checkpoint("best_model")
                    print(f"New best model saved! Accuracy: {self.best_eval_metric:.4f}")

                self.model.train()

            # Save checkpoint
            if (step + 1) % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")

        print(f"\nTraining complete! Best eval accuracy: {self.best_eval_metric:.4f}")

        # Final save
        self.save_checkpoint("final_model")

        return total_loss / self.max_steps

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = False,
        num_return_sequences: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate the model on reasoning tasks.

        Args:
            dataloader: DataLoader to evaluate on
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate (for voting)

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        all_predictions = []
        all_references = []

        print(f"Evaluating on {len(dataloader)} batches...")

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Generate
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

            # Decode predictions
            predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Extract answers from predictions
            for i, pred in enumerate(predictions):
                # Remove the input prompt from prediction
                if i < len(batch["question"]):
                    pred_answer = self.chat_template_handler.extract_answer(pred)
                    all_predictions.append(pred_answer)

            # Get references
            if "numerical_answer" in batch:
                all_references.extend(batch["numerical_answer"])
            elif "boxed_answer" in batch:
                all_references.extend(batch["boxed_answer"])
            elif "answer" in batch:
                all_references.extend(batch["answer"])

        # Compute metrics
        # Determine dataset type
        dataset_type = "gsm8k"  # Default
        if hasattr(dataloader.dataset, '__class__'):
            if "MATH" in dataloader.dataset.__class__.__name__:
                dataset_type = "math"

        metrics = ReasoningMetrics.evaluate_batch(
            all_predictions,
            all_references,
            dataset_type=dataset_type
        )

        return metrics

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save trainable parameters only (for PET)
        if hasattr(self.model, 'get_trainable_parameters'):
            trainable_params = self.model.get_trainable_parameters()
            torch.save(trainable_params, checkpoint_dir / "trainable_params.pt")
            print(f"Saved {len(trainable_params)} trainable parameters")
        else:
            # Save full model
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "best_eval_metric": self.best_eval_metric,
        }
        if self.optimizer:
            training_state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler:
            training_state["scheduler"] = self.scheduler.state_dict()

        torch.save(training_state, checkpoint_dir / "training_state.pt")

        # Save config
        config_dict = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_steps": self.max_steps,
            "global_step": self.global_step,
        }
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        # Load trainable parameters
        if (checkpoint_path / "trainable_params.pt").exists():
            trainable_params = torch.load(checkpoint_path / "trainable_params.pt")
            if hasattr(self.model, 'load_trainable_parameters'):
                self.model.load_trainable_parameters(trainable_params)
            print(f"Loaded trainable parameters from {checkpoint_path}")
        elif (checkpoint_path / "model.pt").exists():
            self.model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
            print(f"Loaded full model from {checkpoint_path}")

        # Load training state
        if (checkpoint_path / "training_state.pt").exists():
            training_state = torch.load(checkpoint_path / "training_state.pt")
            self.global_step = training_state.get("global_step", 0)
            self.best_eval_metric = training_state.get("best_eval_metric", 0.0)

            if self.optimizer and "optimizer" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer"])
            if self.scheduler and "scheduler" in training_state:
                self.scheduler.load_state_dict(training_state["scheduler"])

            print(f"Loaded training state: step={self.global_step}, best_metric={self.best_eval_metric:.4f}")

    @torch.no_grad()
    def interpolate_evaluate(
        self,
        checkpoint_path_1: str,
        checkpoint_path_2: str,
        dataloader: DataLoader,
        num_points: int = 11,
        save_results: bool = True,
    ) -> Dict:
        """
        Perform mode connectivity analysis between two checkpoints.
        Linearly interpolate between two models and evaluate at each point.

        Args:
            checkpoint_path_1: Path to first checkpoint
            checkpoint_path_2: Path to second checkpoint
            dataloader: DataLoader for evaluation
            num_points: Number of interpolation points
            save_results: Whether to save results

        Returns:
            Dictionary with interpolation results
        """
        print(f"\n{'='*60}")
        print("MODE CONNECTIVITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Checkpoint 1: {checkpoint_path_1}")
        print(f"Checkpoint 2: {checkpoint_path_2}")
        print(f"Interpolation points: {num_points}")
        print(f"{'='*60}\n")

        # Load checkpoints
        checkpoint_1 = torch.load(Path(checkpoint_path_1) / "trainable_params.pt")
        checkpoint_2 = torch.load(Path(checkpoint_path_2) / "trainable_params.pt")

        # Interpolation coefficients
        alphas = np.linspace(0, 1, num_points)

        results = {
            "alphas": alphas.tolist(),
            "accuracies": [],
            "losses": [],
        }

        for alpha in alphas:
            print(f"\nEvaluating at α = {alpha:.2f}")

            # Interpolate parameters
            interpolated_params = {}
            for key in checkpoint_1.keys():
                if key in checkpoint_2:
                    interpolated_params[key] = (
                        (1 - alpha) * checkpoint_1[key] + alpha * checkpoint_2[key]
                    )

            # Load interpolated parameters
            if hasattr(self.model, 'load_trainable_parameters'):
                self.model.load_trainable_parameters(interpolated_params)

            # Evaluate
            metrics = self.evaluate(dataloader, do_sample=False)

            results["accuracies"].append(metrics.get("accuracy", 0.0))
            print(f"Accuracy at α={alpha:.2f}: {metrics.get('accuracy', 0.0):.4f}")

        # Save results
        if save_results:
            results_path = self.output_dir / "interpolation_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {results_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("INTERPOLATION SUMMARY")
        print(f"{'='*60}")
        print(f"Endpoint 1 (α=0.0): {results['accuracies'][0]:.4f}")
        print(f"Endpoint 2 (α=1.0): {results['accuracies'][-1]:.4f}")
        print(f"Max accuracy: {max(results['accuracies']):.4f}")
        print(f"Min accuracy: {min(results['accuracies']):.4f}")
        print(f"Mean accuracy: {np.mean(results['accuracies']):.4f}")
        print(f"{'='*60}\n")

        return results
