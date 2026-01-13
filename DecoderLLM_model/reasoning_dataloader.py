"""
Data loaders for reasoning benchmarks (GSM8K, MATH).
Follows evaluation protocols from DAPO and DR GRPO papers.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
import re


class GSM8KDataset(Dataset):
    """
    GSM8K (Grade School Math 8K) dataset.
    8,500 grade school math word problems requiring 2-8 steps.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        chat_template_handler=None,
        max_length: int = 2048,
        use_cot: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            split: 'train' or 'test'
            tokenizer: HuggingFace tokenizer
            chat_template_handler: ChatTemplateHandler instance
            max_length: Max sequence length
            use_cot: Whether to add "Let's think step by step" prompt
            cache_dir: Cache directory for dataset
        """
        self.split = split
        self.tokenizer = tokenizer
        self.chat_template_handler = chat_template_handler
        self.max_length = max_length
        self.use_cot = use_cot

        # Load dataset from HuggingFace
        self.dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

        print(f"Loaded GSM8K {split} split: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        """Get a single example"""
        item = self.dataset[idx]

        question = item["question"]
        answer = item["answer"]

        # Extract numerical answer from the solution
        # GSM8K answers format: "<<reasoning>> #### final_answer"
        numerical_answer = self._extract_numerical_answer(answer)

        # Format with chat template
        if self.chat_template_handler:
            if self.use_cot:
                formatted_question = self.chat_template_handler.format_chat_with_cot(question)
            else:
                formatted_question = self.chat_template_handler.format_chat(question)

            # For training, include the full solution
            if self.split == "train" and self.tokenizer:
                encodings = self.chat_template_handler.tokenize_chat(
                    question=question,
                    answer=answer,
                    max_length=self.max_length
                )
            else:
                # For evaluation, only encode question
                if self.tokenizer:
                    encodings = self.tokenizer(
                        formatted_question,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None
                    )
                else:
                    encodings = {"text": formatted_question}
        else:
            # Fallback without chat template
            encodings = {
                "text": question,
                "answer": answer,
                "numerical_answer": numerical_answer
            }

        # Add metadata
        encodings["question"] = question
        encodings["answer"] = answer
        encodings["numerical_answer"] = numerical_answer
        encodings["idx"] = idx

        return encodings

    def _extract_numerical_answer(self, answer_text: str) -> str:
        """Extract numerical answer from GSM8K answer format"""
        # GSM8K format: "reasoning #### final_answer"
        if "####" in answer_text:
            numerical_answer = answer_text.split("####")[-1].strip()
        else:
            # Fallback: try to find last number in text
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            numerical_answer = numbers[-1] if numbers else ""

        # Clean answer
        numerical_answer = numerical_answer.replace(",", "").strip()

        return numerical_answer


class MATHDataset(Dataset):
    """
    MATH dataset (competition mathematics).
    12,500 problems from mathematics competitions (AMC, AIME).
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        chat_template_handler=None,
        max_length: int = 2048,
        use_cot: bool = True,
        cache_dir: Optional[str] = None,
        difficulty: Optional[str] = None  # Filter by difficulty: 1-5
    ):
        """
        Args:
            split: 'train' or 'test'
            tokenizer: HuggingFace tokenizer
            chat_template_handler: ChatTemplateHandler instance
            max_length: Max sequence length
            use_cot: Whether to add CoT prompt
            cache_dir: Cache directory
            difficulty: Optional difficulty level filter (1-5)
        """
        self.split = split
        self.tokenizer = tokenizer
        self.chat_template_handler = chat_template_handler
        self.max_length = max_length
        self.use_cot = use_cot

        # Load MATH dataset
        self.dataset = load_dataset("hendrycks/competition_math", split=split, cache_dir=cache_dir)

        # Filter by difficulty if specified
        if difficulty is not None:
            self.dataset = self.dataset.filter(lambda x: x["level"] == f"Level {difficulty}")

        print(f"Loaded MATH {split} split: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        """Get a single example"""
        item = self.dataset[idx]

        question = item["problem"]
        answer = item["solution"]
        level = item["level"]
        subject = item["type"]

        # Extract boxed answer
        boxed_answer = self._extract_boxed_answer(answer)

        # Format with chat template
        if self.chat_template_handler:
            if self.use_cot:
                formatted_question = self.chat_template_handler.format_chat_with_cot(question)
            else:
                formatted_question = self.chat_template_handler.format_chat(question)

            # For training
            if self.split == "train" and self.tokenizer:
                encodings = self.chat_template_handler.tokenize_chat(
                    question=question,
                    answer=answer,
                    max_length=self.max_length
                )
            else:
                # For evaluation
                if self.tokenizer:
                    encodings = self.tokenizer(
                        formatted_question,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None
                    )
                else:
                    encodings = {"text": formatted_question}
        else:
            encodings = {
                "text": question,
                "answer": answer,
                "boxed_answer": boxed_answer
            }

        # Add metadata
        encodings["question"] = question
        encodings["answer"] = answer
        encodings["boxed_answer"] = boxed_answer
        encodings["level"] = level
        encodings["subject"] = subject
        encodings["idx"] = idx

        return encodings

    def _extract_boxed_answer(self, solution: str) -> str:
        """Extract the final answer from \\boxed{} in LaTeX"""
        # Match \boxed{...}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, solution)

        if matches:
            # Return last boxed answer
            return matches[-1].strip()

        # Fallback: return last line
        lines = solution.strip().split('\n')
        return lines[-1].strip() if lines else ""


class ReasoningDataCollator:
    """
    Custom collator for reasoning datasets.
    Handles padding and label creation for causal LM training.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples"""
        # Separate numerical features from metadata
        batch = {}
        metadata = {}

        # Extract metadata
        for key in ["question", "answer", "numerical_answer", "boxed_answer", "level", "subject", "idx"]:
            if key in features[0]:
                metadata[key] = [f[key] for f in features]

        # Handle input_ids and attention_mask
        if "input_ids" in features[0]:
            # Pad sequences
            input_ids = [f["input_ids"] for f in features]
            max_length = max(len(ids) for ids in input_ids)

            if self.pad_to_multiple_of:
                max_length = ((max_length + self.pad_to_multiple_of - 1) //
                             self.pad_to_multiple_of * self.pad_to_multiple_of)

            padded_input_ids = []
            attention_masks = []
            labels = []

            for f in features:
                ids = f["input_ids"]
                padding_length = max_length - len(ids)

                # Pad input_ids
                padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                padded_input_ids.append(padded_ids)

                # Create attention mask
                mask = [1] * len(ids) + [0] * padding_length
                attention_masks.append(mask)

                # Create labels (for training)
                if "labels" in f:
                    lbls = f["labels"] + [-100] * padding_length  # -100 is ignored in loss
                    labels.append(lbls)

            batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
            batch["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)

            if labels:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)

        # Add metadata
        batch.update(metadata)

        return batch


def create_reasoning_dataloader(
    dataset_name: str,
    split: str,
    tokenizer,
    chat_template_handler,
    batch_size: int = 8,
    max_length: int = 2048,
    use_cot: bool = True,
    shuffle: bool = None,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for reasoning benchmarks.

    Args:
        dataset_name: 'gsm8k' or 'math'
        split: 'train' or 'test'
        tokenizer: HuggingFace tokenizer
        chat_template_handler: ChatTemplateHandler instance
        batch_size: Batch size
        max_length: Max sequence length
        use_cot: Use Chain-of-Thought prompting
        shuffle: Whether to shuffle (default: True for train, False for test)
        num_workers: Number of data loading workers
        cache_dir: Cache directory
        **kwargs: Additional dataset-specific arguments

    Returns:
        DataLoader instance
    """
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == "train")

    # Create dataset
    if dataset_name.lower() == "gsm8k":
        dataset = GSM8KDataset(
            split=split,
            tokenizer=tokenizer,
            chat_template_handler=chat_template_handler,
            max_length=max_length,
            use_cot=use_cot,
            cache_dir=cache_dir
        )
    elif dataset_name.lower() == "math":
        dataset = MATHDataset(
            split=split,
            tokenizer=tokenizer,
            chat_template_handler=chat_template_handler,
            max_length=max_length,
            use_cot=use_cot,
            cache_dir=cache_dir,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create collator
    collator = ReasoningDataCollator(tokenizer)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    return dataloader
