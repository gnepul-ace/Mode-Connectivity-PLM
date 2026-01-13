"""
Reasoning Dataset Loader for GSM8K and MATH
Follows the structure of fewshot_gym_singletask_t5.py but for decoder LLMs
"""

import os
import re
import json
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch


class ReasoningData:
    """
    Data loader for reasoning benchmarks (GSM8K, MATH)
    Similar structure to NLPFewshotGymSingleTaskData
    """

    def __init__(self, logger, args, dataset_name, split, tokenizer, chat_handler):
        self.logger = logger
        self.args = args
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.chat_handler = chat_handler

        self.dataset = None
        self.dataloader = None

    def load_dataset(self):
        """Load GSM8K or MATH dataset from HuggingFace"""
        self.logger.info(f"Loading {self.dataset_name} dataset, split={self.split}")

        if self.dataset_name.lower() == "gsm8k":
            # Load GSM8K
            raw_dataset = load_dataset("gsm8k", "main", split=self.split)
            self.dataset = GSM8KDataset(
                raw_dataset,
                self.tokenizer,
                self.chat_handler,
                max_length=self.args.max_input_length,
                max_output_length=self.args.max_output_length,
                is_training=(self.split == "train")
            )

        elif self.dataset_name.lower() == "math":
            # Load MATH
            raw_dataset = load_dataset("hendrycks/competition_math", split=self.split)
            self.dataset = MATHDataset(
                raw_dataset,
                self.tokenizer,
                self.chat_handler,
                max_length=self.args.max_input_length,
                max_output_length=self.args.max_output_length,
                is_training=(self.split == "train")
            )

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.logger.info(f"Loaded {len(self.dataset)} examples")

    def load_dataloader(self):
        """Create DataLoader"""
        batch_size = self.args.train_batch_size if self.split == "train" else self.args.eval_batch_size

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(self.split == "train"),
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=True
        )

    def collate_fn(self, batch):
        """Collate function for batching"""
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        answers = [item['answer'] for item in batch]
        questions = [item['question'] for item in batch]

        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        labels = []

        for i, ids in enumerate(input_ids):
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(attention_mask[i] + [0] * padding_length)

            # For training, create labels (mask input, keep output)
            if 'labels' in batch[i]:
                labels.append(batch[i]['labels'] + [-100] * padding_length)

        result = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'answer': answers,
            'question': questions,
        }

        if labels:
            result['labels'] = torch.tensor(labels, dtype=torch.long)

        return result

    def evaluate(self, predictions, references):
        """
        Evaluate predictions against references
        Returns metrics dict
        """
        from dataloader.reasoning.reasoning_metrics import evaluate_reasoning

        metrics = evaluate_reasoning(
            predictions,
            references,
            dataset_type=self.dataset_name
        )

        return metrics

    def itp_evaluate(self, predictions, references):
        """
        Evaluation for interpolation analysis
        Returns metrics and per-sample scores
        """
        from dataloader.reasoning.reasoning_metrics import evaluate_reasoning_with_scores

        metrics, raw_scores = evaluate_reasoning_with_scores(
            predictions,
            references,
            dataset_type=self.dataset_name
        )

        return metrics, raw_scores


class GSM8KDataset(Dataset):
    """GSM8K Dataset"""

    def __init__(self, raw_dataset, tokenizer, chat_handler, max_length=512, max_output_length=512, is_training=False):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.chat_handler = chat_handler
        self.max_length = max_length
        self.max_output_length = max_output_length
        self.is_training = is_training

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        question = item['question']
        answer = item['answer']

        # Extract numerical answer from "#### number" format
        numerical_answer = self._extract_numerical_answer(answer)

        # Format with chat template (adds "Let's think step by step")
        if self.is_training:
            # Training: include full solution
            formatted_text = self.chat_handler.format_chat_completion(question, answer)
        else:
            # Inference: only question with CoT prompt
            formatted_text = self.chat_handler.format_chat_with_cot(question)

        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        result = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'answer': numerical_answer,
            'question': question,
        }

        # For training, add labels
        if self.is_training:
            result['labels'] = encodings['input_ids'].copy()

        return result

    def _extract_numerical_answer(self, answer_text):
        """Extract numerical answer from GSM8K format: '#### number'"""
        if "####" in answer_text:
            numerical_answer = answer_text.split("####")[-1].strip()
        else:
            # Fallback: find last number
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            numerical_answer = numbers[-1] if numbers else ""

        return numerical_answer.replace(",", "").strip()


class MATHDataset(Dataset):
    """MATH Dataset"""

    def __init__(self, raw_dataset, tokenizer, chat_handler, max_length=512, max_output_length=512, is_training=False):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.chat_handler = chat_handler
        self.max_length = max_length
        self.max_output_length = max_output_length
        self.is_training = is_training

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        question = item['problem']
        solution = item['solution']

        # Extract boxed answer
        boxed_answer = self._extract_boxed_answer(solution)

        # Format with chat template
        if self.is_training:
            formatted_text = self.chat_handler.format_chat_completion(question, solution)
        else:
            formatted_text = self.chat_handler.format_chat_with_cot(question)

        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        result = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'answer': boxed_answer,
            'question': question,
        }

        if self.is_training:
            result['labels'] = encodings['input_ids'].copy()

        return result

    def _extract_boxed_answer(self, solution):
        """Extract answer from \\boxed{} format"""
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, solution)

        if matches:
            return matches[-1].strip()

        # Fallback: last line
        lines = solution.strip().split('\n')
        return lines[-1].strip() if lines else ""
