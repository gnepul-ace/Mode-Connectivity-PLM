"""
Evaluation metrics for reasoning benchmarks.
Implements metrics used in DAPO, DR GRPO, and other RL reasoning papers.
"""

import re
import string
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from collections import Counter


class ReasoningMetrics:
    """Metrics for evaluating reasoning tasks"""

    @staticmethod
    def normalize_number(text: str) -> Optional[float]:
        """
        Normalize a number string to float.
        Handles common formats: integers, decimals, fractions, percentages, etc.

        Args:
            text: Text containing a number

        Returns:
            Normalized float or None if cannot parse
        """
        if not text:
            return None

        # Remove common formatting
        text = text.strip()
        text = text.replace(",", "")  # Remove thousand separators
        text = text.replace("$", "")  # Remove dollar signs
        text = text.replace("%", "")  # Remove percent signs

        # Handle fractions like "1/2"
        if "/" in text and text.count("/") == 1:
            try:
                parts = text.split("/")
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                return numerator / denominator if denominator != 0 else None
            except:
                pass

        # Try to parse as float
        try:
            return float(text)
        except:
            # Try to extract first number from text
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    pass

        return None

    @staticmethod
    def extract_answer_number(text: str) -> Optional[str]:
        """
        Extract the final numerical answer from generated text.
        Looks for patterns like:
        - "The answer is 42"
        - "#### 42" (GSM8K format)
        - "\\boxed{42}" (MATH format)
        - Final number in the text

        Args:
            text: Generated text

        Returns:
            Extracted answer as string or None
        """
        if not text:
            return None

        # Pattern 1: GSM8K format "#### answer"
        gsm8k_pattern = r'####\s*([^\n]+)'
        matches = re.findall(gsm8k_pattern, text)
        if matches:
            return matches[-1].strip()

        # Pattern 2: LaTeX boxed format "\\boxed{answer}"
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, text)
        if matches:
            return matches[-1].strip()

        # Pattern 3: "The answer is X" or "the final answer is X"
        answer_patterns = [
            r'[Tt]he (?:final )?answer is[:\s]+([^\n\.]+)',
            r'[Aa]nswer[:\s]+([^\n\.]+)',
            r'[Ss]olution[:\s]+([^\n\.]+)',
        ]
        for pattern in answer_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()

        # Pattern 4: Last number in the text
        # Split by lines and check last few lines
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in reversed(lines[-3:]):  # Check last 3 lines
            numbers = re.findall(r'-?\d+\.?\d*', line)
            if numbers:
                return numbers[-1]

        # Fallback: any number in text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        return None

    @staticmethod
    def numerical_match(prediction: str, reference: str, tolerance: float = 1e-4) -> bool:
        """
        Check if predicted number matches reference (within tolerance).

        Args:
            prediction: Predicted answer text
            reference: Ground truth answer text
            tolerance: Numerical tolerance for floating point comparison

        Returns:
            True if answers match
        """
        pred_num = ReasoningMetrics.normalize_number(prediction)
        ref_num = ReasoningMetrics.normalize_number(reference)

        if pred_num is None or ref_num is None:
            # Fallback to string match
            return ReasoningMetrics.normalize_text(prediction) == ReasoningMetrics.normalize_text(reference)

        # Check numerical equality within tolerance
        return abs(pred_num - ref_num) <= tolerance

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison.
        - Lowercase
        - Remove punctuation
        - Remove articles (a, an, the)
        - Strip whitespace

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)

        # Collapse whitespace
        text = ' '.join(text.split())

        return text.strip()

    @staticmethod
    def exact_match(prediction: str, reference: str) -> bool:
        """
        Check exact match between prediction and reference (after normalization).

        Args:
            prediction: Predicted text
            reference: Reference text

        Returns:
            True if exact match
        """
        return ReasoningMetrics.normalize_text(prediction) == ReasoningMetrics.normalize_text(reference)

    @staticmethod
    def f1_score(prediction: str, reference: str) -> float:
        """
        Token-level F1 score (used in SQuAD-style QA).

        Args:
            prediction: Predicted text
            reference: Reference text

        Returns:
            F1 score (0-1)
        """
        pred_tokens = ReasoningMetrics.normalize_text(prediction).split()
        ref_tokens = ReasoningMetrics.normalize_text(reference).split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return float(len(pred_tokens) == len(ref_tokens))

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    @staticmethod
    def evaluate_gsm8k(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate on GSM8K dataset.

        Args:
            predictions: List of predicted answers
            references: List of ground truth answers (numerical or text)

        Returns:
            Dictionary with metrics
        """
        assert len(predictions) == len(references), "Predictions and references must have same length"

        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            # Extract numerical answer from prediction
            pred_answer = ReasoningMetrics.extract_answer_number(pred)

            if pred_answer and ReasoningMetrics.numerical_match(pred_answer, ref):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

    @staticmethod
    def evaluate_math(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate on MATH dataset.

        Args:
            predictions: List of predicted answers
            references: List of ground truth answers (from \\boxed{})

        Returns:
            Dictionary with metrics
        """
        assert len(predictions) == len(references), "Predictions and references must have same length"

        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            # Extract answer from prediction
            pred_answer = ReasoningMetrics.extract_answer_number(pred)

            # Try numerical match first
            if pred_answer and ReasoningMetrics.numerical_match(pred_answer, ref):
                correct += 1
            # Fallback to exact string match
            elif pred_answer and ReasoningMetrics.exact_match(pred_answer, ref):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

    @staticmethod
    def evaluate_batch(
        predictions: List[str],
        references: List[str],
        dataset_type: str = "gsm8k"
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.

        Args:
            predictions: Predicted answers
            references: Ground truth answers
            dataset_type: 'gsm8k' or 'math'

        Returns:
            Metrics dictionary
        """
        if dataset_type.lower() == "gsm8k":
            return ReasoningMetrics.evaluate_gsm8k(predictions, references)
        elif dataset_type.lower() == "math":
            return ReasoningMetrics.evaluate_math(predictions, references)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


class MajorityVoting:
    """
    Majority voting for multiple generations (used in reasoning tasks).
    Common technique: Generate multiple solutions and vote on final answer.
    """

    @staticmethod
    def vote(predictions: List[str], method: str = "numerical") -> str:
        """
        Perform majority voting on multiple predictions.

        Args:
            predictions: List of predicted answers
            method: 'numerical' or 'text'

        Returns:
            Most common answer
        """
        if not predictions:
            return ""

        if method == "numerical":
            # Extract numerical answers and vote
            numbers = []
            for pred in predictions:
                answer = ReasoningMetrics.extract_answer_number(pred)
                if answer:
                    num = ReasoningMetrics.normalize_number(answer)
                    if num is not None:
                        numbers.append(num)

            if numbers:
                # Count most common
                counter = Counter(numbers)
                most_common = counter.most_common(1)[0][0]
                return str(most_common)

        # Fallback to text voting
        normalized = [ReasoningMetrics.normalize_text(p) for p in predictions]
        if normalized:
            counter = Counter(normalized)
            most_common = counter.most_common(1)[0][0]
            return most_common

        return predictions[0] if predictions else ""


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric.
    Used in code generation and reasoning tasks.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k

    Returns:
        pass@k score
    """
    if n - c < k:
        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
