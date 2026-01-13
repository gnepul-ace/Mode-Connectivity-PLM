"""
Evaluation metrics for reasoning tasks (GSM8K, MATH)
Similar to metrics_t5.py but for reasoning benchmarks
"""

import re
import string


def normalize_number(text):
    """Normalize a number string to float for comparison"""
    if not text:
        return None

    text = text.strip().replace(",", "").replace("$", "").replace("%", "")

    # Handle fractions
    if "/" in text and text.count("/") == 1:
        try:
            parts = text.split("/")
            return float(parts[0]) / float(parts[1])
        except:
            pass

    # Try to parse as float
    try:
        return float(text)
    except:
        # Extract first number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass

    return None


def extract_answer_from_generation(text):
    """
    Extract the answer from generated text
    Looks for patterns like #### answer or \\boxed{answer} or "the answer is X"
    """
    if not text:
        return None

    # Pattern 1: #### answer (GSM8K format)
    match = re.search(r'####\s*([^\n]+)', text)
    if match:
        return match.group(1).strip()

    # Pattern 2: \\boxed{answer} (MATH format)
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Pattern 3: "the answer is X"
    patterns = [
        r'[Tt]he (?:final )?answer is[:\s]+([^\n\.]+)',
        r'[Aa]nswer[:\s]+([^\n\.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Pattern 4: Last number in text
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in reversed(lines[-3:]):
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if numbers:
            return numbers[-1]

    # Fallback: any number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return None


def numerical_match(prediction, reference, tolerance=1e-4):
    """Check if predicted number matches reference within tolerance"""
    pred_num = normalize_number(prediction)
    ref_num = normalize_number(reference)

    if pred_num is None or ref_num is None:
        # Fallback to string comparison
        return normalize_text(prediction) == normalize_text(reference)

    return abs(pred_num - ref_num) <= tolerance


def normalize_text(text):
    """Normalize text for comparison"""
    if not text:
        return ""

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())

    return text.strip()


def evaluate_gsm8k(predictions, references):
    """Evaluate GSM8K predictions"""
    correct = 0
    scores = []

    for pred, ref in zip(predictions, references):
        # Extract answer from prediction
        pred_answer = extract_answer_from_generation(pred)

        # Check if correct
        is_correct = numerical_match(pred_answer, ref) if pred_answer else False
        scores.append(1 if is_correct else 0)

        if is_correct:
            correct += 1

    accuracy = correct / len(predictions) if predictions else 0.0

    return {"accuracy": accuracy}, scores


def evaluate_math(predictions, references):
    """Evaluate MATH predictions"""
    correct = 0
    scores = []

    for pred, ref in zip(predictions, references):
        pred_answer = extract_answer_from_generation(pred)

        # Try numerical match first, then string match
        is_correct = False
        if pred_answer:
            if numerical_match(pred_answer, ref):
                is_correct = True
            elif normalize_text(pred_answer) == normalize_text(ref):
                is_correct = True

        scores.append(1 if is_correct else 0)

        if is_correct:
            correct += 1

    accuracy = correct / len(predictions) if predictions else 0.0

    return {"accuracy": accuracy}, scores


def evaluate_reasoning(predictions, references, dataset_type="gsm8k"):
    """
    Unified evaluation function for reasoning tasks
    Returns only metrics dict
    """
    if dataset_type.lower() == "gsm8k":
        metrics, _ = evaluate_gsm8k(predictions, references)
    elif dataset_type.lower() == "math":
        metrics, _ = evaluate_math(predictions, references)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return metrics


def evaluate_reasoning_with_scores(predictions, references, dataset_type="gsm8k"):
    """
    Unified evaluation function for reasoning tasks
    Returns metrics dict and per-sample scores (for interpolation analysis)
    """
    if dataset_type.lower() == "gsm8k":
        metrics, scores = evaluate_gsm8k(predictions, references)
    elif dataset_type.lower() == "math":
        metrics, scores = evaluate_math(predictions, references)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return metrics, scores
