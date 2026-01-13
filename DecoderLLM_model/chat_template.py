"""
Chat template utilities for different decoder-only LLMs.
Handles proper formatting for Qwen, Llama, and other models.
"""

from typing import List, Dict, Optional
from transformers import AutoTokenizer


class ChatTemplateHandler:
    """
    Handles chat templates for different LLM models.
    Important: Different models use different chat formats (like DAPO/DR GRPO).
    """

    # Model-specific system prompts for reasoning tasks
    REASONING_SYSTEM_PROMPTS = {
        "qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that excels at mathematical reasoning and problem solving.",
        "llama": "You are a helpful, respectful and honest assistant. You excel at mathematical reasoning and step-by-step problem solving.",
        "default": "You are a helpful AI assistant that excels at reasoning and problem solving.",
    }

    def __init__(self, model_name_or_path: str, tokenizer: Optional[AutoTokenizer] = None):
        """
        Args:
            model_name_or_path: HuggingFace model identifier or path
            tokenizer: Optional tokenizer instance (will load if not provided)
        """
        self.model_name = model_name_or_path.lower()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect model type from name"""
        if "qwen" in self.model_name:
            return "qwen"
        elif "llama" in self.model_name:
            return "llama"
        else:
            return "default"

    def get_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """Get system prompt for the model"""
        if custom_prompt:
            return custom_prompt
        return self.REASONING_SYSTEM_PROMPTS.get(
            self.model_type,
            self.REASONING_SYSTEM_PROMPTS["default"]
        )

    def format_chat(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = True
    ) -> str:
        """
        Format a single question into chat format.

        Args:
            question: The question/instruction
            system_prompt: Optional custom system prompt
            add_generation_prompt: Whether to add generation prompt

        Returns:
            Formatted chat string
        """
        system = self.get_system_prompt(system_prompt)

        # Build messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        else:
            # Fallback to manual formatting
            formatted = self._manual_format(messages, add_generation_prompt)

        return formatted

    def format_chat_with_cot(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        cot_instruction: str = "Let's think step by step."
    ) -> str:
        """
        Format question with Chain-of-Thought instruction.
        This is important for reasoning tasks like in DAPO/DR GRPO.

        Args:
            question: The question/instruction
            system_prompt: Optional custom system prompt
            cot_instruction: Chain-of-thought instruction

        Returns:
            Formatted chat string with CoT
        """
        # Append CoT instruction to question
        question_with_cot = f"{question}\n\n{cot_instruction}"

        return self.format_chat(question_with_cot, system_prompt)

    def format_chat_completion(
        self,
        question: str,
        answer: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format a complete Q&A pair (for training).

        Args:
            question: The question/instruction
            answer: The answer/response
            system_prompt: Optional custom system prompt

        Returns:
            Formatted chat string including answer
        """
        system = self.get_system_prompt(system_prompt)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            formatted = self._manual_format(messages, add_generation_prompt=False)

        return formatted

    def _manual_format(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        """Manual chat formatting for models without chat template"""
        if self.model_type == "llama":
            return self._format_llama(messages, add_generation_prompt)
        elif self.model_type == "qwen":
            return self._format_qwen(messages, add_generation_prompt)
        else:
            return self._format_default(messages, add_generation_prompt)

    def _format_llama(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        """Llama chat format"""
        formatted = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

        if add_generation_prompt:
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted

    def _format_qwen(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        """Qwen chat format"""
        formatted = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        return formatted

    def _format_default(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        """Default chat format"""
        formatted = ""

        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"{role}: {content}\n\n"

        if add_generation_prompt:
            formatted += "Assistant: "

        return formatted

    def tokenize_chat(
        self,
        question: str,
        answer: Optional[str] = None,
        max_length: int = 2048,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Tokenize chat for training/inference.

        Args:
            question: The question
            answer: Optional answer (for training)
            max_length: Maximum sequence length
            system_prompt: Optional system prompt

        Returns:
            Dictionary with input_ids, attention_mask, and optionally labels
        """
        if answer is not None:
            # Training mode: format with answer
            formatted = self.format_chat_completion(question, answer, system_prompt)
            encodings = self.tokenizer(
                formatted,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )

            # For causal LM, labels are the same as input_ids
            # We'll mask the prompt part in the trainer
            encodings["labels"] = encodings["input_ids"].copy()

        else:
            # Inference mode: format without answer
            formatted = self.format_chat(question, system_prompt)
            encodings = self.tokenizer(
                formatted,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )

        return encodings

    def extract_answer(self, generated_text: str) -> str:
        """
        Extract the answer from generated text.
        Removes chat template markers and extra whitespace.

        Args:
            generated_text: Generated text from the model

        Returns:
            Cleaned answer
        """
        # Remove common chat markers
        markers = [
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|im_start|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]

        answer = generated_text
        for marker in markers:
            answer = answer.replace(marker, "")

        # Remove "assistant:" prefix if present
        if answer.lower().startswith("assistant:"):
            answer = answer[10:].strip()

        return answer.strip()

    def batch_format_chat(
        self,
        questions: List[str],
        answers: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Batch format multiple questions.

        Args:
            questions: List of questions
            answers: Optional list of answers
            system_prompt: Optional system prompt

        Returns:
            List of formatted chat strings
        """
        if answers is not None:
            return [
                self.format_chat_completion(q, a, system_prompt)
                for q, a in zip(questions, answers)
            ]
        else:
            return [
                self.format_chat(q, system_prompt)
                for q in questions
            ]
