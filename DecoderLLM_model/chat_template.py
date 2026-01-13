"""
Chat Template Handler for Different Decoder-Only LLMs
Critical for correct reasoning performance - different models need different templates!
"""

from typing import Optional


class ChatTemplateHandler:
    """
    Handles chat templates for different LLMs (Qwen, Llama, etc.)
    IMPORTANT: Using the wrong template can significantly hurt performance!
    """

    # System prompts for reasoning
    REASONING_SYSTEM_PROMPTS = {
        "qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "llama": "You are a helpful, respectful and honest assistant.",
        "default": "You are a helpful assistant.",
    }

    def __init__(self, model_name_or_path: str, tokenizer):
        """
        Args:
            model_name_or_path: HuggingFace model identifier
            tokenizer: HuggingFace tokenizer
        """
        self.model_name = model_name_or_path.lower()
        self.tokenizer = tokenizer
        self.model_type = self._detect_model_type()

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _detect_model_type(self):
        """Detect model type from name"""
        if "qwen" in self.model_name:
            return "qwen"
        elif "llama" in self.model_name:
            return "llama"
        else:
            return "default"

    def get_system_prompt(self, custom_prompt: Optional[str] = None):
        """Get system prompt"""
        if custom_prompt:
            return custom_prompt
        return self.REASONING_SYSTEM_PROMPTS.get(self.model_type, self.REASONING_SYSTEM_PROMPTS["default"])

    def format_chat(self, question: str, system_prompt: Optional[str] = None, add_generation_prompt: bool = True):
        """
        Format a question into chat format
        Uses model's native chat template if available
        """
        system = self.get_system_prompt(system_prompt)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]

        # Try to use native chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
                return formatted
            except:
                pass

        # Fallback to manual formatting
        return self._manual_format(messages, add_generation_prompt)

    def format_chat_with_cot(self, question: str, system_prompt: Optional[str] = None):
        """
        Format question with Chain-of-Thought instruction
        This is CRITICAL for reasoning tasks!
        """
        cot_instruction = "Let's think step by step."
        question_with_cot = f"{question}\n\n{cot_instruction}"
        return self.format_chat(question_with_cot, system_prompt)

    def format_chat_completion(self, question: str, answer: str, system_prompt: Optional[str] = None):
        """
        Format a complete Q&A pair (for training)
        """
        system = self.get_system_prompt(system_prompt)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return formatted
            except:
                pass

        return self._manual_format(messages, add_generation_prompt=False)

    def _manual_format(self, messages, add_generation_prompt):
        """Manual formatting for models without native chat template"""
        if self.model_type == "qwen":
            return self._format_qwen(messages, add_generation_prompt)
        elif self.model_type == "llama":
            return self._format_llama(messages, add_generation_prompt)
        else:
            return self._format_default(messages, add_generation_prompt)

    def _format_qwen(self, messages, add_generation_prompt):
        """
        Qwen chat format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user}<|im_end|>
        <|im_start|>assistant
        {assistant}<|im_end|>
        """
        formatted = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        return formatted

    def _format_llama(self, messages, add_generation_prompt):
        """
        Llama 3 chat format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {assistant}<|eot_id|>
        """
        formatted = ""

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            if i == 0:
                formatted += "<|begin_of_text|>"

            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

        if add_generation_prompt:
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted

    def _format_default(self, messages, add_generation_prompt):
        """Default simple format"""
        formatted = ""

        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"{role}: {content}\n\n"

        if add_generation_prompt:
            formatted += "Assistant: "

        return formatted

    def extract_answer(self, generated_text):
        """
        Extract answer from generated text
        Removes chat template markers
        """
        # Remove chat markers
        markers = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>",
                  "<|im_start|>", "<|start_header_id|>", "<|end_header_id|>"]

        answer = generated_text
        for marker in markers:
            answer = answer.replace(marker, "")

        # Remove role prefix if present
        if answer.lower().startswith("assistant:"):
            answer = answer[10:].strip()

        return answer.strip()
