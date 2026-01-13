"""
Decoder-only LLM models with Parameter-Efficient Tuning support
Supports: Qwen, Llama, and other causal LMs for reasoning tasks
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)
from typing import Optional, Dict, List
import os
import sys

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.adapter import Adapter
from module.lora import LoRALayer


class DecoderLLMWithPET(nn.Module):
    """
    Decoder-only LLM with Parameter-Efficient Tuning (PET) support.
    Supports:
    - Adapter tuning
    - LoRA (Low-Rank Adaptation)
    - Full fine-tuning
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load base model
        self.model = self._load_base_model()

        # Apply parameter-efficient tuning if specified
        if config.apply_adapter:
            self._add_adapters()
        elif config.apply_lora:
            self._add_lora()

        # Freeze parameters based on tuning method
        self._freeze_parameters()

    def _load_base_model(self):
        """Load the base causal LM model"""
        model_name = self.config.model_name_or_path

        # Load with specific model class if specified
        if "qwen" in model_name.lower():
            model = Qwen2ForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.config.device_map_auto else None,
            )
        elif "llama" in model_name.lower():
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
                device_map="auto" if self.config.device_map_auto else None,
            )
        else:
            # Generic causal LM loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.config.device_map_auto else None,
            )

        return model

    def _add_adapters(self):
        """Add adapter modules to the model"""
        adapter_config = {
            'adapter_size': self.config.adapter_size,
            'adapter_type': self.config.adapter_type,
        }

        # Add adapters to each transformer layer
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Add adapter after self-attention
            if hasattr(layer, 'self_attn'):
                hidden_size = layer.self_attn.hidden_size if hasattr(layer.self_attn, 'hidden_size') else self.config.hidden_size
                layer.attn_adapter = Adapter(
                    dim=hidden_size,
                    r=self.config.adapter_size,
                    act=self.config.adapter_activation,
                )

            # Add adapter after MLP
            if hasattr(layer, 'mlp'):
                hidden_size = self.config.hidden_size
                layer.mlp_adapter = Adapter(
                    dim=hidden_size,
                    r=self.config.adapter_size,
                    act=self.config.adapter_activation,
                )

        print(f"Added {self.config.adapter_type} adapters with size {self.config.adapter_size}")

    def _add_lora(self):
        """Add LoRA modules to the model"""
        # Target modules for LoRA: typically attention q, k, v projections
        target_modules = self.config.lora_target_modules

        for layer in self.model.model.layers:
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn

                # Apply LoRA to specified modules
                if 'q_proj' in target_modules and hasattr(attn, 'q_proj'):
                    attn.q_proj = self._convert_to_lora(attn.q_proj)
                if 'k_proj' in target_modules and hasattr(attn, 'k_proj'):
                    attn.k_proj = self._convert_to_lora(attn.k_proj)
                if 'v_proj' in target_modules and hasattr(attn, 'v_proj'):
                    attn.v_proj = self._convert_to_lora(attn.v_proj)
                if 'o_proj' in target_modules and hasattr(attn, 'o_proj'):
                    attn.o_proj = self._convert_to_lora(attn.o_proj)

            # Optionally add LoRA to MLP
            if 'mlp' in target_modules and hasattr(layer, 'mlp'):
                if hasattr(layer.mlp, 'gate_proj'):
                    layer.mlp.gate_proj = self._convert_to_lora(layer.mlp.gate_proj)
                if hasattr(layer.mlp, 'up_proj'):
                    layer.mlp.up_proj = self._convert_to_lora(layer.mlp.up_proj)
                if hasattr(layer.mlp, 'down_proj'):
                    layer.mlp.down_proj = self._convert_to_lora(layer.mlp.down_proj)

        print(f"Added LoRA to modules: {target_modules} with rank {self.config.lora_rank}")

    def _convert_to_lora(self, layer):
        """Convert a linear layer to LoRA layer"""
        if not isinstance(layer, nn.Linear):
            return layer

        in_features = layer.in_features
        out_features = layer.out_features

        # Create LoRA matrices
        lora_A = nn.Parameter(torch.zeros(self.config.lora_rank, in_features))
        lora_B = nn.Parameter(torch.zeros(out_features, self.config.lora_rank))

        # Initialize
        nn.init.kaiming_uniform_(lora_A, a=5**0.5)
        nn.init.zeros_(lora_B)

        # Add LoRA parameters to the layer
        layer.lora_A = lora_A
        layer.lora_B = lora_B
        layer.lora_scaling = self.config.lora_alpha / self.config.lora_rank

        # Override forward method
        original_forward = layer.forward

        def lora_forward(x):
            result = original_forward(x)
            if hasattr(layer, 'lora_A') and hasattr(layer, 'lora_B'):
                lora_result = (x @ layer.lora_A.T @ layer.lora_B.T) * layer.lora_scaling
                result = result + lora_result
            return result

        layer.forward = lora_forward

        return layer

    def _freeze_parameters(self):
        """Freeze parameters based on tuning method"""
        if self.config.tune_method == 'adapter':
            # Freeze all base model parameters
            for name, param in self.model.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad = False
            print("Froze base model parameters, only training adapters")

        elif self.config.tune_method == 'lora':
            # Freeze all parameters except LoRA
            for name, param in self.model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
            print("Froze base model parameters, only training LoRA")

        elif self.config.tune_method == 'model':
            # Full fine-tuning
            print("Full model fine-tuning enabled")

        # Optionally freeze embeddings
        if self.config.freeze_embeds:
            if hasattr(self.model.model, 'embed_tokens'):
                self.model.model.embed_tokens.weight.requires_grad = False
            print("Froze embedding layer")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with adapter integration"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def generate(self, input_ids, attention_mask=None, **generation_kwargs):
        """Generate text with the model"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only trainable parameters (for saving checkpoints)"""
        trainable_params = {}

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param

        return trainable_params

    def load_trainable_parameters(self, state_dict: Dict[str, torch.Tensor]):
        """Load trainable parameters from checkpoint"""
        missing_keys = []
        unexpected_keys = []

        model_state = self.state_dict()

        for key, value in state_dict.items():
            if key in model_state:
                model_state[key] = value
            else:
                unexpected_keys.append(key)

        self.load_state_dict(model_state, strict=False)

        return missing_keys, unexpected_keys

    def print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}%"
        )


class DecoderLLMConfig:
    """Configuration for decoder-only LLMs with PET"""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        # PET settings
        apply_adapter: bool = False,
        apply_lora: bool = True,
        tune_method: str = "lora",  # "adapter", "lora", "model"
        # Adapter settings
        adapter_size: int = 64,
        adapter_type: str = "houlsby",
        adapter_activation: str = "gelu",
        # LoRA settings
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # Model settings
        hidden_size: int = 896,  # Qwen2.5-0.5B default
        freeze_embeds: bool = True,
        use_bf16: bool = True,
        device_map_auto: bool = False,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.apply_adapter = apply_adapter
        self.apply_lora = apply_lora
        self.tune_method = tune_method
        self.adapter_size = adapter_size
        self.adapter_type = adapter_type
        self.adapter_activation = adapter_activation
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        self.hidden_size = hidden_size
        self.freeze_embeds = freeze_embeds
        self.use_bf16 = use_bf16
        self.device_map_auto = device_map_auto

        # Add any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
