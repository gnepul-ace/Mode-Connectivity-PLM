"""
Decoder-only LLM Configuration and Loading
Supports Qwen, Llama with LoRA/Adapter tuning
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel


def load_decoder_llm_with_pet(args):
    """
    Load decoder-only LLM (full parameter or with PET)
    Supports: Full model, LoRA, Adapter

    Args:
        args: Arguments with model config

    Returns:
        model, config, tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else None
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base config
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # Determine dtype
    if hasattr(args, 'bf16') and args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,  # We'll move to device manually
    )

    # Get tune_method (default to "model" for full parameter)
    tune_method = getattr(args, 'tune_method', 'model')

    # Apply Parameter-Efficient Tuning (optional)
    if tune_method == "lora":
        # Apply LoRA
        lora_config = LoraConfig(
            r=args.lora_rank if hasattr(args, 'lora_rank') else 8,
            lora_alpha=args.lora_alpha if hasattr(args, 'lora_alpha') else 16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Standard attention modules
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif args.tune_method == "adapter":
        # Apply Adapter (using existing adapter modules from module/)
        from module.adapter import Adapter

        # Add adapters to each layer
        for layer_idx, layer in enumerate(model.model.layers):
            hidden_size = config.hidden_size

            # Add adapter after self-attention
            layer.attn_adapter = Adapter(
                dim=hidden_size,
                r=args.adapter_size,
                act='gelu'
            )

            # Add adapter after MLP
            layer.mlp_adapter = Adapter(
                dim=hidden_size,
                r=args.adapter_size,
                act='gelu'
            )

        # Freeze base model parameters
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        print(f"Added adapters with size {args.adapter_size}")
        print_trainable_parameters(model)

    elif args.tune_method == "model":
        # Full fine-tuning
        print("Full model fine-tuning enabled")

    # Freeze embeddings if specified
    if args.freeze_embeds:
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens.weight.requires_grad = False
        print("Froze embedding layer")

    return model, config, tokenizer


def print_trainable_parameters(model):
    """Print trainable parameters"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )
