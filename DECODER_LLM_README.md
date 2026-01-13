# Mode Connectivity Analysis for Decoder-Only LLMs (After RL Training)

This extension adds support for investigating mode connectivity of decoder-only Large Language Models (like Qwen, Llama) after reinforcement learning (RL) training on reasoning tasks.

## Overview

This implementation enables you to:

1. **Train/Fine-tune** decoder-only LLMs (Qwen, Llama) on reasoning benchmarks using Parameter-Efficient Tuning (LoRA, Adapters)
2. **Evaluate** models on reasoning tasks (GSM8K, MATH) with proper chat templates
3. **Analyze Mode Connectivity** between two trained models by linear interpolation
4. **Test on Reasoning Benchmarks** similar to DAPO and DR GRPO papers

## Key Features

- **Supported Models**: Qwen, Llama, and other HuggingFace causal LMs
- **Reasoning Benchmarks**: GSM8K (grade school math), MATH (competition math)
- **Proper Chat Templates**: Handles model-specific chat formats correctly
- **Parameter-Efficient Tuning**: LoRA and Adapter support
- **Chain-of-Thought**: Automatic CoT prompting for better reasoning
- **Mode Connectivity**: Linear interpolation analysis between checkpoints

## Installation

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv_decoder
source venv_decoder/bin/activate  # On Windows: venv_decoder\Scripts\activate

# Install dependencies
pip install -r requirements_decoder_llm.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('Transformers installed successfully')"
```

## Quick Start

### 1. Train a Model on GSM8K

Train a Qwen model with LoRA on GSM8K:

```bash
python scripts/decoder_llm/train_reasoning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir ./outputs/qwen_gsm8k_seed42 \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 10000 \
  --eval_steps 500 \
  --save_steps 1000 \
  --seed 42 \
  --use_cot \
  --bf16
```

### 2. Train with Different Seed (for mode connectivity)

```bash
python scripts/decoder_llm/train_reasoning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir ./outputs/qwen_gsm8k_seed123 \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 10000 \
  --eval_steps 500 \
  --save_steps 1000 \
  --seed 123 \
  --use_cot \
  --bf16
```

### 3. Evaluate a Trained Model

```bash
python scripts/decoder_llm/evaluate_reasoning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --checkpoint_path ./outputs/qwen_gsm8k_seed42/best_model \
  --dataset gsm8k \
  --split test \
  --tune_method lora \
  --batch_size 8 \
  --max_new_tokens 512 \
  --temperature 0.0 \
  --use_cot \
  --output_file ./results/qwen_gsm8k_eval.json
```

### 4. Mode Connectivity Analysis

Analyze the loss landscape between two checkpoints:

```bash
python scripts/decoder_llm/mode_connectivity.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --checkpoint_1 ./outputs/qwen_gsm8k_seed42/best_model \
  --checkpoint_2 ./outputs/qwen_gsm8k_seed123/best_model \
  --dataset gsm8k \
  --split test \
  --tune_method lora \
  --num_points 11 \
  --batch_size 8 \
  --output_dir ./outputs/mode_connectivity/qwen_gsm8k \
  --plot \
  --use_cot
```

This will:
- Interpolate between the two models at 11 points (α = 0.0, 0.1, ..., 1.0)
- Evaluate accuracy at each point
- Generate visualization plots
- Save results to JSON

## Detailed Usage

### Training Options

#### Model Selection

**Qwen Models:**
```bash
--model_name Qwen/Qwen2.5-0.5B-Instruct  # Small, fast
--model_name Qwen/Qwen2.5-1.5B-Instruct  # Medium
--model_name Qwen/Qwen2.5-7B-Instruct    # Large
```

**Llama Models:**
```bash
--model_name meta-llama/Llama-3.2-1B-Instruct
--model_name meta-llama/Llama-3.2-3B-Instruct
--model_name meta-llama/Llama-3.1-8B-Instruct
```

#### Dataset Options

**GSM8K** (Grade School Math - 8,500 problems, 2-8 steps):
```bash
--dataset gsm8k
```

**MATH** (Competition Math - 12,500 problems):
```bash
--dataset math
```

#### Parameter-Efficient Tuning

**LoRA** (Recommended - fewer parameters, faster):
```bash
--tune_method lora \
--lora_rank 8 \        # Rank of LoRA matrices (4, 8, 16, 32)
--lora_alpha 16        # Scaling factor (typically 2*rank)
```

**Adapter** (Houlsby-style adapters):
```bash
--tune_method adapter \
--adapter_size 64      # Hidden dimension of adapter (32, 64, 128)
```

**Full Fine-tuning** (Most parameters, expensive):
```bash
--tune_method model
```

#### Chain-of-Thought Prompting

Enable CoT for better reasoning:
```bash
--use_cot  # Adds "Let's think step by step" to prompts
```

#### Training Hyperparameters

```bash
--learning_rate 5e-5              # Learning rate (1e-5 to 1e-4 for LoRA)
--batch_size 4                    # Batch size per device
--gradient_accumulation_steps 4   # Effective batch = 4 * 4 = 16
--max_steps 10000                 # Total training steps
--warmup_steps 500                # Warmup steps
--weight_decay 0.01               # Weight decay
--max_grad_norm 1.0               # Gradient clipping
```

#### Hardware Options

```bash
--device cuda          # Use GPU
--bf16                 # Use BF16 (recommended for Ampere+ GPUs)
--fp16                 # Use FP16 (for older GPUs)
```

### Evaluation Options

```bash
--max_new_tokens 512    # Max tokens to generate
--temperature 0.0       # Temperature (0.0 = greedy, >0 = sampling)
--do_sample             # Use sampling instead of greedy
--num_return_sequences 1  # Generate multiple solutions (for voting)
```

### Mode Connectivity Options

```bash
--num_points 11         # Number of interpolation points (including endpoints)
--plot                  # Generate visualization plots
```

## Reasoning Benchmarks

### GSM8K (Grade School Math)

- **Size**: 8,500 problems (7,473 train + 1,319 test)
- **Difficulty**: Grade school level (2-8 reasoning steps)
- **Format**: Word problems with numerical answers
- **Metric**: Numerical accuracy (exact match)

**Example:**
```
Question: James writes a 3-page letter to 2 different friends twice a week.
How many pages does he write a year?

Answer: He writes each friend 3*2=6 pages a week
So he writes 6*2=12 pages every week
That means he writes 12*52=624 pages a year
#### 624
```

### MATH (Competition Mathematics)

- **Size**: 12,500 problems (7,500 train + 5,000 test)
- **Difficulty**: AMC, AIME level (competition math)
- **Categories**: 7 subjects (Algebra, Geometry, Number Theory, etc.)
- **Format**: LaTeX math problems with boxed answers
- **Metric**: Exact match on extracted answer

**Example:**
```
Question: Find the value of x that satisfies the equation 2x + 5 = 13

Solution: We solve for x:
2x + 5 = 13
2x = 8
x = 4
\boxed{4}
```

## Expected Performance

Based on similar work (DAPO, DR GRPO), here are approximate accuracy ranges:

| Model | GSM8K (Base) | GSM8K (After SFT) | MATH (Base) | MATH (After SFT) |
|-------|--------------|-------------------|-------------|------------------|
| Qwen2.5-0.5B | ~10-20% | ~30-45% | ~5-10% | ~15-25% |
| Qwen2.5-1.5B | ~30-40% | ~50-65% | ~10-15% | ~25-35% |
| Qwen2.5-7B | ~60-70% | ~75-85% | ~25-35% | ~45-60% |
| Llama-3.2-1B | ~15-25% | ~35-50% | ~8-12% | ~18-28% |
| Llama-3.2-3B | ~40-50% | ~60-70% | ~15-20% | ~30-40% |

Note: Actual performance depends on training hyperparameters, data quality, and training steps.

## Mode Connectivity Experiments

### Typical Experimental Setups

#### 1. Different Random Seeds

Train two models with different random seeds to analyze mode connectivity:

```bash
# Model 1 (seed 42)
python scripts/decoder_llm/train_reasoning.py --seed 42 --output_dir ./outputs/model_seed42

# Model 2 (seed 123)
python scripts/decoder_llm/train_reasoning.py --seed 123 --output_dir ./outputs/model_seed123

# Analyze connectivity
python scripts/decoder_llm/mode_connectivity.py \
  --checkpoint_1 ./outputs/model_seed42/best_model \
  --checkpoint_2 ./outputs/model_seed123/best_model
```

#### 2. Different Training Steps

Compare models at different training stages:

```bash
# Analyze connectivity between early and late checkpoints
python scripts/decoder_llm/mode_connectivity.py \
  --checkpoint_1 ./outputs/model/checkpoint-5000 \
  --checkpoint_2 ./outputs/model/checkpoint-10000
```

#### 3. Different Hyperparameters

Train with different learning rates:

```bash
# Model 1 (lr=5e-5)
python scripts/decoder_llm/train_reasoning.py --learning_rate 5e-5 --output_dir ./outputs/model_lr5e5

# Model 2 (lr=1e-4)
python scripts/decoder_llm/train_reasoning.py --learning_rate 1e-4 --output_dir ./outputs/model_lr1e4

# Analyze connectivity
python scripts/decoder_llm/mode_connectivity.py \
  --checkpoint_1 ./outputs/model_lr5e5/best_model \
  --checkpoint_2 ./outputs/model_lr1e4/best_model
```

## Integration with RL Training

This codebase is designed to work with models trained using RL methods like DAPO, DR GRPO, or GRPO.

### After RL Training

If you have models trained with RL frameworks (e.g., using TRL library):

1. **Convert checkpoints**: Extract LoRA/Adapter weights if needed
2. **Place in compatible format**: Ensure weights are in `trainable_params.pt`
3. **Run mode connectivity**: Use the same analysis scripts

Example structure for RL checkpoints:
```
./outputs/rl_model_checkpoint/
├── trainable_params.pt  # LoRA or adapter weights
├── config.json          # Training configuration
└── training_state.pt    # Optional: optimizer state
```

## Visualizing Results

The mode connectivity script generates plots automatically. The plot shows:

- **X-axis**: Interpolation coefficient α (0 to 1)
- **Y-axis**: Accuracy on test set
- **Red stars**: Endpoints (original models)
- **Blue line**: Interpolation path

**Interpretation:**
- **Flat line**: Strong mode connectivity (models are in same mode)
- **V-shape**: Barrier between modes (different modes)
- **Higher than endpoints**: Beneficial interpolation (rare but possible)

## Troubleshooting

### Out of Memory (OOM)

Reduce memory usage:
```bash
--batch_size 1 \
--gradient_accumulation_steps 16 \
--max_length 1024  # Reduce sequence length
```

For very large models, use model parallelism:
```bash
# In model config, set device_map_auto=True
# This will automatically distribute model across GPUs
```

### Slow Training

Speed up training:
```bash
--bf16  # Use BF16 mixed precision
--gradient_accumulation_steps 8  # Larger effective batch size
--max_steps 5000  # Reduce steps for experimentation
```

### Low Accuracy

Improve model performance:
```bash
--use_cot  # Enable Chain-of-Thought
--learning_rate 1e-4  # Try different learning rates
--lora_rank 16  # Increase LoRA capacity
--max_steps 20000  # Train longer
```

### Chat Template Issues

If generation output looks wrong, verify chat template:
```python
from transformers import AutoTokenizer
from DecoderLLM_model import ChatTemplateHandler

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
handler = ChatTemplateHandler("Qwen/Qwen2.5-0.5B-Instruct", tokenizer)

# Test formatting
formatted = handler.format_chat("What is 2+2?")
print(formatted)
```

## References

This implementation is inspired by and follows evaluation protocols from:

1. **DAPO**: Decoupled Advantage Policy Optimization for reasoning tasks
2. **DR GRPO**: Dr. GRPO removes standard deviation normalization in advantage computation
3. **GRPO**: Group Relative Policy Optimization for LLM reasoning

**Reasoning Benchmarks:**
- GSM8K: [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
- MATH: [https://github.com/hendrycks/math](https://github.com/hendrycks/math)

**Mode Connectivity:**
- Original paper: "Exploring Mode Connectivity for Pre-trained Language Models" (EMNLP 2022)

## Citation

If you use this code, please cite the original mode connectivity paper:

```bibtex
@inproceedings{he-etal-2022-exploring,
    title = "Exploring Mode Connectivity for Pre-trained Language Models",
    author = "He, Yifei and ...",
    booktitle = "Proceedings of EMNLP",
    year = "2022",
}
```

## Project Structure

```
Mode-Connectivity-PLM/
├── DecoderLLM_model/               # New: Decoder-only LLM support
│   ├── __init__.py
│   ├── modeling_decoder_llm.py     # Model with PET support
│   ├── chat_template.py            # Chat template handling
│   ├── reasoning_dataloader.py     # GSM8K, MATH loaders
│   ├── reasoning_metrics.py        # Evaluation metrics
│   └── decoder_trainer.py          # Trainer with mode connectivity
├── scripts/decoder_llm/            # New: Training scripts
│   ├── train_reasoning.py          # Training script
│   ├── evaluate_reasoning.py       # Evaluation script
│   └── mode_connectivity.py        # Mode connectivity analysis
├── requirements_decoder_llm.txt    # Dependencies
├── DECODER_LLM_README.md          # This file
├── T5_model/                       # Original: T5 support
├── RoBERTa_model/                  # Original: RoBERTa support
└── ...
```

## License

This project follows the same license as the original Mode-Connectivity-PLM repository.

## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to original repository for base functionality

---

**Happy researching! 🚀**
