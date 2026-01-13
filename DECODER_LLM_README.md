# Mode Connectivity for Decoder-Only LLMs on Reasoning Tasks

This extension adds support for investigating mode connectivity of decoder-only LLMs (Qwen, Llama) after RL training on reasoning benchmarks (GSM8K, MATH), following DAPO/DR GRPO evaluation protocols.

## Key Features

- ✅ **Decoder-Only LLM Support**: Qwen, Llama, and other HuggingFace causal LMs
- ✅ **Reasoning Benchmarks**: GSM8K (grade school math), MATH (competition math)
- ✅ **Chat Template Handling**: Correct templates for different models (critical for performance!)
- ✅ **Parameter-Efficient Tuning**: LoRA and Adapter support
- ✅ **Mode Connectivity Analysis**: Linear interpolation between checkpoints
- ✅ **Based on Existing Architecture**: Extends the proven T5/RoBERTa infrastructure

## Installation

```bash
pip install torch transformers datasets peft pandas numpy matplotlib
```

## Quick Start

### 1. Train Two Models (Different Seeds)

Train first model:
```bash
python train_decoder_llm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir ./outputs/qwen_gsm8k_seed42 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --train_iters 10000 \
  --valid_interval 1000 \
  --seed 42 \
  --bf16 \
  --do_train
```

Train second model (different seed):
```bash
python train_decoder_llm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir ./outputs/qwen_gsm8k_seed123 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --train_iters 10000 \
  --valid_interval 1000 \
  --seed 123 \
  --bf16 \
  --do_train
```

### 2. Mode Connectivity Analysis

```bash
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --load_PET_path_1 ./outputs/qwen_gsm8k_seed42/checkpoint-best.pt \
  --load_PET_path_2 ./outputs/qwen_gsm8k_seed123/checkpoint-best.pt \
  --itpl_points 11 \
  --output_dir ./outputs/mode_connectivity \
  --max_input_length 512 \
  --max_output_length 512 \
  --eval_batch_size 8 \
  --seed 42
```

This will:
- Interpolate between the two models at 11 points (α = 0.0, 0.1, ..., 1.0)
- Evaluate accuracy at each point
- Save results to CSV file

## Architecture

### Code Structure

```
Mode-Connectivity-PLM/
├── DecoderLLM_model/              # NEW: Decoder LLM support
│   ├── __init__.py
│   ├── modeling_decoder.py        # Model loading with LoRA/Adapter
│   ├── chat_template.py           # Chat template handling (CRITICAL!)
│   └── decoder_trainer.py         # Trainer following T5 trainer structure
│
├── dataloader/reasoning/          # NEW: Reasoning datasets
│   ├── __init__.py
│   ├── reasoning_loader.py        # GSM8K, MATH loaders
│   └── reasoning_metrics.py       # Evaluation metrics
│
├── train_decoder_llm.py           # NEW: Training script
├── decoder_interpolation.py       # NEW: Mode connectivity analysis
│
├── T5_model/                      # EXISTING: T5 support
├── RoBERTa_model/                 # EXISTING: RoBERTa support
├── module/                        # EXISTING: Adapter modules
└── utils/options.py               # UPDATED: Added decoder LLM args
```

### Design Philosophy

**Minimal Changes, Maximum Compatibility**: This implementation extends the existing codebase by:
- Following the same `Trainer` structure as `T5_trainer.py`
- Using the same `option` argument system
- Implementing `itp_valid()` and `itp_test()` methods for mode connectivity
- Reusing `task_interpolation.py` logic but adapted for decoder models

## Chat Templates (IMPORTANT!)

**Different models require different chat templates!** Using the wrong template can significantly hurt performance.

### Qwen Format
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

### Llama Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

The `ChatTemplateHandler` automatically handles these differences and adds Chain-of-Thought prompts ("Let's think step by step").

## Datasets

### GSM8K (Grade School Math)
- **Size**: 7,473 train + 1,319 test
- **Format**: Word problems → numerical answers
- **Answer format**: `#### number`
- **Example**:
  ```
  Q: James writes 3 pages to 2 friends twice a week. How many pages per year?
  A: He writes 3*2=6 pages per week to each friend.
     So 6*2=12 pages total per week.
     That's 12*52=624 pages per year.
     #### 624
  ```

### MATH (Competition Mathematics)
- **Size**: 7,500 train + 5,000 test
- **Format**: Competition problems → LaTeX answers
- **Answer format**: `\boxed{answer}`
- **Difficulty**: AMC, AIME level

## Supported Models

### Qwen Models
```bash
--model Qwen/Qwen2.5-0.5B-Instruct  # Small, fast
--model Qwen/Qwen2.5-1.5B-Instruct  # Medium
--model Qwen/Qwen2.5-7B-Instruct    # Large
```

### Llama Models
```bash
--model meta-llama/Llama-3.2-1B-Instruct
--model meta-llama/Llama-3.2-3B-Instruct
--model meta-llama/Llama-3.1-8B-Instruct
```

Note: Llama models may require HuggingFace authentication:
```bash
huggingface-cli login
```

## Parameter-Efficient Tuning

### LoRA (Recommended)
```bash
--tune_method lora \
--lora_rank 8 \      # Rank (4, 8, 16, 32)
--lora_alpha 16      # Alpha (typically 2*rank)
```

**When to use**: Most cases - fewer parameters, faster training

### Adapter
```bash
--tune_method adapter \
--adapter_size 64    # Size (32, 64, 128)
```

**When to use**: Alternative to LoRA, uses existing adapter modules

### Full Fine-tuning
```bash
--tune_method model
```

**When to use**: Maximum performance, but expensive

## Key Arguments

### Training Arguments
```bash
--train_batch_size 4              # Batch size per GPU
--gradient_accumulation_steps 4   # Effective batch = 4*4 = 16
--learning_rate 5e-5              # Learning rate (1e-5 to 1e-4 for LoRA)
--train_iters 10000               # Total training steps
--valid_interval 1000             # Validate every N steps
--max_grad_norm 1.0               # Gradient clipping
--weight_decay 0.01               # Weight decay
--warmup_steps 500                # Warmup steps
```

### Model Arguments
```bash
--model <model_name>              # HuggingFace model ID
--dataset gsm8k                   # gsm8k or math
--tune_method lora                # lora, adapter, or model
--bf16                            # Use BF16 mixed precision
--freeze_embeds                   # Freeze embedding layer
```

### Mode Connectivity Arguments
```bash
--load_PET_path_1 <path>          # First checkpoint
--load_PET_path_2 <path>          # Second checkpoint
--itpl_points 11                  # Number of interpolation points
```

## Expected Performance

Based on DAPO/DR GRPO papers:

| Model | GSM8K Accuracy | MATH Accuracy |
|-------|----------------|---------------|
| Qwen2.5-0.5B + LoRA | 30-45% | 15-25% |
| Qwen2.5-1.5B + LoRA | 50-65% | 25-35% |
| Qwen2.5-7B + LoRA | 75-85% | 45-60% |

*Note: Actual results depend on training hyperparameters and data quality.*

## Mode Connectivity Results

The interpolation script generates a CSV file with columns:
- `prefix`: Dataset name
- `metric`: Evaluation metric (accuracy)
- `x`: Interpolation coefficient (0 to 1)
- `dev_performance`: Validation accuracy
- `test_performance`: Test accuracy

**Interpreting Results:**
- **Flat curve**: Strong mode connectivity (models are in same mode)
- **V-shaped curve**: Barrier between modes (different modes)
- **Higher than endpoints**: Beneficial interpolation (rare)

## Integration with RL Training

This implementation works with models trained using:
- **DAPO** (Decoupled Advantage Policy Optimization)
- **DR GRPO** (without std normalization)
- **GRPO** (Group Relative Policy Optimization)

**Steps:**
1. Train model with RL framework (e.g., using TRL library)
2. Save checkpoint in compatible format
3. Run mode connectivity analysis

## Troubleshooting

### Out of Memory
```bash
--train_batch_size 1 \
--gradient_accumulation_steps 16 \
--max_input_length 512  # Reduce if needed
```

### Slow Training
```bash
--bf16  # Use mixed precision
--train_iters 5000  # Reduce for testing
```

### Model Not Found
```bash
# For gated models (Llama):
huggingface-cli login

# Or set cache directory:
--cache_dir /path/to/cache
```

### Wrong Chat Template
The `ChatTemplateHandler` automatically detects model type from the name. If your model uses a custom template, you may need to modify `DecoderLLM_model/chat_template.py`.

## Example: Complete Workflow

```bash
# 1. Train model 1 (seed 42)
python train_decoder_llm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --output_dir ./outputs/model_seed42 \
  --train_iters 5000 \
  --seed 42 \
  --bf16 \
  --do_train

# 2. Train model 2 (seed 123)
python train_decoder_llm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --output_dir ./outputs/model_seed123 \
  --train_iters 5000 \
  --seed 123 \
  --bf16 \
  --do_train

# 3. Mode connectivity analysis
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --load_PET_path_1 ./outputs/model_seed42/checkpoint-best.pt \
  --load_PET_path_2 ./outputs/model_seed123/checkpoint-best.pt \
  --itpl_points 11 \
  --output_dir ./outputs/connectivity \
  --seed 42

# 4. Check results
cat ./outputs/connectivity/interpolation_results_gsm8k.csv
```

## References

- **Mode Connectivity**: "Exploring Mode Connectivity for Pre-trained Language Models" (EMNLP 2022)
- **DAPO**: Decoupled Advantage Policy Optimization
- **DR GRPO**: Dr. GRPO (removing std normalization)
- **GSM8K**: https://github.com/openai/grade-school-math
- **MATH**: https://github.com/hendrycks/math

## Citation

If you use this code, please cite the original mode connectivity paper:

```bibtex
@inproceedings{he-etal-2022-exploring,
    title = "Exploring Mode Connectivity for Pre-trained Language Models",
    booktitle = "Proceedings of EMNLP",
    year = "2022",
}
```

## License

Follows the same license as the original Mode-Connectivity-PLM repository.
