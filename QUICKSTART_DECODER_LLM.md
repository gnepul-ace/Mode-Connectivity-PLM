# Quick Start Guide: Mode Connectivity for Decoder-Only LLMs

This is a **5-minute quick start** to get you running mode connectivity analysis on reasoning tasks.

## Setup (One-time)

```bash
# 1. Install dependencies
pip install -r requirements_decoder_llm.txt

# 2. Verify installation
python -c "import torch; from transformers import AutoTokenizer; print('✓ Setup complete!')"
```

## Run Complete Workflow (Automated)

The fastest way to see everything in action:

```bash
# Run the complete workflow: train 2 models, evaluate, analyze mode connectivity
bash scripts/decoder_llm/quick_start.sh
```

This script will:
1. Train Model 1 (seed 42) on GSM8K with LoRA
2. Train Model 2 (seed 123) on GSM8K with LoRA
3. Evaluate both models
4. Perform mode connectivity analysis
5. Generate visualization plots

Results will be in `./outputs/quick_start/`

## Manual Step-by-Step

If you prefer to run each step manually:

### Step 1: Train First Model (5-10 minutes on GPU)

```bash
python scripts/decoder_llm/train_reasoning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --output_dir ./outputs/model1 \
  --max_steps 2000 \
  --seed 42 \
  --bf16
```

### Step 2: Train Second Model (5-10 minutes on GPU)

```bash
python scripts/decoder_llm/train_reasoning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --output_dir ./outputs/model2 \
  --max_steps 2000 \
  --seed 123 \
  --bf16
```

### Step 3: Mode Connectivity Analysis

```bash
python scripts/decoder_llm/mode_connectivity.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --checkpoint_1 ./outputs/model1/best_model \
  --checkpoint_2 ./outputs/model2/best_model \
  --dataset gsm8k \
  --output_dir ./outputs/connectivity \
  --plot
```

Check the results:
- Plot: `./outputs/connectivity/mode_connectivity_plot.png`
- Data: `./outputs/connectivity/mode_connectivity_results.json`

## Understanding the Results

### Mode Connectivity Plot

The plot shows accuracy across the interpolation path:

```
Accuracy
   ^
   |     *-----------*  ← Flat = good connectivity
   |    /             \
   |   *               * ← Endpoints (your models)
   |
   +-------------------→
   0.0      α       1.0
```

**Interpretation:**
- **Flat line**: Models are connected (same mode)
- **V-shape**: Barrier between models (different modes)
- **Height**: Overall performance level

### Typical Results

For GSM8K with Qwen-0.5B + LoRA:
- Endpoint 1: ~35-40% accuracy
- Endpoint 2: ~35-40% accuracy
- Interpolation: ~30-40% (should be relatively flat)

## Different Model Sizes

**Small models** (fast, good for experimentation):
```bash
--model_name Qwen/Qwen2.5-0.5B-Instruct  # ~5 min per 2000 steps
```

**Medium models** (better performance):
```bash
--model_name Qwen/Qwen2.5-1.5B-Instruct  # ~15 min per 2000 steps
```

**Large models** (best performance, slower):
```bash
--model_name Qwen/Qwen2.5-7B-Instruct    # ~1 hour per 2000 steps
```

## Different Datasets

**GSM8K** (easier, grade school math):
```bash
--dataset gsm8k
```

**MATH** (harder, competition math):
```bash
--dataset math
```

## Evaluation Only

To evaluate an existing checkpoint:

```bash
python scripts/decoder_llm/evaluate_reasoning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --checkpoint_path ./outputs/model1/best_model \
  --dataset gsm8k
```

## Common Issues

### Out of Memory

Reduce batch size:
```bash
--batch_size 1 --gradient_accumulation_steps 16
```

### Slow Training

Reduce steps for quick testing:
```bash
--max_steps 1000
```

### Model Not Found

Make sure you have HuggingFace access:
```bash
huggingface-cli login
```

For Llama models, you need to accept the license on HuggingFace.

## Next Steps

1. **Read full documentation**: `DECODER_LLM_README.md`
2. **Try different models**: Llama, Qwen of different sizes
3. **Experiment with hyperparameters**: See `configs/decoder_llm_examples.yaml`
4. **Integrate RL training**: Use checkpoints from DAPO/DR GRPO

## File Structure

```
Mode-Connectivity-PLM/
├── DecoderLLM_model/           # Core implementation
├── scripts/decoder_llm/        # Training scripts
│   ├── train_reasoning.py
│   ├── evaluate_reasoning.py
│   ├── mode_connectivity.py
│   └── quick_start.sh
├── configs/
│   └── decoder_llm_examples.yaml
├── requirements_decoder_llm.txt
├── DECODER_LLM_README.md       # Full documentation
└── QUICKSTART_DECODER_LLM.md   # This file
```

## Support

- Full docs: `DECODER_LLM_README.md`
- Example configs: `configs/decoder_llm_examples.yaml`
- Issues: Open a GitHub issue

---

**Ready to start? Run:**
```bash
bash scripts/decoder_llm/quick_start.sh
```
