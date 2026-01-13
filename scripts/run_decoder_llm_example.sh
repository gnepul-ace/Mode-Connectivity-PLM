#!/bin/bash
# Example workflow for decoder LLM mode connectivity analysis

set -e  # Exit on error

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="gsm8k"
TUNE_METHOD="lora"
OUTPUT_BASE="./outputs/example"

echo "=========================================="
echo "Decoder LLM Mode Connectivity Example"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Tuning: $TUNE_METHOD"
echo "=========================================="

# Step 1: Train first model (seed 42)
echo ""
echo "Step 1: Training Model 1 (seed=42)..."
python train_decoder_llm.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir "${OUTPUT_BASE}/model_seed42" \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --train_iters 2000 \
  --valid_interval 500 \
  --max_input_length 512 \
  --max_output_length 512 \
  --seed 42 \
  --bf16 \
  --do_train

echo "✓ Model 1 training complete!"

# Step 2: Train second model (seed 123)
echo ""
echo "Step 2: Training Model 2 (seed=123)..."
python train_decoder_llm.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir "${OUTPUT_BASE}/model_seed123" \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --train_iters 2000 \
  --valid_interval 500 \
  --max_input_length 512 \
  --max_output_length 512 \
  --seed 123 \
  --bf16 \
  --do_train

echo "✓ Model 2 training complete!"

# Step 3: Mode connectivity analysis
echo ""
echo "Step 3: Mode Connectivity Analysis..."
python decoder_interpolation.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --load_PET_path_1 "${OUTPUT_BASE}/model_seed42/checkpoint-best.pt" \
  --load_PET_path_2 "${OUTPUT_BASE}/model_seed123/checkpoint-best.pt" \
  --itpl_points 11 \
  --output_dir "${OUTPUT_BASE}/connectivity" \
  --max_input_length 512 \
  --max_output_length 512 \
  --eval_batch_size 8 \
  --seed 42

echo ""
echo "=========================================="
echo "✓ Analysis Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - Model 1: ${OUTPUT_BASE}/model_seed42/"
echo "  - Model 2: ${OUTPUT_BASE}/model_seed123/"
echo "  - Connectivity: ${OUTPUT_BASE}/connectivity/"
echo ""
echo "View results:"
echo "  cat ${OUTPUT_BASE}/connectivity/interpolation_results_${DATASET}.csv"
echo "=========================================="
