#!/bin/bash
# Quick start script for mode connectivity analysis of decoder-only LLMs
# This script demonstrates a complete workflow: train two models, evaluate, and analyze mode connectivity

set -e  # Exit on error

echo "=========================================="
echo "Mode Connectivity Analysis - Quick Start"
echo "=========================================="

# Configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"  # Change to your preferred model
DATASET="gsm8k"  # gsm8k or math
TUNE_METHOD="lora"
OUTPUT_BASE="./outputs/quick_start"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET"
echo "  Tuning: $TUNE_METHOD"
echo "  Output: $OUTPUT_BASE"
echo ""

# Step 1: Train first model (seed 42)
echo "=========================================="
echo "Step 1: Training Model 1 (seed=42)"
echo "=========================================="
python scripts/decoder_llm/train_reasoning.py \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --output_dir "${OUTPUT_BASE}/model_seed42" \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 5000 \
  --eval_steps 500 \
  --save_steps 1000 \
  --seed 42 \
  --use_cot \
  --bf16

echo ""
echo "✓ Model 1 training complete!"
echo ""

# Step 2: Train second model (seed 123)
echo "=========================================="
echo "Step 2: Training Model 2 (seed=123)"
echo "=========================================="
python scripts/decoder_llm/train_reasoning.py \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --output_dir "${OUTPUT_BASE}/model_seed123" \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 5000 \
  --eval_steps 500 \
  --save_steps 1000 \
  --seed 123 \
  --use_cot \
  --bf16

echo ""
echo "✓ Model 2 training complete!"
echo ""

# Step 3: Evaluate both models
echo "=========================================="
echo "Step 3: Evaluating Both Models"
echo "=========================================="

echo "Evaluating Model 1..."
python scripts/decoder_llm/evaluate_reasoning.py \
  --model_name "$MODEL_NAME" \
  --checkpoint_path "${OUTPUT_BASE}/model_seed42/best_model" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --use_cot \
  --output_file "${OUTPUT_BASE}/model_seed42_eval.json"

echo ""
echo "Evaluating Model 2..."
python scripts/decoder_llm/evaluate_reasoning.py \
  --model_name "$MODEL_NAME" \
  --checkpoint_path "${OUTPUT_BASE}/model_seed123/best_model" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --use_cot \
  --output_file "${OUTPUT_BASE}/model_seed123_eval.json"

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 4: Mode connectivity analysis
echo "=========================================="
echo "Step 4: Mode Connectivity Analysis"
echo "=========================================="
python scripts/decoder_llm/mode_connectivity.py \
  --model_name "$MODEL_NAME" \
  --checkpoint_1 "${OUTPUT_BASE}/model_seed42/best_model" \
  --checkpoint_2 "${OUTPUT_BASE}/model_seed123/best_model" \
  --dataset "$DATASET" \
  --tune_method "$TUNE_METHOD" \
  --num_points 11 \
  --output_dir "${OUTPUT_BASE}/mode_connectivity" \
  --use_cot \
  --plot

echo ""
echo "=========================================="
echo "✓ All steps complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Model 1: ${OUTPUT_BASE}/model_seed42/"
echo "  - Model 2: ${OUTPUT_BASE}/model_seed123/"
echo "  - Mode connectivity: ${OUTPUT_BASE}/mode_connectivity/"
echo ""
echo "Check the mode connectivity plot at:"
echo "  ${OUTPUT_BASE}/mode_connectivity/mode_connectivity_plot.png"
echo ""
