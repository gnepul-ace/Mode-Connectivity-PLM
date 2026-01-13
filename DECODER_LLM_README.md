# Mode Connectivity Analysis for Decoder-Only LLMs

**专注于分析** - 无需训练，直接使用已训练好的模型进行mode connectivity分析。

## 核心功能

✅ **直接从HuggingFace加载模型**
✅ **支持reasoning benchmarks** (GSM8K, MATH)
✅ **Mode connectivity分析** (线性插值)
✅ **正确的chat template处理** (关键!)
✅ **支持多种checkpoint格式** (LoRA, Adapter, Full model)

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers datasets peft pandas numpy matplotlib
```

### 2. 准备模型

你可以：
- 从HuggingFace下载已训练好的模型
- 使用其他repo训练的模型checkpoint
- 使用RL训练（DAPO/DR GRPO等）得到的checkpoint

### 3. Mode Connectivity分析

```bash
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --load_PET_path_1 ./checkpoints/model1_lora.pt \
  --load_PET_path_2 ./checkpoints/model2_lora.pt \
  --itpl_points 11 \
  --output_dir ./outputs/connectivity \
  --max_input_length 512 \
  --max_output_length 512 \
  --eval_batch_size 8
```

### 4. 单独评估一个模型

```bash
python evaluate_decoder_llm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --checkpoint ./checkpoints/model_lora.pt \
  --output_dir ./outputs/eval \
  --max_input_length 512 \
  --max_output_length 512
```

## 支持的Checkpoint格式

脚本会自动识别以下格式：

### LoRA weights
```python
{
  'lora': {
    'base_model.model.layers.0.self_attn.q_proj.lora_A': tensor(...),
    'base_model.model.layers.0.self_attn.q_proj.lora_B': tensor(...),
    ...
  }
}
```

### Adapter weights
```python
{
  'adapter': {
    'layers.0.attn_adapter.adapter_A': tensor(...),
    'layers.0.attn_adapter.adapter_B': tensor(...),
    ...
  }
}
```

### Full model
```python
{
  'model': {
    'model.layers.0.self_attn.q_proj.weight': tensor(...),
    ...
  }
}
```

### 直接state dict
```python
{
  'model.layers.0.self_attn.q_proj.weight': tensor(...),
  ...
}
```

## 支持的模型

### Qwen系列
```bash
--model Qwen/Qwen2.5-0.5B-Instruct
--model Qwen/Qwen2.5-1.5B-Instruct
--model Qwen/Qwen2.5-7B-Instruct
```

### Llama系列
```bash
--model meta-llama/Llama-3.2-1B-Instruct
--model meta-llama/Llama-3.2-3B-Instruct
--model meta-llama/Llama-3.1-8B-Instruct
```

**注意**: Llama模型需要HuggingFace认证:
```bash
huggingface-cli login
```

## 支持的数据集

### GSM8K (Grade School Math)
- 8,500道小学数学题
- 需要2-8步推理
- 评估：数值精确匹配

```bash
--dataset gsm8k
```

### MATH (Competition Mathematics)
- 12,500道竞赛数学题
- AMC/AIME难度
- 评估：提取答案精确匹配

```bash
--dataset math
```

## Chat Template的重要性 ⚠️

**不同模型需要不同的chat format！** 使用错误的模板会严重影响性能。

### Qwen格式
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?

Let's think step by step.<|im_end|>
<|im_start|>assistant
```

### Llama格式
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is 2+2?

Let's think step by step.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

**好消息**: `ChatTemplateHandler`会自动检测模型类型并使用正确的格式！

## 关键参数说明

### 模型相关
```bash
--model <model_name>          # HuggingFace model ID
--tune_method lora            # lora, adapter, or model
--checkpoint <path>           # 单个模型评估时使用
```

### Mode Connectivity相关
```bash
--load_PET_path_1 <path>      # 第一个checkpoint
--load_PET_path_2 <path>      # 第二个checkpoint
--itpl_points 11              # 插值点数量（包括端点）
```

### 数据集相关
```bash
--dataset gsm8k               # gsm8k 或 math
--max_input_length 512        # 最大输入长度
--max_output_length 512       # 最大输出长度
--eval_batch_size 8           # 评估batch size
```

### 性能优化
```bash
--bf16                        # 使用BF16混合精度（推荐）
--cache_dir /path/to/cache    # 模型缓存目录
```

## 输出结果

### Mode Connectivity结果

CSV文件格式 (`interpolation_results_gsm8k.csv`):
```csv
x,metric,performance,loss
0.0,accuracy,0.42,0.0
0.1,accuracy,0.41,0.0
0.2,accuracy,0.40,0.0
...
1.0,accuracy,0.43,0.0
```

### 可视化图表

自动生成 `interpolation_plot_gsm8k.png`:
- X轴: 插值系数 (0到1)
- Y轴: 准确率
- 红星: 端点（原始模型）

**解读**:
- 平坦曲线: 强mode connectivity（模型在同一mode）
- V形曲线: mode之间有barrier
- 高于端点: beneficial interpolation（罕见）

## 实际使用场景

### 场景1: 分析从HuggingFace下载的模型

```bash
# 假设你从HuggingFace下载了两个LoRA checkpoint
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --load_PET_path_1 ./downloaded/model1_lora.pt \
  --load_PET_path_2 ./downloaded/model2_lora.pt \
  --itpl_points 11 \
  --output_dir ./results
```

### 场景2: 分析RL训练后的模型

```bash
# 使用DAPO/DR GRPO训练得到的checkpoint
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --load_PET_path_1 ./rl_trained/before_rl.pt \
  --load_PET_path_2 ./rl_trained/after_rl.pt \
  --itpl_points 21 \
  --output_dir ./rl_analysis
```

### 场景3: 分析不同训练方法

```bash
# 比较LoRA vs Adapter
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset math \
  --tune_method lora \
  --load_PET_path_1 ./lora_model/checkpoint.pt \
  --load_PET_path_2 ./adapter_model/checkpoint.pt \
  --itpl_points 11 \
  --output_dir ./comparison
```

## 代码结构

```
Mode-Connectivity-PLM/
├── DecoderLLM_model/
│   ├── decoder_evaluator.py      # 评估器（无训练功能）
│   ├── modeling_decoder.py       # 模型加载
│   └── chat_template.py          # Chat template处理 ⚠️ 关键!
│
├── dataloader/reasoning/
│   ├── reasoning_loader.py       # GSM8K, MATH加载
│   └── reasoning_metrics.py      # 评估指标
│
├── decoder_interpolation.py      # Mode connectivity分析
├── evaluate_decoder_llm.py       # 单模型评估
└── utils/options.py               # 命令行参数
```

## 常见问题

### Q: 如何处理不同格式的checkpoint?

A: 脚本会自动尝试多种格式。如果遇到问题，可以手动指定：

```python
# 在Python中手动加载
import torch
checkpoint = torch.load('model.pt')

# 查看结构
print(checkpoint.keys())

# 提取需要的部分
if 'lora' in checkpoint:
    weights = checkpoint['lora']
```

### Q: 模型太大，内存不足怎么办?

A: 使用更小的batch size和混合精度:

```bash
--eval_batch_size 1 \
--bf16
```

### Q: 如何使用自己训练的checkpoint?

A: 确保checkpoint格式正确：

```python
# 保存LoRA checkpoint
torch.save({
    'lora': model.state_dict()  # 或只保存LoRA参数
}, 'my_lora_checkpoint.pt')
```

然后直接使用:
```bash
--load_PET_path_1 ./my_lora_checkpoint.pt
```

### Q: 支持哪些LoRA框架?

A: 支持：
- PEFT library (HuggingFace)
- 手动实现的LoRA
- 任何保存为state dict的参数

关键是参数命名要包含`lora_A`, `lora_B`等标识。

## 性能参考

基于DAPO/DR GRPO论文的结果：

| 模型 | GSM8K | MATH |
|------|-------|------|
| Qwen2.5-0.5B + LoRA | 30-45% | 15-25% |
| Qwen2.5-1.5B + LoRA | 50-65% | 25-35% |
| Qwen2.5-7B + LoRA | 75-85% | 45-60% |

## 参考资料

- **原始论文**: "Exploring Mode Connectivity for Pre-trained Language Models" (EMNLP 2022)
- **GSM8K**: https://github.com/openai/grade-school-math
- **MATH**: https://github.com/hendrycks/math
- **DAPO**: Decoupled Advantage Policy Optimization
- **DR GRPO**: Dr. GRPO (removing std normalization)

## 引用

```bibtex
@inproceedings{he-etal-2022-exploring,
    title = "Exploring Mode Connectivity for Pre-trained Language Models",
    booktitle = "Proceedings of EMNLP",
    year = "2022",
}
```

## License

遵循原始Mode-Connectivity-PLM仓库的License。
