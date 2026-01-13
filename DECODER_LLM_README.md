# Mode Connectivity Analysis for Decoder-Only LLMs

**专注于分析** - 直接使用已训练好的模型（full parameter或PET）进行mode connectivity分析。

## 核心功能

✅ **支持完整参数模型** (Full Parameter) - 最常用！
✅ **直接从HuggingFace加载模型**
✅ **支持reasoning benchmarks** (GSM8K, MATH)
✅ **Mode connectivity分析** (线性插值)
✅ **正确的chat template处理** (关键!)
✅ **灵活的checkpoint加载** (自动检测格式)

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers datasets pandas numpy matplotlib
# 如果使用LoRA: pip install peft
```

### 2. Mode Connectivity分析（Full Parameter模型）

**支持两种模型来源**：
1. 本地checkpoint文件（.pt, .pth, .bin）
2. HuggingFace模型ID（username/model-name）

#### 示例1：两个本地checkpoint

```bash
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --model_path_1 ./model1/checkpoint.pt \
  --model_path_2 ./model2/checkpoint.pt \
  --itpl_points 11 \
  --output_dir ./outputs/connectivity \
  --max_input_length 512 \
  --max_output_length 512 \
  --eval_batch_size 8
```

#### 示例2：两个HuggingFace模型

```bash
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --model_path_1 username/qwen-model-rl1 \
  --model_path_2 username/qwen-model-rl2 \
  --itpl_points 11 \
  --output_dir ./outputs/connectivity
```

#### 示例3：混合（本地 + HuggingFace）

```bash
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset math \
  --model_path_1 ./local_checkpoint.pt \
  --model_path_2 username/hf-finetuned-model \
  --itpl_points 21 \
  --output_dir ./outputs/connectivity
```

**注意**：
- 不需要指定`--tune_method`，默认就是full parameter模式！
- `--model`参数指定基础模型架构（用于初始化tokenizer和配置）
- `--load_PET_path_1/2`已弃用，请使用`--model_path_1/2`

### 3. 评估单个模型

```bash
python evaluate_decoder_llm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --checkpoint ./your_model/checkpoint.pt \
  --output_dir ./outputs/eval
```

## 支持的Checkpoint格式

代码会**自动检测**以下格式（按优先级）：

### 1. 直接State Dict（最常见，Full Parameter）
```python
{
  'model.layers.0.self_attn.q_proj.weight': tensor(...),
  'model.layers.0.self_attn.k_proj.weight': tensor(...),
  ...
}
```

**这是最常见的格式！** 直接保存`model.state_dict()`的结果。

### 2. 包装在'model'或'state_dict'键中
```python
{
  'model': {
    'model.layers.0.self_attn.q_proj.weight': tensor(...),
    ...
  },
  'optimizer': {...},  # 可选
  'epoch': 10,         # 可选
}
```

或：
```python
{
  'state_dict': {
    'model.layers.0.self_attn.q_proj.weight': tensor(...),
    ...
  }
}
```

### 3. LoRA weights（如果使用）
```python
{
  'lora': {
    'base_model.model.layers.0.self_attn.q_proj.lora_A': tensor(...),
    ...
  }
}
```

### 4. Adapter weights（如果使用）
```python
{
  'adapter': {
    'layers.0.attn_adapter.adapter_A': tensor(...),
    ...
  }
}
```

**代码会自动尝试所有格式，你不需要担心！**

## 实际使用示例

### 示例1：分析RL训练前后的模型（Full Parameter）

```bash
# 你有两个完整参数的checkpoint
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset gsm8k \
  --load_PET_path_1 ./checkpoints/before_rl.pt \
  --load_PET_path_2 ./checkpoints/after_rl.pt \
  --itpl_points 21 \
  --output_dir ./results/rl_analysis
```

### 示例2：比较不同训练seed的模型

```bash
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset math \
  --load_PET_path_1 ./checkpoints/seed_42.pt \
  --load_PET_path_2 ./checkpoints/seed_123.pt \
  --itpl_points 11 \
  --output_dir ./results/seed_comparison
```

### 示例3：分析不同训练阶段

```bash
python decoder_interpolation.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset gsm8k \
  --load_PET_path_1 ./checkpoints/epoch_1.pt \
  --load_PET_path_2 ./checkpoints/epoch_5.pt \
  --itpl_points 11 \
  --output_dir ./results/training_stages
```

### 示例4：使用LoRA checkpoint（如果你有）

```bash
python decoder_interpolation.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --tune_method lora \
  --load_PET_path_1 ./lora_checkpoints/model1.pt \
  --load_PET_path_2 ./lora_checkpoints/model2.pt \
  --itpl_points 11 \
  --output_dir ./results/lora_analysis
```

## 支持的模型

### Qwen系列（推荐）
```bash
--model Qwen/Qwen2.5-0.5B-Instruct
--model Qwen/Qwen2.5-1.5B-Instruct
--model Qwen/Qwen2.5-7B-Instruct
--model Qwen/Qwen2.5-14B-Instruct
--model Qwen/Qwen2.5-32B-Instruct
```

### Llama系列
```bash
--model meta-llama/Llama-3.2-1B-Instruct
--model meta-llama/Llama-3.2-3B-Instruct
--model meta-llama/Llama-3.1-8B-Instruct
```

**注意**: Llama模型需要HuggingFace认证：
```bash
huggingface-cli login
```

### 其他Instruct模型
理论上支持任何HuggingFace上的instruct模型，只要它：
- 是decoder-only架构
- 有对应的tokenizer
- 支持`generate()`方法

## 支持的数据集

### GSM8K (Grade School Math)
- **简介**: 8,500道小学数学题
- **难度**: 2-8步推理
- **格式**: 答案格式为 `#### 数字`
- **评估**: 数值精确匹配

```bash
--dataset gsm8k
```

### MATH (Competition Mathematics)
- **简介**: 12,500道竞赛数学题
- **难度**: AMC/AIME级别
- **格式**: 答案在`\boxed{}`中
- **评估**: 提取答案后精确匹配

```bash
--dataset math
```

## Chat Template的重要性 ⚠️

**不同模型需要不同的chat format！** 这对推理性能**至关重要**！

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

**好消息**: `ChatTemplateHandler`会**自动检测**模型类型并使用正确格式 + 添加CoT提示词！

## 关键参数

### 基本参数（必需）
```bash
--model <model_name>              # HuggingFace model ID
--dataset gsm8k                   # gsm8k 或 math
--load_PET_path_1 <path>          # 第一个checkpoint
--load_PET_path_2 <path>          # 第二个checkpoint
--output_dir ./results            # 输出目录
```

### 可选参数
```bash
--tune_method model               # model(默认), lora, adapter
--itpl_points 11                  # 插值点数（默认11）
--max_input_length 512            # 最大输入长度
--max_output_length 512           # 最大输出长度
--eval_batch_size 8               # 评估batch size
--bf16                            # 使用BF16（推荐）
--cache_dir /path/to/cache        # 模型缓存目录
```

## 输出结果

### 1. CSV结果文件

`interpolation_results_gsm8k.csv`:
```csv
x,metric,performance,loss
0.0,accuracy,0.42,0.0
0.1,accuracy,0.41,0.0
0.2,accuracy,0.40,0.0
0.3,accuracy,0.39,0.0
0.4,accuracy,0.38,0.0
0.5,accuracy,0.39,0.0
0.6,accuracy,0.40,0.0
0.7,accuracy,0.41,0.0
0.8,accuracy,0.42,0.0
0.9,accuracy,0.42,0.0
1.0,accuracy,0.43,0.0
```

### 2. 可视化图表

自动生成 `interpolation_plot_gsm8k.png`:
- **X轴**: 插值系数 (0到1)
- **Y轴**: 准确率
- **红星**: 端点（原始模型）
- **蓝线**: 插值路径

### 3. 日志文件

`interpolation_log.txt`: 包含详细的运行日志

## 解读结果

### Mode Connectivity的含义

**平坦曲线** ✅
```
accuracy
  ^
  |  ____________________
  | /                    \
  |*                      *
  +-----------------------> x
  0                       1
```
→ **强mode connectivity**: 两个模型在同一个mode中

**V形曲线** ⚠️
```
accuracy
  ^
  |*                      *
  | \                    /
  |  \                  /
  |   \________________/
  +-----------------------> x
  0                       1
```
→ **Barrier存在**: 两个模型在不同mode中

**高于端点** 🎉
```
accuracy
  ^
  |      /\
  |     /  \
  |    /    \
  |*  /      \  *
  +-----------------------> x
  0           1
```
→ **Beneficial interpolation**: 插值模型优于两个端点（罕见）

## 常见checkpoint格式问题

### Q1: 我的checkpoint格式是什么样的？

**查看方法**:
```python
import torch

checkpoint = torch.load('your_checkpoint.pt')
print("Keys:", checkpoint.keys())

# 如果是嵌套的
if 'model' in checkpoint:
    print("Model keys (first 5):", list(checkpoint['model'].keys())[:5])
```

### Q2: 我的checkpoint很大，怎么办？

**常见格式**:
```python
# 如果你保存时这样做的：
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'checkpoint.pt')

# 代码会自动提取'model'部分！
```

### Q3: 如何确保checkpoint兼容？

**确保两个checkpoint**:
1. 来自同一个base model（如都是Qwen2.5-0.5B）
2. 有相同的参数结构
3. 都是完整参数或都是同一种PET方法

代码会自动检查参数是否匹配！

## 性能参考

基于DAPO/DR GRPO等论文的结果：

| 模型 | GSM8K | MATH |
|------|-------|------|
| Qwen2.5-0.5B | 30-45% | 15-25% |
| Qwen2.5-1.5B | 50-65% | 25-35% |
| Qwen2.5-7B | 75-85% | 45-60% |

*实际性能取决于训练方法、数据质量等因素*

## 代码结构

```
Mode-Connectivity-PLM/
├── DecoderLLM_model/
│   ├── decoder_evaluator.py      # 评估器（只评估，无训练）
│   ├── modeling_decoder.py       # 模型加载（支持full/LoRA/adapter）
│   └── chat_template.py          # Template处理 ⚠️ 关键!
│
├── dataloader/reasoning/
│   ├── reasoning_loader.py       # GSM8K, MATH加载器
│   └── reasoning_metrics.py      # 评估指标
│
├── decoder_interpolation.py      # 主脚本：Mode connectivity分析
├── evaluate_decoder_llm.py       # 单模型评估
└── utils/options.py               # 命令行参数
```

## 故障排除

### 内存不足 (OOM)

```bash
# 减小batch size
--eval_batch_size 1

# 使用混合精度
--bf16

# 如果还不够，使用更小的模型
--model Qwen/Qwen2.5-0.5B-Instruct
```

### Checkpoint加载失败

```python
# 1. 检查checkpoint内容
import torch
ckpt = torch.load('checkpoint.pt')
print(ckpt.keys())

# 2. 手动提取state dict
if 'model' in ckpt:
    state_dict = ckpt['model']
    torch.save(state_dict, 'clean_checkpoint.pt')
```

### 两个checkpoint参数不匹配

确保：
- 来自同一个base model
- 参数数量相同
- 参数名称匹配

脚本会自动只使用共同的参数！

### 模型生成结果不对

检查chat template：
```python
from transformers import AutoTokenizer
from DecoderLLM_model.chat_template import ChatTemplateHandler

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
handler = ChatTemplateHandler("Qwen/Qwen2.5-0.5B-Instruct", tokenizer)

# 测试格式
print(handler.format_chat("What is 2+2?"))
```

## 与训练框架的集成

### 从PyTorch Lightning保存的checkpoint

```python
# PyTorch Lightning格式
checkpoint = torch.load('checkpoint.ckpt')
state_dict = checkpoint['state_dict']

# 可能需要移除'model.'前缀
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

torch.save(state_dict, 'clean_checkpoint.pt')
```

### 从HuggingFace Trainer保存的checkpoint

```python
# HuggingFace格式
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('./trainer_output')
torch.save(model.state_dict(), 'checkpoint.pt')
```

### 从DeepSpeed保存的checkpoint

```bash
# DeepSpeed会保存多个文件，需要合并
python zero_to_fp32.py ./deepspeed_output/checkpoint ./merged_checkpoint.pt
```

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
