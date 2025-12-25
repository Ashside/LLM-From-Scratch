# MiniMind Trainer Commands (2×4090 / DDP)

> 按项目约定，建议先 `cd ./trainer` 再运行所有训练脚本（相对路径与默认数据/权重路径都按此设计）。

## 统一模型参数（保持一致）

- `hidden_size=640`
- `num_hidden_layers=12`
- `use_moe=1`
- `headwise_attn_output_gate=1`
- 全部开启 `--use_wandb`
- 全部使用双卡：`torchrun --nproc_per_node 2 ...`

```bash
cd ./trainer
export CUDA_VISIBLE_DEVICES=0,1
```

## 训练脚本命令

### 1) 预训练 Pretrain

```bash
torchrun --nproc_per_node 2 train_pretrain.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 2) 监督微调 Full SFT

```bash
torchrun --nproc_per_node 2 train_full_sft.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 3) LoRA 微调

```bash
torchrun --nproc_per_node 2 train_lora.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 4) DPO

```bash
torchrun --nproc_per_node 2 train_dpo.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 5) 推理蒸馏（Reasoning Distill）

```bash
torchrun --nproc_per_node 2 train_distill_reason.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 6) 蒸馏（Student/Teacher Distillation）

> 为满足“参数配置保持一致”，这里让 Student/Teacher 使用同一组结构参数；如需更强 Teacher，可自行把 `teacher_hidden_size/teacher_num_layers` 调大。

```bash
torchrun --nproc_per_node 2 train_distillation.py \
  --student_hidden_size 640 --student_num_layers 12 \
  --teacher_hidden_size 640 --teacher_num_layers 12 \
  --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 7) PPO（RLAIF）

```bash
torchrun --nproc_per_node 2 train_ppo.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 8) GRPO（RLAIF）

```bash
torchrun --nproc_per_node 2 train_grpo.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb
```

### 9) SPO（RLAIF）

```bash
torchrun --nproc_per_node 2 train_spo.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --use_wandb

> 注意：`headwise_attn_output_gate` 会改变注意力 `q_proj` 的参数形状；如果你要加载的是“未启用该选项”训练出来的旧权重，请显式加 `--headwise_attn_output_gate 0`（或换一个 `--save_weight/--save_dir` 以避免覆盖/混淆）。
```
