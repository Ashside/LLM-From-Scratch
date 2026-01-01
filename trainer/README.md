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
  --data_path ../dataset/sft_512.jsonl \
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
CUDA_LAUNCH_BLOCKING=1 \
torchrun --nproc_per_node 2 train_grpo.py \
  --hidden_size 640 --num_hidden_layers 12 --use_moe 1 \
  --headwise_attn_output_gate 1 \
  --data_path /root/autodl-tmp/LLM-From-Scratch/dataset/sft_mini_512.jsonl \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --num_generations 1 \
  --max_gen_len 128 \
  --reasoning 0 \
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

---

## PPO / GRPO / DPO 原理（结合本仓库实现）

本节目标：用“公式 + 直觉 + 代码变量对照”的方式，帮助你读懂本仓库的
`train_ppo.py`、`train_grpo.py`、`train_dpo.py`。

### 0) 统一符号与对齐方式

- prompt 记为 $x$，模型生成的 completion/response 记为 $y$。
- 策略模型（要更新的模型）记为 $\pi_\theta$。
- 参考模型（冻结、用于约束）记为 $\pi_{\text{ref}}$。
- reward model 给整段 response 一个标量奖励 $R(x,y)$。

本仓库的常见变量对照：

- PPO：
  - `actor_model` 对应 $\pi_\theta$，`old_actor_model` 对应 $\pi_{\text{old}}$，`ref_model` 对应 $\pi_{\text{ref}}$。
  - 序列级 logprob：`actor_logp` / `old_logp` / `ref_logp` 对应 $\log \pi(y|x)$。
  - 优势：`advantages = rewards - values.detach()` 对应 $A = R - V$。
  - 裁剪：`args.clip_epsilon` 对应 PPO 的 $\epsilon$。
  - KL 正则：`args.kl_coef * kl_ref`。
  - value loss：`args.vf_coef * value_loss`。

- GRPO：
  - `model` 对应 $\pi_\theta$，`ref_model` 对应 $\pi_{\text{ref}}$。
  - 多次采样：`args.num_generations`。
  - token 级 logprob：`per_token_logps` / `ref_per_token_logps`。
  - 组内标准化优势：`advantages`（由 `grouped_rewards` 计算得到）。
  - KL 系数：`args.beta`。

- DPO：
  - `model` 对应 $\pi_\theta$，`ref_model` 对应 $\pi_{\text{ref}}$。
  - `chosen_*` / `rejected_*` 对应偏好对 $(y^+, y^-)$。
  - `args.beta`（DPO 的温度/强度参数）。

### 1) PPO（train_ppo.py）

PPO 的核心思想：用“旧策略”产生的数据来更新“新策略”，但更新不能太激进。

1) **采样与奖励**

- 采样：$y \sim \pi_\theta(\cdot|x)$（实现上用 `actor_model.generate` 采样）
- 奖励：$R = R(x,y)$（实现上 `calculate_rewards`）

2) **Critic/value 与优势**

经典 PPO 会用 value 网络 $V_\phi$ 估计期望回报，并构造优势 $A$。
本仓库用一个 `CriticModel`（在 LM 顶上加 `value_head`）输出每个位置的 value，
取序列最后一个非 padding token 的 value 作为整段序列的 $V$：

$$A = R - V$$

对应代码：

- `values = values_seq[..., last_indices]`
- `advantages = rewards - values.detach()`

3) **PPO clipped objective（序列级）**

标准 PPO 的 clipped surrogate 目标：

$$L^{\text{clip}}(\theta) = \mathbb{E}[\min(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon,1+\epsilon)A)]$$

其中 $r(\theta)=\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}$。
本仓库用“序列级” logprob 来做 $\log \pi(y|x)$：对 response token 的 logprob 求和。

对应代码：

- `ratio = exp(actor_logp - old_logp)`
- `surr1/surr2` 和 `policy_loss = -min(surr1,surr2).mean()`

4) **参考模型 KL 正则（防止跑飞）**

实践中常引入参考策略约束：

$$L = L_{\text{policy}} + c_v L_{\text{value}} + c_{\text{kl}}\,\text{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$$

本仓库用 `kl_ref = (actor_logp - ref_logp).mean()` 作为近似的监控/惩罚项，
并加到总 loss：`loss = policy_loss + vf_coef * value_loss + kl_coef * kl_ref`。

注意：这里的 `kl_ref` 严格意义上不是完整 KL（没有对分布求期望），
但常被用作简单可用的“不要偏离 ref 太远”的约束信号。

### 2) GRPO（train_grpo.py）

GRPO 的关键点：

1) **同一 prompt 采样多次**

对每个 prompt $x$，采样 $G$ 条 response：

$$y_1,\dots,y_G \sim \pi_\theta(\cdot|x)$$

对应代码：`num_return_sequences=args.num_generations`，输出 batch 形状变为 `[B*G, ...]`。

2) **组内标准化优势（不需要 critic）**

对同一 prompt 的奖励做标准化：

$$A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

其中 $\mu_R,\sigma_R$ 是该 prompt 的 $G$ 条样本的均值/标准差。
直觉：优势只关心“在同一 prompt 的候选里，相对更好/更差”，
从而减小方差，避免训练 value 网络。

对应代码：

- `grouped_rewards = rewards.view(-1, num_generations)`
- `mean_r/std_r` + `advantages = (rewards-mean_r)/(std_r+eps)`

3) **token 级 logprob 与 mask**

GRPO 这里用 token 级 logprob（每个 completion token 的 logp），并用 `completion_mask`
只统计 EOS 之前的 token。

对应代码：

- `per_token_logps`: `[B*G, R]`
- `completion_mask`: `[B*G, R]`

4) **策略梯度项（实现技巧）**

经典 REINFORCE 的形式是：

$$\nabla_\theta\,\mathbb{E}[A\,\log \pi_\theta]$$

代码里写成 `exp(logp - logp.detach()) * A` 的形式：

- 前向：$\exp(\log p - \text{stopgrad}(\log p)) = 1$（不改变标量值）
- 反向：对 $\log p$ 仍有梯度，相当于实现 $A\,\nabla_\theta \log \pi_\theta$

这能让 loss 写法更接近“ratio 风格”，同时数值稳定。

5) **reference KL 惩罚（token 级）**

本仓库用一个非负的 KL proxy（常见于 RLHF 实现）来惩罚偏离 ref：

$$\text{KL}_{\text{proxy}}(p\|q) = \exp(\log q - \log p) - (\log q - \log p) - 1$$

对应代码：

- `kl_div = ref_per_token_logps - per_token_logps`
- `per_token_kl = exp(kl_div) - kl_div - 1`
- 总 loss：`-(policy_term - beta * per_token_kl)`

### 3) DPO（train_dpo.py）

DPO 的出发点：不直接做 RL rollout，而把“人类偏好/奖励模型偏好”变成
一个监督式的对比学习目标。

对同一 prompt $x$，有偏好对 $(y^+, y^-)$（chosen vs rejected）。
DPO 的经典目标可以写成：

$$L = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x)) - (\log\pi_{\text{ref}}(y^+|x)-\log\pi_{\text{ref}}(y^-|x))\big]\Big)$$

直觉：让策略模型相对 ref 更“偏好 chosen、厌恶 rejected”。

本仓库实现要点：

- `logits_to_log_probs` 把 `[B, T, V]` 变成每个 token 的 logprob `[B, T]`。
- `dpo_loss` 先用 `mask` 做序列平均 logprob，再拆成 chosen/rejected 两半计算对比项。
- `beta` 越大，更新越激进；越小越保守。

### 4) 什么时候选 PPO / GRPO / DPO（经验对比）

- DPO：数据驱动（偏好对），工程最简单、训练稳定；但受数据质量/覆盖面影响大。
- PPO：更“正统”的 RLHF/RLAIF，能显式控制 KL 与 value；但工程复杂、对超参更敏感。
- GRPO：不需要 critic，适合“每个 prompt 多采样、用组内相对优势稳定训练”的场景；
  但采样开销更大（num_generations 倍），而且实现/目标函数在不同论文/代码库里细节差异较多。

