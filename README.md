# miniDeepSeek

用 DeepSeek-V4 的结构与训练方法，做一个 **60M 级、结构同构**的研究模型。
本仓库是**自包含**的代码根：模型、tokenizer、数据、预训练/SFT、GRPO、
多教师 OPD、一致性检查与评估全部在这里；不 import 任何同级参考仓库。

> **当前阶段**：按项目要求，只实现并验证**代码逻辑**，用自带的
> `debugmodel`（与 V4 debug 配置同构）跑测试；**不真实训练 mini-60M**。
> 代码按**多卡 CUDA** 的目标环境编写，CPU/MPS 只用于逻辑验证。

---

## 1. 已完成的能力

| 路线 | 代码 | 验证方式 |
|---|---|---|
| 自包含 V4 模型（SWA/CSA/HCA、mHC、MTP、MoE） | `src/minideepseek/model/` | 与参考实现逐层数值对齐；debug forward/backward |
| mini flavor 构建 | `model/config.py` | `mini_60m` 65.26M（含 MTP depth 1）、`mini_110m` 120.65M、`mini_200m` 218.21M、dense 对照 62.52M |
| 6 routed + 1 shared、top-2、hash routing | `model/moe.py` | 路由录制/回放、专家负载、路由熵 |
| CSA(4) + HCA(32) + SWA | `model/attention.py`, `model/compressor.py` | 三种层型覆盖；indexer top-k 与参考一致 |
| mHC / Sinkhorn / MTP depth 1 | `model/mhc.py`, `model/model.py` | MTP logits/loss/backward |
| SwiGLU clamp + 监控 | `model/nn.py`, `train/metrics.py` | clamp 计数、clamp 率 |
| CSA indexer 蒸馏 loss | `model/compressor.py` | loss 有限、indexer 有梯度、eval 关闭 |
| Muon + AdamW 混合优化器 | `train/optimizer.py` | 官方 8+2 hybrid NS、gamma=0.18、AdamW eps=1e-20、单步更新 |
| loss-free bias + sequence balance loss | `train/losses.py` | bias 更新、balance loss 有限值 |
| Anticipatory Routing | `model/anticipatory.py`, `train/trainer.py` | 一步预取队列 + cache → replay；默认关 |
| 预训练 / SFT | `train/trainer.py`, `train/run.py` | 2 步 smoke、checkpoint 重载 logits 一致；batch-token warmup；flat→cosine LR；val_path；2K→4K→8K 分阶段 runner |
| GRPO / 多教师 OPD | `rl/` | rollout↔actor logprob、1 step、full-vocab reverse KL |
| router replay + 三道一致性门禁 | `model/routing.py`, `consistency/checks.py` | logprob Δ < 1e-3、routing 100%、ckpt 0 |
| 域条件路由指标 | `model/routing.py` | 按 domain 的专家负载/熵，GRPO 可开 |
| 权重版本 + logprob 门禁 | `rl/rollout.py` | vLLM 版本计数与 actor/rollout 对比 |
| 数据管线 | `data/` | 打包、mask、MinHash+段落去重、13-gram、fertility、manifest、license、加权混料、语言/vendored/math 过滤 |
| 评估 | `eval/` | math / code / world rubric 报告；可选 lm-eval general 基准 |
| world judge | `rl/judge.py` | 启发式默认；`world_judge: http` 走 OpenAI-compatible API |

测试：`125 passed`（CPU，debug model，约 28 秒；基础安装不含 `transformers` 时 HF 的 6 个测试自动跳过）。

### 模型自包含说明

`src/minideepseek/model/` 是参考实现（torchtitan `models/deepseek_v4`）的
**重构副本**：去掉了 torchtitan 的 config/parallel/sharding 框架，保留
V4 的数学与参数布局。state dict 的 key 与形状和参考实现一致，可直接对照。
来源、commit 与改动见 [doc/sources.md](doc/sources.md)。

---

## 2. 快速开始

环境：Python 3.11+；CPU 即可跑通 debug 模型，CUDA 上走 bf16。

```bash
# 1) 建 venv、装 torch / tokenizers 等轻量依赖
./scripts/setup_env.sh

# 2) 生成 debug 数据集（MiniMind 样本，token id 限到 2048 以匹配 debug 词表）
./scripts/prepare_debug_data.sh

# 3) 端到端门禁：训练 smoke → 一致性 → 全部测试
./scripts/debug_smoke.sh

# 4) 多卡（CUDA）：torchrun + FSDP2
NPROC_PER_NODE=8 ./scripts/train_pretrain.sh configs/pretrain/full_6b.yaml

# 5) 2K→4K→8K 长度课程（先准备三段数据）
.venv/bin/python -m minideepseek.data.prepare curriculum \
  --input data/raw/pretrain.jsonl --out-dir data/tokenized --seq-lens 2048 4096 8192
./scripts/train_curriculum.sh configs/pretrain/length_curriculum.yaml
```

单独运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m minideepseek.train.smoke --flavor debugmodel --steps 2
.venv/bin/python -m minideepseek.consistency.check --flavor debugmodel --demo
```

---

## 3. 配置与入口

```text
configs/
├── model/{debugmodel,mini_60m,mini_110m,mini_200m}.yaml
├── pretrain/{debug_smoke,pilot_0.6b,full_6b,dense_control_1b}.yaml
├── sft/{debug_smoke,math,code,world}.yaml
├── grpo/{debug_math,math,code,world}.yaml
├── opd/{debug_three_teacher,single_math,three_teacher}.yaml
└── eval/{debug_math,math,code,world}.yaml
```

```bash
# 数据
.venv/bin/python -m minideepseek.data.prepare pretrain --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare sft      --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare grpo     --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare opd      --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare fertility --input ...
.venv/bin/python -m minideepseek.data.prepare contamination --train ... --eval ...
.venv/bin/python -m minideepseek.data.prepare manifest --manifest data/manifests/pretrain_6b.yaml
.venv/bin/python -m minideepseek.data.prepare curriculum --input ... --out-dir data/tokenized

# 训练 / RL
./scripts/train_pretrain.sh configs/pretrain/full_6b.yaml
./scripts/train_sft.sh      configs/sft/math.yaml
./scripts/train_grpo.sh     configs/grpo/math.yaml
./scripts/train_opd.sh      configs/opd/three_teacher.yaml

# 一致性与评估
.venv/bin/python -m minideepseek.consistency.check \
  --flavor mini_60m --seq-len 4096 --checkpoint checkpoints/pretrain_full/final \
  --prompts data/eval/prompts.jsonl
.venv/bin/python -m minideepseek.eval.run --config configs/eval/math.yaml

# HF 导出（可选 extra: `pip install -e '.[hf]'`）
.venv/bin/python scripts/export_hf.py \
  --checkpoint checkpoints/pretrain_full/final --out hf/mini_60m --flavor mini_60m --seq-len 8192

# verl 数据 + 配置导出（可选 CUDA extra: `pip install -e '.[verl]'`）
.venv/bin/python scripts/export_verl.py \
  --hf-model hf/mini_60m --train data/grpo/math.jsonl --out-dir runs/verl_math --rollout hf
```

vLLM rollout 在 CUDA 上启用：`configs/grpo/*.yaml` 设置

```yaml
rl:
  rollout_backend: vllm
  vllm_tensor_parallel_size: 1
  vllm_gpu_memory_utilization: 0.4
```

> vLLM 需要能通过 `auto_map` 加载 `MiniDeepSeekForCausalLM`。当前代码只做
> HF 导出、采样和保守的权重重建；没有为自定义稀疏注意力写 paged-attention
> kernel。若目标 vLLM 版本不支持该架构，用 `rollout_backend: torch` 或
> verl 的 HF rollout。

---

## 4. 目录

```text
src/minideepseek/
├── model/        # 自包含 V4 模型、flavor、PackedBatch、wrapper、路由、checkpoint
├── tokenizer/    # MiniMind 6,400 tokenizer + chat template + response mask
├── data/         # pretrain/SFT/GRPO/OPD 数据、去重、污染、fertility、manifest
├── train/        # Muon+AdamW、scheduler、metrics、losses、trainer、CLI
├── rl/           # torch rollout、reward、GRPO、多教师 OPD
├── consistency/  # logprob / routing / checkpoint 一致性门禁
└── eval/         # math / code / world 评估与报告
tests/            # 125 个 CPU 测试，全部使用 debug model
doc/              # decisions / sources / PROCESS / testing
scripts/          # 环境、数据、训练、评估脚本
```

---

## 5. 关键设计决定

- **只验证逻辑，不训练**：所有测试固定使用 `debugmodel`；mini 配置只构建并统计参数量。
- **mini-60M 实测 65.26M**（含 MTP depth 1、tied embeddings），目标 63M，偏差 +3.6%。
- **6 routed + 1 shared，top-2**；hash routing 前 3 层；`route_norm=True`，`route_scale=1.5`。
- **CSA(4) + HCA(32) + SWA**；dense 对照用 `*_dense` flavor。
- **SwiGLU clamp 10.0 常开**；**Anticipatory Routing 实现但默认关闭**。
- **4K dense-over-compressed → 8K sparse 的 index top-k 调度**：`IndexTopKSchedule`。
- **GRPO/OPD 从 checkpoint 初始化**：`train.resume` 分别加载 SFT specialist 与 base student。
- **checkpoint 主链**：torch state dict → safetensors → DCP；三者均已验证可回读。
- **长度课程**：`train/curriculum.py` 分阶段 2K→4K→8K，累计 step + resume；不是单次动态改 seq_len。
- **数据混料**：`data.sources` 权重混料；manifest 继续负责 license/epoch 校验。
- **AR**：预取下一 optimizer step 的 batch 并用当前 gate 缓存 routing；无异步 overlap。

---

## 6. 未完成 / 明确不在当前阶段

- 没有真实训练 mini-60M，也没有 0.6B/6B 数据产出；数据只用了 MiniMind 小样本。
- 没有多卡 CUDA 实机：FSDP2/DDP 的 2 进程正确性已在 CPU/Gloo 验证，
  NCCL、真实 bf16 吞吐、8 卡扩展性、EP 未验证。
- EP（专家并行）未实现：FSDP2 下 6 个 routed expert 是复制分片，尚未做
  all-to-all expert dispatch。
- vLLM 自定义模型/paged-attention kernel 未实现；verl 的 HF rollout 路径
  未在 GPU 上实跑。vLLM/verl 都只是可选集成。
- rollout 默认仍是 full-recompute torch（无 KV cache），适合一致性验证，不适合吞吐。
- world 的默认 judge 仍是启发式占位；HTTP judge 代码已接，但需要 API 与环境变量，
  未在 CI 中真实调用。
- 没有跑 lm-evaluation-harness（`eval/lm_eval.py` 适配已写，需可选 extra 与 GPU 实跑）。
- 没有 DSpark / 推测解码（第一版明确跳过）。
- Anticipatory Routing 是“stale-gate probe + replay”的两段式近似，默认关闭。
- 去重、污染检查是单进程实现，6B 规模需要换成分布式版本。

逐项实现审计（哪些只是代码落地、哪些还没做）见
[doc/roadmap_audit.md](doc/roadmap_audit.md)。更多细节见
[doc/decisions.md](doc/decisions.md)、[doc/testing.md](doc/testing.md)、
[doc/PROCESS.md](doc/PROCESS.md)。
