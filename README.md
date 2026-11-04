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
| Muon + AdamW 混合优化器 | `train/optimizer.py` | 参数分组、NS 正交化、单步更新 |
| loss-free bias + sequence balance loss | `train/losses.py` | bias 更新、balance loss 有限值 |
| Anticipatory Routing | `model/anticipatory.py` | stale gate probe → replay 一致性 |
| 预训练 / SFT | `train/trainer.py`, `train/run.py` | 2 步 smoke、checkpoint 重载 logits 一致 |
| GRPO / 多教师 OPD | `rl/` | rollout↔actor logprob、1 step、full-vocab reverse KL |
| router replay + 三道一致性门禁 | `model/routing.py`, `consistency/checks.py` | logprob Δ < 1e-3、routing 100%、ckpt 0 |
| 数据管线 | `data/` | 打包、mask、MinHash、13-gram、fertility、manifest、license |
| 评估 | `eval/` | math / code / world rubric 报告 |

测试：`94 passed`（CPU，debug model，约 19 秒）。

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
```

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
tests/            # 94 个 CPU 测试，全部使用 debug model
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

---

## 6. 未完成 / 明确不在当前阶段

- 没有真实训练 mini-60M，也没有 0.6B/6B 数据产出；数据只用了 MiniMind 小样本。
- 多 GPU / FSDP / EP 路径尚未在本机验证；训练器目前是单进程实现。
- 没有 HF `transformers` 架构类；safetensors 是权重格式。
- 没有接入 verl / vLLM；rollout 是 full-recompute 的 torch 实现（无 KV cache）。
- world 的默认 judge 只是启发式占位，不是 GRM。
- 没有接 lm-evaluation-harness；math/code/world 是自写评估。
- 没有 DSpark / 推测解码（第一版明确跳过）。
- Anticipatory Routing 是“stale-gate probe + replay”的两段式近似，默认关闭。
- 去重、污染检查是单进程实现，6B 规模需要换成分布式版本。

更多细节见 [doc/decisions.md](doc/decisions.md)、[doc/testing.md](doc/testing.md)、
[doc/PROCESS.md](doc/PROCESS.md)。
