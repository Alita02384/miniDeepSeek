# miniDeepSeek

用 DeepSeek-V4 的结构与训练方法，做一个 **60M 级、结构同构**的研究模型。
本仓库是代码根：模型、数据、预训练/SFT、GRPO、多教师 OPD、一致性与评估全部在这里。

> **当前阶段**：按项目要求，只实现并验证**代码逻辑**，用 torchtitan 的
> `deepseek_v4_debugmodel` 跑测试；**不真实训练 mini-60M**。mini flavor 的配置与
> 参数规模已校准并记录，等代码验收后再进入训练阶段。

---

## 1. 已完成的能力

| 路线 | 代码 | 验证方式（只用 V4 debug model） |
|---|---|---|
| V4 同构模型 + mini flavor | `src/minideepseek/model/flavors.py` | debug forward/backward；mini 配置构建与参数统计 |
| 6 routed + 1 shared、top-2、hash routing | 复用 torchtitan `DeepSeekV4Model` | 路由录制/回放、专家负载、路由熵 |
| CSA(4) + HCA(32) + SWA | flavor 配置 | debug 模型覆盖三种层型；dense 对照 config |
| mHC / Sinkhorn / MTP depth 1 | 复用模型；MTP loss 自实现 | MTP logits 形状、loss、backward |
| SwiGLU clamp | `LimitedSwiGLU` + `swiglu_monitor` | monitor 计数、clamp 率 |
| Muon + AdamW 混合优化器 | `train/optimizer.py` | 参数分组、NS 正交化、单步更新 |
| loss-free bias + sequence balance loss | `train/losses.py` | bias 更新、balance loss 有限值 |
| Anticipatory Routing | `model/anticipatory.py` | stale gate probe → replay 一致性 |
| 预训练 | `train/trainer.py`, `train/run.py` | 2 步 smoke、checkpoint 重载 logits 一致 |
| 领域 SFT（response-only mask） | `data/sft.py`, `tokenizer/` | 多轮 mask、SFT 单步 |
| GRPO | `rl/grpo.py`, `rl/rollout.py`, `rl/reward.py` | rollout↔actor logprob、1 step |
| 多教师 OPD（full-vocab reverse KL） | `rl/opd.py` | 2 teacher、teacher 不更新、KL 分量 |
| router replay | `model/routing.py` | record/replay 一致率 100% |
| 一致性门禁 | `consistency/checks.py` | logprob Δ < 1e-6、routing 100%、ckpt 0 |
| 评估 | `eval/` | math / code / world rubric 报告 |
| 数据管线 | `data/` | 打包、mask、MinHash、13-gram、fertility、manifest |

测试：`93 passed`（CPU，V4 debug model，约 20 秒）。

---

## 2. 快速开始

环境：Python 3.11+，macOS/Linux CPU 即可；有 CUDA 时训练器自动使用 bf16 autocast。

```bash
# 1) 建 venv、装 torch / tokenizers / sparseml 之外的轻量依赖
./scripts/setup_env.sh

# 2) 生成 debug 数据集（MiniMind 样本，token id 限到 2048 以匹配 debug 词表）
./scripts/prepare_debug_data.sh

# 3) 端到端门禁：训练 smoke → 一致性 → 全部测试
./scripts/debug_smoke.sh
```

> `setup_env.sh` **不会**安装或改动 `../torchtitan`。`minideepseek._compat`
> 在运行时把同级 checkout 加到 `sys.path`；也可以设置
> `MINIDEEPSEEK_TORCHITAN_PATH=/path/to/torchtitan`。

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
├── model/        # flavor 构建、PackedBatch、wrapper、路由、checkpoint、anticipatory
├── tokenizer/    # MiniMind 6,400 tokenizer + chat template + response mask
├── data/         # pretrain/SFT/GRPO/OPD 数据、去重、污染、fertility、manifest
├── train/        # Muon+AdamW、scheduler、metrics、losses、trainer、CLI
├── rl/           # torch rollout、reward、GRPO、多教师 OPD
├── consistency/  # logprob / routing / checkpoint 一致性门禁
└── eval/         # math / code / world 评估与报告
tests/            # 93 个 CPU 测试，全部使用 V4 debug model
doc/              # decisions / sources / PROCESS / testing
scripts/          # 环境、数据、训练、评估脚本
```

模型实现复用只读的 `../torchtitan`（V4 架构），只把 **flavor 构建器** 复制进
`src/minideepseek/model/flavors.py` 并扩展 mini/dense 配置；来源与改动见
[doc/sources.md](doc/sources.md)。

---

## 5. 关键设计决定

- **只验证逻辑，不训练**：所有测试固定使用 `debugmodel`；mini-60M 只构建配置并统计参数量。
- **mini-60M 实测 65.26M**（含 MTP depth 1、tied embeddings），目标 63M，偏差 +3.6%。
- **6 routed + 1 shared，top-2**；hash routing 前 3 层；`route_norm=True`，`route_scale=1.5`。
- **CSA(4) + HCA(32) + SWA**；dense 对照用 `*_dense` flavor。
- **SwiGLU clamp 10.0 常开**；**Anticipatory Routing 实现但默认关闭**。
- **loss spike 检测 / 可选 rollback / 自动开启 AR**：`train/spike.py`，默认关闭。
- **4K dense-over-compressed → 8K sparse 的 index top-k 调度**：`IndexTopKSchedule`，
  默认由 `index_topk_final=64` 在最后 5% steps 内收紧。
- **GRPO/OPD 从 checkpoint 初始化**：`train.resume` 分别加载 SFT specialist 与 base student。
- **GRPO/OPD 用原生 torch 实现**（rollout + full-vocab reverse KL），因为 verl/vLLM
  在当前 CPU 验证环境不可用；算法与接口保持一致，见 [doc/decisions.md](doc/decisions.md)。
- **checkpoint 主链**：torch state dict → safetensors → DCP；三者均已验证可回读。

---

## 6. 未完成 / 明确不在当前阶段

- 没有真实训练 mini-60M，也没有 0.6B/6B 数据产出；数据只用了 MiniMind 小样本。
- 没有接入 verl / vLLM；rollout 是 full-recompute 的 torch 实现（正确性优先，无 KV cache）。
- 没有 HF `transformers` 架构类；safetensors 是权重格式，verl 直连需要再加 adapter。
- world 的默认 judge 只是启发式占位，不是 GRM；`lora_medical` 也只做了 schema 支持。
- 没有接 lm-evaluation-harness；math/code/world 是自写评估。
- 没有 DSpark / 推测解码（第一版明确跳过）。
- 多 GPU / FSDP / EP 路径未在本机验证；训练器是单进程实现。
- Anticipatory Routing 是“stale-gate probe + replay”的两段式近似，默认关闭。
- 去重、污染检查是单进程实现，6B 规模需要换成分布式版本。

更多细节见 [doc/decisions.md](doc/decisions.md)、[doc/testing.md](doc/testing.md)、
[doc/PROCESS.md](doc/PROCESS.md)。
