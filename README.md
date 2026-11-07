# miniDeepSeek

用 DeepSeek-V4 的结构与训练方法，做一组**同构的研究模型**：
**默认 `mini_220m_a80m`**（216.65M 总参 / 82.17M 激活），另有
`mini_145m_a60m`（145.84M / 60.91M）和 `mini_60m_dense_d768`（60.88M / 60.88M）。
本仓库是**自包含**的代码根：模型、tokenizer、数据、预训练/SFT、GRPO、
多教师 OPD、一致性检查与评估全部在这里；不 import 任何同级参考仓库。

> **当前阶段**：按项目要求，只实现并验证**代码逻辑**，用自带的
> `debugmodel`（与 V4 debug 配置同构）跑测试；**不真实训练这三个模型**。
> 代码按**多卡 CUDA** 的目标环境编写，CPU/MPS 只用于逻辑验证。

> **2026-11-07 路线更新**：三个 specialist 改为 **general / tool-use / code**。
> 当前可训练 general 与 tool-use；code 的数据与 GRPO reward 暂缺。OPD 先跑
> 2-teacher（general + tool），code 补齐后切 3-teacher。world/medical 不再作为
> specialist。数据平铺在 `data/*.jsonl`，tokenization 在 `data/tokenization/`。
> 详见 [doc/data_pipeline.md](doc/data_pipeline.md)。

---

## 1. 已完成的能力

| 路线 | 代码 | 验证方式 |
|---|---|---|
| 自包含 V4 模型（SWA/CSA/HCA、mHC、MTP、MoE） | `src/minideepseek/model/` | 与参考实现逐层数值对齐；debug forward/backward |
| 模型配置加载 | `model/config.py`, `configs/model/` | `mini_220m_a80m` 216.65M / 82.17M 激活、`mini_145m_a60m` 145.84M / 60.91M、`mini_60m_dense_d768` 60.88M（均含 MTP depth 1、tied embeddings） |
| MoE：8 routed/top-2（220m）、6 routed/top-2（145m）、1 dense expert（60m） | `model/moe.py` | 路由录制/回放、专家负载、路由熵 |
| CSA(4) + HCA(32) + SWA | `model/attention.py`, `model/compressor.py` | 三种层型覆盖；indexer top-k 与参考一致 |
| mHC / Sinkhorn / MTP depth 1 | `model/mhc.py`, `model/model.py` | MTP logits/loss/backward |
| SwiGLU clamp + 监控 | `model/nn.py`, `train/metrics.py` | clamp 计数、clamp 率 |
| CSA indexer 蒸馏 loss | `model/compressor.py` | loss 有限、indexer 有梯度、eval 关闭 |
| Muon + AdamW 混合优化器 | `train/optimizer.py` | 官方 8+2 hybrid NS、gamma=0.18、AdamW eps=1e-20、单步更新 |
| loss-free bias + sequence balance loss | `train/losses.py` | bias 更新、balance loss 有限值 |
| Anticipatory Routing | `model/anticipatory.py`, `train/trainer.py` | 一步预取队列 + cache → replay；默认关 |
| 预训练 / SFT | `train/trainer.py`, `train/run.py` | 2 步 smoke、checkpoint 重载 logits 一致；batch-token warmup；flat→cosine LR；val_path；2K→4K→8K 分阶段 runner |
| GRPO / 多教师 OPD | `rl/` | rollout↔actor logprob、1 step、full-vocab reverse KL；FSDP2 多卡路径 |
| general specialist RLAIF | `rl/reward_model.py`, `configs/grpo/general.yaml` | `internlm2-1_8b-reward` + 长度/思考/重复 shaping |
| tool-use specialist | `rl/agent_rollout.py`, `rl/tools.py`, `configs/grpo/tool.yaml` | 多轮工具执行、observation mask、tool reward |
| 一键流水线 | `pipeline.py`, `configs/pipeline/` | tokenize→pretrain→SFT→GRPO→OPD→eval→export；按实际 token 自动算 step |
| router replay + 三道一致性门禁 | `model/routing.py`, `consistency/checks.py` | logprob Δ < 1e-3、routing 100%、ckpt 0 |
| 域条件路由指标 | `model/routing.py` | 按 domain 的专家负载/熵，GRPO 可开 |
| 权重版本 + logprob 门禁 | `rl/rollout.py` | vLLM 版本计数与 actor/rollout 对比 |
| 数据管线 | `data/` | 打包、mask、MinHash+段落去重、13-gram、fertility、manifest、license、加权混料、语言/vendored/math 过滤 |
| 训练监控与可视化 | `train/metrics.py`, `viz/` | tok/s、峰值显存、ETA、路由/负载/SwiGLU 指标；`report.html` 内联 SVG 图表 |
| 吞吐 / 显存基准 | `scripts/benchmark.py`, `train/benchmark.py` | 单卡与 FSDP2 的 step 时间、tok/s、峰值显存 |
| Scaling law | `scripts/scaling_law.py`, `train/scaling.py` | val loss 幂律拟合 α/R² + Chinchilla 参考线 |
| 评估 | `eval/` | math / code / world rubric 报告；可选 lm-eval general 基准 |
| world judge | `rl/judge.py` | 启发式默认；`world_judge: http` 走 OpenAI-compatible API |

测试：`210 passed`（CPU，debug model，约 40 秒；基础安装不含 `transformers` 时 HF 的 6 个测试自动跳过）。

### 模型自包含说明

`src/minideepseek/model/` 是参考实现（torchtitan `models/deepseek_v4`）的
**重构副本**：去掉了 torchtitan 的 config/parallel/sharding 框架，保留
V4 的数学与参数布局。state dict 的 key 与形状和参考实现一致，可直接对照。
来源、commit 与改动见 [doc/sources.md](doc/sources.md)。

---

## 2. 快速开始

环境：Python 3.11+；CPU 即可跑通 debug 模型，CUDA 上走 bf16。

```bash
# 1) 本地：环境 + debug 数据 + 门禁
./scripts/setup_env.sh
./scripts/debug_smoke.sh

# 2) 本地：从 minimind 生成平铺 JSONL（不修改 minimind/dataset）
bash scripts/build_data.sh

# 3) 本地：打包上传（只含 JSONL + manifests，不含 tokenization）
bash scripts/package_dataset.sh dist/minideepseek-data
```

服务器：

```bash
git clone <repo> miniDeepSeek && cd miniDeepSeek
DATA_REPO_URL=https://www.modelscope.cn/datasets/<ns>/<name>.git \
  bash scripts/server_setup.sh

# 先 dry-run 看完整命令与 token 预算
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh \
  configs/pipeline/minimind_145m.yaml --dry-run

# 一键训练：tokenize → 2K/4K/8K pretrain → SFT general/tool
# → GRPO general/tool → 2-teacher OPD → eval → HF export
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh \
  configs/pipeline/minimind_145m.yaml
```

单独运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m minideepseek.data.tokenize --smoke --seq-lens 2048 \
  --out-dir data/tokenization
.venv/bin/python -m minideepseek.train.smoke --flavor debugmodel --steps 2
.venv/bin/python -m minideepseek.consistency.check --flavor debugmodel --demo
```

数据流：`data/*.jsonl` 是上传/下载的唯一数据源；`data/tokenization/` 在服务器
生成，是中间产物，不打包。

## 3. 配置与入口

```text
configs/
├── model/{debugmodel,mini_220m_a80m,mini_145m_a60m,mini_60m_dense_d768}.yaml
├── pretrain/{debug_smoke,pilot_0.6b,full_6b,pretrain_145m,curriculum_145m_4x4090}.yaml
├── sft/{debug_smoke,general,tool,code}.yaml
├── grpo/{debug_math,general,tool,code}.yaml
├── opd/{debug_three_teacher,two_teacher_general_tool,three_teacher}.yaml
├── eval/{debug_math,general,tool,code}.yaml
└── pipeline/minimind_145m.yaml
```

`configs/model/*.yaml` 是完整的架构文件；训练配置用 `model_file` 选择它，
`model:` 下的字段只覆盖本次运行（上下文长度、MTP、精度等）。CLI 也可以直接覆盖：

```bash
# 临时换模型架构，不复制 training config
.venv/bin/python -m minideepseek.train.run \
  --config configs/pretrain/full_6b.yaml \
  --model-config configs/model/mini_145m_a60m.yaml
```

单机 4×4090/3090（24 GB）跑 145M + 2K→4K→8K 的完整流程见
[doc/server_runbook.md](doc/server_runbook.md)；显存/时长预算见
[doc/hardware_budget.md](doc/hardware_budget.md)。

```bash
# 数据（平铺 JSONL；manifest 见 data/manifests/specialists.yaml）
.venv/bin/python -m minideepseek.data.prepare pretrain --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare sft      --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare grpo     --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare opd      --input ... --output ...
.venv/bin/python -m minideepseek.data.prepare fertility --input ...
.venv/bin/python -m minideepseek.data.prepare contamination --train ... --eval ...
.venv/bin/python -m minideepseek.data.prepare manifest --manifest data/manifests/pretrain_6b.yaml
.venv/bin/python -m minideepseek.data.prepare curriculum --input ... --out-dir data/tokenization

# 训练 / RL
./scripts/train_pretrain.sh configs/pretrain/full_6b.yaml
./scripts/train_sft.sh      configs/sft/general.yaml
./scripts/train_grpo.sh     configs/grpo/general.yaml
./scripts/train_opd.sh      configs/opd/two_teacher_general_tool.yaml

# 一致性与评估
.venv/bin/python -m minideepseek.consistency.check \
  --flavor mini_220m_a80m --seq-len 4096 --checkpoint checkpoints/curriculum_145m/stage_02_seq8192/final \
  --prompts data/eval/prompts.jsonl
.venv/bin/python -m minideepseek.eval.run --config configs/eval/general.yaml

# 训练监控：自动刷新 report.html（YAML 里设 train.report_every: 50），或手动生成
.venv/bin/python scripts/plot_training.py checkpoints/pretrain_full/metrics.jsonl \
  --out checkpoints/pretrain_full/report.html --meta model=mini_220m_a80m

# 吞吐 / 显存基准（单卡；多卡见 doc/run_plan.md）
.venv/bin/python scripts/benchmark.py --config configs/pretrain/full_6b.yaml \
  --seq-len 4096 --batch-size 1 --steps 10 --json-out bench.json

# Scaling law 拟合（需要 ≥3 个 val 点）
.venv/bin/python scripts/scaling_law.py checkpoints/pretrain_full/metrics.jsonl \
  --params-m 216.65 --out checkpoints/pretrain_full/scaling_law.html

# HF 导出（可选 extra: `pip install -e '.[hf]'`）
.venv/bin/python scripts/export_hf.py \
  --checkpoint checkpoints/curriculum_145m/stage_02_seq8192/final --out hf/mini_220m_a80m \
  --flavor mini_220m_a80m --seq-len 4096 --max-context 8192

# verl 数据 + 配置导出（可选 CUDA extra: `pip install -e '.[verl]'`）
.venv/bin/python scripts/export_verl.py \
  --hf-model hf/mini_220m_a80m --train data/grpo/math.jsonl --out-dir runs/verl_math --rollout hf
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
├── model/        # 自包含 V4 模型、架构配置加载、PackedBatch、wrapper、路由、checkpoint
├── tokenizer/    # MiniMind 6,400 tokenizer + chat template + response mask
├── data/         # pretrain/SFT/GRPO/OPD 数据、去重、污染、fertility、manifest
├── train/        # Muon+AdamW、scheduler、metrics、losses、trainer、CLI
├── rl/           # torch rollout、reward、GRPO、多教师 OPD
├── consistency/  # logprob / routing / checkpoint 一致性门禁
└── eval/         # math / code / world 评估与报告
tests/            # 165 个 CPU 测试，全部使用 debug model
doc/              # decisions / sources / PROCESS / testing
scripts/          # 环境、数据、训练、评估脚本
```

---

## 5. 关键设计决定

- **只验证逻辑，不训练**：所有测试固定使用 `debugmodel`；三个目标模型只构建、统计参数量并做 CPU 1-step/forward-backward。
- **默认模型 `mini_220m_a80m`**：216.65M 总参 / 82.17M 激活（MTP depth 1、tied embeddings）。
- **8 routed + 0 shared，top-2，前 2 层 hash**；`route_norm=True`，`route_scale=1.5`。
- **`mini_60m_dense_d768` 的单一 expert 权重固定为 1.0**，等价于 dense FFN；无 hash / 无负载均衡。
- **`mini_145m_a60m`**：6 routed / top-2，active FFN 宽度与 dense 控制组对齐。
- **CSA(4) + HCA(32) + SWA**；`seq_len=4096`，`max_context=8192`；index top-k 由模型文件派生（4K→1024、8K→2048）。
- **SwiGLU clamp 10.0 常开**；**Anticipatory Routing 实现但默认关闭**。
- **4K dense-over-compressed → 8K sparse 的 index top-k 调度**：`IndexTopKSchedule`。
- **GRPO/OPD 从 checkpoint 初始化**：`train.resume` 分别加载 SFT specialist 与 base student。
- **checkpoint 主链**：torch state dict → safetensors → DCP；三者均已验证可回读。
- **bf16**：CUDA 上 wrapper 统一 autocast；权重保持 fp32 master。
- **长度课程**：`train/curriculum.py` 分阶段 2K→4K→8K，累计 step + resume；不是单次动态改 seq_len。
- **数据混料**：`data.sources` 权重混料；manifest 继续负责 license/epoch 校验。
- **AR**：预取下一 optimizer step 的 batch 并用当前 gate 缓存 routing；无异步 overlap。

---

## 6. 未完成 / 明确不在当前阶段

- 没有真实训练 `mini_220m_a80m` / `mini_145m_a60m` / `mini_60m_dense_d768`，也没有 0.6B/6B 数据产出；数据只用了 MiniMind 小样本。
- 没有多卡 CUDA 实机：FSDP2/DDP 的 2 进程正确性已在 CPU/Gloo 验证，
  NCCL、真实 bf16 吞吐、8 卡扩展性未验证。
- EP（专家并行）未实现：FSDP2 下 routed experts 是复制分片，尚未做
  all-to-all expert dispatch。
- vLLM 自定义模型/paged-attention kernel 未实现；verl 的 HF rollout 路径
  未在 GPU 上实跑。vLLM/verl 都只是可选集成。
- rollout 默认仍是 full-recompute torch（无 KV cache），适合一致性验证，不适合吞吐。
- world 的默认 judge 仍是启发式占位；HTTP judge 代码已接，但需要 API 与环境变量，
  未在 CI 中真实调用。
- 没有跑 lm-evaluation-harness（`eval/lm_eval.py` 适配已写，需可选 extra 与 GPU 实跑）。
- 没有 DSpark / 推测解码（第一版明确跳过）。
- Anticipatory Routing 有一步预取队列，但 probe 仍同步执行、没有异步 overlap；默认关闭。
- 去重、污染检查是单进程实现，6B 规模需要换成分布式版本。
- 分布式 GRPO/OPD 已有 FSDP2 路径（2 进程 CPU 验证）；vLLM/verl GPU 路径未验证。
- FIM/token-splitting、长文档/多语言数据、teacher hidden-state cache 未做。

逐项实现审计（哪些只是代码落地、哪些还没做）见
[doc/roadmap_audit.md](doc/roadmap_audit.md)。更多细节见
[doc/decisions.md](doc/decisions.md)、[doc/testing.md](doc/testing.md)、
[doc/PROCESS.md](doc/PROCESS.md)。
