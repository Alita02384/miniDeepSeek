# 代码来源与改动

> 规则见 `AGENTS.md` §3.3：保留版权头、只复制最小闭包、记录来源 commit 与改动。
> 本仓库**不修改**任何第三方 checkout，**运行时不 import** 它们。
> 第三方目录只作为参考实现；需要的代码已经重构进 `src/minideepseek/`。

## 1. torchtitan（BSD-3-Clause）

- 来源：`../torchtitan`，checkout commit
  `01586df70d7599d59a87663f39670a8c694924c3`。
- 用途：DeepSeek-V4 的结构、数学、参数布局与 debug 配置对照。

### 复制/重构的文件

torchtitan 的 V4 实现依赖 `models/common`、`config`、`distributed`、
`spmd_types` 等约 1.4 万行框架代码。我们把**模型数学与参数布局**重构为
普通 `nn.Module`，去掉 sharding / SPMD / config 框架，而不是整目录搬入。

| 本仓库文件 | 上游参考 | 改动 |
|---|---|---|
| `src/minideepseek/model/config.py` | `deepseek_v4/flavors.py` | 用 dataclass 重写架构描述；模型数值外置到 `configs/model/*.yaml`，本文件只保留校验、派生（压缩层/MTP/index top-k）与构建 |
| `src/minideepseek/model/nn.py` | `models/common/{activation,nn_modules,linear}.py` | `RMSNorm`、`LimitedSwiGLU`、初始化、Hadamard、SwiGLU clamp |
| `src/minideepseek/model/rope.py` | `models/common/rope.py` | 实数版 adjacent-pair RoPE + YaRN，缓存为 buffer |
| `src/minideepseek/model/mhc.py` | `deepseek_v4/mhc.py` | 保留 HcSplitSinkhorn / HcPre / HcPost / HcHead 数学与参数名 |
| `src/minideepseek/model/moe.py` | `deepseek_v4/moe.py`, `models/common/moe/*` | 自写 hash/learned router、grouped expert 参数（key 与上游一致）、shared expert、loss-free bias |
| `src/minideepseek/model/compressor.py` | `deepseek_v4/compressor.py` | Compressor / Indexer / SparseIndexerLoss；用 torch 重写 reference top-k 选择，去掉 fused kernel 依赖 |
| `src/minideepseek/model/attention.py` | `deepseek_v4/attention.py`, `models/common/attn_gym` | SWA/CSA/HCA + gather attention + sink；`gather_attn` 参考实现按原数学重写 |
| `src/minideepseek/model/model.py` | `deepseek_v4/model.py`, `deepseek_v3/mtp.py` | Decoder、TransformerBlock、MTPBlock、MTP roll；去掉 PP/CP/SPMD |
| `src/minideepseek/model/routing.py` | — | 本项目新增：record/replay、gate snapshot、路由指标 |
| `src/minideepseek/model/batch.py` | — | 本项目新增：多文档 packed batch 与 MTP target shift |
| `src/minideepseek/model/wrapper.py` | — | 本项目新增：loss/logprob/generate/参数统计 |

改动原则：

1. 保留上游版权头，并在文件头注明“refactored from …”。
2. state dict 的 key 与形状对齐上游（例如 `layers.0.feed_forward.routed_experts.w13.weight`、
   `attention.compressor_128.*`、`hc_attn_pre.hc_fn`），便于对照与转换。
3. 去掉 `spmd_types` / `torch_remat` / sharding / TP/CP/EP 的运行时代码；
   多卡并行在本项目自己的 `train/distributed.py` 中实现（FSDP2）。
4. 参考实现当前在 CPU 上仍会把 routed expert 输入 cast 到 bf16；本项目
   在 fp32 下不强制 cast，因此逐层对比的相对误差约 1e-4（见
   `scripts/verify_reference.py`）。
5. 未改动 `torchtitan/` 下任何文件。

### 数值对齐

`scripts/verify_reference.py`（需要本地存在 `../torchtitan`，仅开发用）
构建同一份 debug 配置、拷入同一 state dict，逐层比较 block 输出：

```text
seq=48 : rel diff  layer0 1.4e-4 / layer1 1.1e-4 / layer2 8.5e-5 / layer3 7.5e-5
seq=512: rel diff  layer0 1.5e-4 / layer1 1.0e-4 / layer2 8.1e-5 / layer3 7.9e-5
```

差异全部来自上游对 expert 输入的 bf16 cast（fp32 下无法消除）。

## 2. minimind（Apache-2.0）

| 本仓库文件 | 上游文件 | 说明 |
|---|---|---|
| `tokenizer/tokenizer.json` | `minimind/model/tokenizer.json` | 6,400 词表，36 个 added tokens；未修改；sha256 `71f32c68…a668` |
| `tokenizer/tokenizer_config.json` | `minimind/model/tokenizer_config.json` | chat template 原样复制；未修改 |

minimind commit：`233d00fa068ec2b8c0d4f07b703934d6391a5a73`。

数据样本（`pretrain_t2t_mini.jsonl` / `sft_t2t_mini.jsonl` / `agent_rl_math.jsonl`）
只在本地数据准备脚本中读取，**不复制进 git**。

## 3. verl（Apache-2.0）与 DeepSeek-V4-Flash（MIT）

- verl：阅读 `docs/algo/opd.md`、`examples/on_policy_distillation_trainer/`、
  `utils/veomni/router_replay.py` 的设计，作为 `rl/opd.py`、`rl/rollout.py`、
  `model/routing.py` 的算法参考；未复制运行时代码。
- DeepSeek-V4-Flash：阅读 `config.json`、`README.md`、inference 结构，用于
  对照超参与 chat 格式；没有复制 `inference/model.py`。
- vLLM / verl 只作为**可选 extra**（`pyproject.toml`），不装也能运行本项目；
  详见 `doc/decisions.md` 的 D11/D15/D24/D25。

## 4. 本项目新增（非复制）

| 文件 | 说明 |
|---|---|
| `src/minideepseek/model/hf/` | HF `PretrainedConfig`/`PreTrainedModel` 适配、训练↔HF key 映射、导出；参考 transformers 文档与 `HFRollout` 的 `generate` 契约，未复制 verl/vLLM 代码 |
| `src/minideepseek/train/distributed.py` | `torchrun` 启动、FSDP2 `fully_shard`、manual DDP、global grad clip、state gather；按 PyTorch FSDP2 文档实现 |
| `src/minideepseek/rl/backends/vllm.py` | vLLM `LLM` 采样封装（可选、CUDA-only）；接口参考 verl `vllm_rollout` 的 `generate_sequences` 语义 |
| `src/minideepseek/rl/verl.py` | verl 数据 schema、reward entrypoint、Hydra config；参考 verl `main_ppo` 配置与 `custom_reward_function` 接口 |
| `scripts/export_hf.py`, `scripts/export_verl.py`, `scripts/test_distributed.py`, `scripts/verify_reference.py` | 导出与验证脚本 |
| `src/minideepseek/eval/lm_eval.py` | lm-evaluation-harness 适配（可选依赖），未复制其代码 |
| `src/minideepseek/rl/judge.py` | OpenAI-compatible world judge，未复制第三方实现 |

这些文件不复制上游实现；verl/vLLM 的运行时依赖是可选项，真正执行前需在
CUDA 机器上按 `doc/testing.md` 的未验证项补 smoke。

## 5. 复现来源

```bash
git -C ../torchtitan rev-parse HEAD      # 01586df70...
git -C ../minimind rev-parse HEAD        # 233d00f...
shasum -a 256 tokenizer/tokenizer.json   # 71f32c68...
```

参考实现升级时，先跑 `pytest tests/test_model.py` 与本文件的数值对齐脚本，
再核对 state dict key/shape 是否变化。

## 6. MiniMind tool / reward 组件（2026-11-07）

| 本项目文件 | 来源 | 改动 |
|---|---|---|
| `src/minideepseek/rl/tools.py` | `minimind/trainer/train_agent.py` 的 `TOOLS`/`MOCK_RESULTS`/`CHECK_ARGS`/`parse_tool_calls`/`execute_tool`/`safe_math_eval` | 拆成独立模块；去掉训练脚本全局变量；保留同样的 mock 结果 |
| `src/minideepseek/rl/agent_rollout.py` | `minimind/trainer/train_agent.py` 的 `rollout_single`/`rollout_batch` | 适配本项目 `RolloutSample`/token mask；用 chat template 前缀一致性校验替代 marker hack；observation 不入 loss |
| `src/minideepseek/rl/reward_model.py` | `minimind/trainer/trainer_utils.py::LMForRewardModel` | 去掉全局依赖；`transformers` 改为运行时可选导入；device 自动选择 |
| `src/minideepseek/rl/reward.py` tool/general + shaping | `minimind/trainer/train_grpo.py`、`train_agent.py` 的 `calculate_rewards`/`rep_penalty` | 保留权重与 clip；接入本项目 `RewardSettings`/`GRPORecord`；math/code 原有可验证 reward 不替换 |

数据来源：`minimind/dataset/{rlaif.jsonl, agent_rl.jsonl, agent_rl_math.jsonl,
sft_t2t_mini.jsonl, sft_t2t.jsonl, pretrain_t2t.jsonl}`；许可见 MiniMind 数据集卡
（pretrain/SFT 为 CC-BY-NC-4.0，agent/rlaif 为混合来源，接入前核对上游）。

## 7. 可选 fused attention：attn_gym（2026-11-09）

| 本项目 | 来源 | 说明 |
|---|---|---|
| `src/minideepseek/model/attention.py::gather_attention` | `attn_gym.sparse.gather_attn`（Attention Gym，BSD-3-Clause，PyPI `attn_gym>=0.0.17`） | **不复制源码**，通过 optional extra `kernels` 运行时导入；入参签名与 torchtitan `deepseek_v4/attention.py::_gather_attn` 一致 |
| 集成方式参考 | torchtitan `torchtitan/models/deepseek_v4/attention.py`（BSD-3） | torchtitan 的 V4 参考实现直接调用 attn_gym；本项目保留 torch 参考实现作为 CPU/缺包回退 |

- `MINIDEEPSEEK_ATTN_BACKEND=auto|torch|attn_gym` 控制后端；`auto` 在 CUDA 且有 attn_gym 时启用 fused。
- 未复制任何 attn_gym 源码；`pyproject.toml` 的 `kernels` extra 只是可选依赖。

## 8. 增量 decode / compressor 状态机（2026-11-11）

| 本项目 | 来源 | 改动 |
|---|---|---|
| `src/minideepseek/model/compressor.py::project_tokens/capture_state/decode_step` | `DeepSeek-V4-Flash/inference/model.py` 的 `Compressor.forward`/`kv_state`/`score_state`（MIT） | 只移植**状态布局与 overlap 语义**：`[2R, 2D]` 状态、完成 block 时 `cat([state[:R,:D], state[R:,D:]])` 加权、完成后左移。改成批处理接口（`[B, ...]`），并接到本项目的 `Compressor`/`Indexer` 参数命名。 |
| `src/minideepseek/model/cache.py` | 本项目原创 | slot 寻址的 ring KV + compressed KV + indexer K；`copy_slot` 做 group 分叉。布局参考 official `inference/model.py` 的 `kv_cache`/`kv_state`，未复制代码。 |
| `src/minideepseek/model/attention.py::DeepSeekV4Attention.decode`、`model.py::decode` | 本项目原创 | 与训练前向同一套投影/RoPE/mHC/MoE；attention 用 local window + top-k compressed + sink 的显式实现。 |
| `src/minideepseek/rl/cached_rollout.py`、`graph_decode.py` | 本项目原创 | 批量 prefill、group 分叉、活跃 slot 压缩、CUDA graph 分桶。设计参考 verl 的 rollout/weight-sync 分层与 vLLM 的 paged decode 思路，但**不 import、不复制**二者代码。 |

- 未引入 vLLM/verl 运行时依赖；`rollout_backend=vllm` 仍是可选路径（D82 说明当前模型不可用）。
- 数值门禁：`tests/test_cache.py` 在 fp32 CPU 上逐 token 对齐全量前向
  （max|Δ|≈2e-6）；CUDA bf16 的 4e-2 差值按 D83 处理。
