# 测试与验收

> 当前阶段只验证**代码逻辑**。所有模型测试固定使用项目自带的
> `debugmodel`（与 DeepSeek-V4 debug 配置同构；`mtp` 变体），不训练 mini-60M。

## 运行

```bash
./scripts/setup_env.sh          # 一次性
.venv/bin/python -m pytest -q   # 165 passed, ~29s on CPU；无 transformers 时 HF 的 6 个测试跳过
.venv/bin/ruff check src tests  # All checks passed
```

额外验证：

```bash
.venv/bin/python scripts/test_distributed.py --world-size 2 --mode fsdp2
.venv/bin/python scripts/test_rl_distributed.py --method grpo
.venv/bin/python scripts/test_rl_distributed.py --method opd
.venv/bin/python -m minideepseek.data.prepare curriculum \
  --input /tmp/corpus.jsonl --out-dir /tmp/tokenized --seq-lens 2048 4096 8192
```

端到端门禁（训练 smoke + 一致性 + 测试）：

```bash
./scripts/debug_smoke.sh
```

## 覆盖矩阵

| 测试文件 | 覆盖内容 |
|---|---|
| `test_batch.py` | 文档打包、positions、target mask、MTP shift、跨文档边界、padded→packed |
| `test_tokenizer.py` | 6,400 词表、特殊 token、chat template、response-only mask、多轮 |
| `test_model.py` | forward/backward、MTP 形状、loss、logprob、greedy 生成、参数报告、mini 配置结构、五档参数校准、SwiGLU clamp 监控 |
| `test_routing.py` | hash/learned router 录制、replay 一致率、指标、gate 快照、hash 表持久化 |
| `test_checkpoint.py` | torch / safetensors / DCP 三种存取、logits 位级一致、RNG 状态、指纹 |
| `test_optimizer.py` | Newton-Schulz、Muon 单步、混合参数分组、warmup/cosine |
| `test_trainer.py` | 2 步预训练、梯度累积、SFT、expert bias、AR 训练、checkpoint |
| `test_data.py` | pretrain 打包、MinHash 去重、13-gram 污染、质量过滤、fertility、manifest、GRPO/OPD 转换、batch-token warmup |
| `test_grpo.py` | math/code/world/medical reward、advantage、1 步 GRPO、ratio↔logprob 一致性、SFT checkpoint 初始化 |
| `test_opd.py` | 1 步单/多教师 OPD、teacher 不更新、teacher/base checkpoint 加载 |
| `test_consistency.py` | logprob / routing / checkpoint 三道门禁 |
| `test_anticipatory.py` | stale gate probe、replay、状态环形缓冲、closure |
| `test_spike.py` | spike 检测、rollback 快照、自动 AR、index top-k 调度 |
| `test_eval.py` | math/world 评估、报告落盘 |
| `test_config.py` | 全部 YAML 可加载、未知键拒绝 |
| `test_distributed.py` | 单进程 FSDP2 wrap、forward/backward、optimizer、state/gather、trainer 保存/恢复 |
| `test_hf.py` | HF config 往返、key prefix 映射、export/load logits 一致、greedy generate、trust_remote_code 加载 |
| `test_integrations.py` | verl 数据 schema、reward entrypoint、verl GRPO/OPD config、torch backend、vLLM/lm-eval 缺依赖守卫、HTTP judge 解析 |
| `test_independence.py` | 源码无 torchtitan/minimind 等 import、无 sys.path 注入、无 workspace checkout 时仍可 import/build |
| `test_curriculum.py` | 2K→4K→8K stage 校验、配置覆盖 |
| `test_mix.py` | 加权 batch 配额、确定性、source 重启 |
| `test_rl_distributed.py` | FSDP2 GRPO/OPD world_size=1 单步、保存 |

## 关键验收数值（本机实测）

- debugmodel：无 MTP 10.60M；MTP depth 1 12.98M（12,984,104）。
- mini 配置参数（含 MTP depth 1、tied embeddings）：`mini_60m` 65,262,922、
  `mini_110m` 120,652,773、`mini_200m` 218,205,261、`mini_60m_dense` 62,526,593。
- 一致性（float32 CPU）：
  - rollout ↔ teacher-forced logprob：max |Δ| ≈ **9.5e-7**（阈值 1e-3）；
  - router replay：top-k 集合一致率 **100%**（阈值 99.9%）；
  - checkpoint 重载 logits：max |Δ| = **0**。
- smoke：2 步 debug 训练 + checkpoint 回读，`checkpoint_max_abs_diff = 0.0`。
- GRPO：1 步，`approx_kl ≈ 1e-6`（rollout/actor 同温度），ratio 均值 ≈ 1。
- OPD：2 teacher，KL 分量均 > 0，teacher 参数逐位不变。
- FSDP2 / DDP：`scripts/test_distributed.py --world-size 1|2 --steps 1`
  在相同 seed/batch 下，world=1 与 world=2 的 `model.pt` sha256 逐位一致
  （说明 DTensor 版 Muon、global grad-norm clip、expert-bias 同步都正确）。
- HF：export → `from_pretrained` 的 logits 与训练模型逐位一致（max |Δ| = 0）；
  `trust_remote_code=True` 可加载 config/model/tokenizer，greedy generate 可跑。
- verl：数据 schema、reward 返回值、Hydra override 生成均单测覆盖；
  未安装 verl 也能 import 本项目。

## 与参考实现的数值对齐（开发用）

`scripts/verify_reference.py` 需要本地 `../torchtitan`，不进默认测试：

```text
seq=320, packed 2 docs, debugmodel + MTP depth 1
layer 0: max rel diff 1.5e-4
layer 1: max rel diff 1.1e-4
layer 2: max rel diff 7.8e-5
layer 3: max rel diff 8.3e-5
logits depth 0: mean abs 2.3e-4, max abs 8.3e-2
logits depth 1: mean abs 2.6e-4, max abs 8.2e-2
```

逐层误差来自参考实现把 routed expert 输入 cast 到 bf16；logits 的 max abs
由极少数 token 的离散路由翻转造成，mean abs 仍是 2e-4 量级。

## 已知警告

- `torch.distributed is disabled... assuming single process`：DCP 单进程保存/读取。

## 未覆盖 / 未验证

- 多卡 CUDA 实机：FSDP2/DDP 只在 CPU/Gloo 两进程验证；NCCL、bf16 数值、
  8 卡吞吐、EP/TP/CP、激活重计算未验证。
- EP（专家并行）：当前 FSDP2 复制专家分片，没有 all-to-all dispatch。
- vLLM：未安装、未实跑；自定义稀疏注意力的 paged-attention kernel 不存在，
  只提供保守的 HF 导出 + 采样 + 重建式权重同步。
- verl：未安装、未实跑；HF rollout + FSDP actor 路径按 verl 当前接口编写，
  需要在 CUDA 集群上做一次 smoke。
- 6B 数据规模下的吞吐、去重与污染检查性能。
- 训练收敛性 / benchmark 分数（当前阶段明确不追求）。
