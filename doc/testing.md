# 测试与验收

> 当前阶段只验证**代码逻辑**。所有模型测试固定使用 torchtitan 的
> `deepseek_v4_debugmodel`（`debugmodel` / `mtp` 变体），不训练 mini-60M。

## 运行

```bash
./scripts/setup_env.sh          # 一次性
.venv/bin/python -m pytest -q   # 93 passed, ~20s on CPU
.venv/bin/ruff check src tests  # All checks passed
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
| `test_model.py` | forward/backward、MTP 形状、loss、logprob、greedy 生成、参数报告、mini 配置结构、SwiGLU clamp 监控 |
| `test_routing.py` | hash/learned router 录制、replay 一致率、指标、gate 快照、hash 表持久化 |
| `test_checkpoint.py` | torch / safetensors / DCP 三种存取、logits 位级一致、RNG 状态、指纹 |
| `test_optimizer.py` | Newton-Schulz、Muon 单步、混合参数分组、warmup/cosine |
| `test_trainer.py` | 2 步预训练、梯度累积、SFT、expert bias、AR 训练、checkpoint |
| `test_data.py` | pretrain 打包、MinHash 去重、13-gram 污染、质量过滤、fertility、manifest、GRPO/OPD 转换 |
| `test_grpo.py` | math/code/world/medical reward、advantage、1 步 GRPO、ratio↔logprob 一致性、SFT checkpoint 初始化 |
| `test_opd.py` | 1 步单/多教师 OPD、teacher 不更新、teacher/base checkpoint 加载 |
| `test_consistency.py` | logprob / routing / checkpoint 三道门禁 |
| `test_anticipatory.py` | stale gate probe、replay、状态环形缓冲、closure |
| `test_spike.py` | spike 检测、rollback 快照、自动 AR、index top-k 调度 |
| `test_eval.py` | math/world 评估、报告落盘 |
| `test_config.py` | 全部 YAML 可加载、未知键拒绝 |

## 关键验收数值（本机实测）

- debugmodel：无 MTP 10.60M；MTP depth 1 12.98M。
- mini 配置参数：`mini_60m` 65.26M、`mini_110m` 120.65M、`mini_200m` 218.21M、
  `mini_60m_dense` 62.52M。
- 一致性（float32 CPU）：
  - rollout ↔ teacher-forced logprob：max |Δ| ≈ **9.5e-7**（阈值 1e-3）；
  - router replay：top-k 集合一致率 **100%**（阈值 99.9%）；
  - checkpoint 重载 logits：max |Δ| = **0**。
- smoke：2 步 debug 训练 + checkpoint 回读，`checkpoint_max_abs_diff = 0.0`。
- GRPO：1 步，`approx_kl ≈ 1e-6`（rollout/actor 同温度），ratio 均值 ≈ 1。
- OPD：2 teacher，KL 分量均 > 0，teacher 参数逐位不变。

## 已知警告

- `compressor.py:144 UserWarning: In CPU autocast, but the target dtype is not supported.`
  来自上游在 CPU 上尝试 fp32 autocast；torch 会禁用该 autocast，数值仍是 float32，
  测试全部通过。这不是本仓库的代码。
- `torch.distributed is disabled... assuming single process`：DCP 单进程保存/读取。

## 未覆盖 / 未验证

- 多 GPU、FSDP2、EP、TP、CP 路径；本机只有 CPU。
- 真实 bf16 数值稳定性；测试用 float32。
- vLLM / verl rollout、权重同步、router replay 的 kernel 级一致性。
- 6B 数据规模下的吞吐、去重与污染检查性能。
- 训练收敛性 / benchmark 分数（当前阶段明确不追求）。
