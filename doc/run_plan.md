# 训练执行计划（Run Plan）

> 这份文档回答一个问题：**在真正开跑之前，我们凭什么相信这个配置跑得完、跑得对？**
> 原则：表里的数字必须来自实测（benchmark / smoke），不能拍脑袋；
> 没实测的格子写 `TBD`，不要填估计值冒充结果。

相关工具：

| 工具 | 用途 |
|---|---|
| `scripts/benchmark.py` | 单卡 / torchrun 多卡的 tok/s、step 时间、峰值显存 |
| `scripts/plot_training.py` | 把 `metrics.jsonl` 渲染成 HTML 报告（loss/lr/梯度/吞吐/显存/路由） |
| `scripts/scaling_law.py` | 从 val loss 拟合 `L = c·T^α`，标 Chinchilla 预算 |
| `src/minideepseek/train/trainer.py` | 训练中每 `report_every` 步自动刷新 `report.html` |

---

## 1. 三个档位与 token 预算

| 档位 | 总参 / 激活 | 主训长度 | Chinchilla 20× | 计划 token | 备注 |
|---|---|---|---|---|---|
| `mini_60m_dense_d768` | 60.88M / 60.88M | 4K | 1.2B | 2–3B | dense 对照，跑通优先 |
| `mini_145m_a60m` | 145.84M / 60.91M | 4K | 2.9B | 3–5B | MoE 主实验的轻量档 |
| `mini_220m_a80m`（默认） | 216.65M / 82.17M | 4K→8K | 4.3B | 6B | `configs/pretrain/full_6b.yaml` |

> Chinchilla 只是参考下界；V4 的方法论复现更看重"同一 token 预算下的消融"，
> 所以 60M 档的 2–3B 是**有意过训**，用来观察路由/注意力机制是否稳定。

时间估算（填实测后才有意义）：

```
每天 token = tokens_per_sec × 86400
预计天数   = 计划 token / 每天 token
其中 tokens_per_sec 取 benchmark 的稳定值（warmup 之后 ≥10 步）
```

---

## 2. 阶段门禁

每个阶段通过才进入下一阶段；失败时先回退配置，不要"带着问题往下跑"。

### 阶段 0a：debug 模型冒烟（CPU 即可）

```bash
./scripts/debug_smoke.sh
```

通过条件：

- `pytest` 全绿、`ruff` 干净；
- 2 步训练 loss 有限，checkpoint 重载后 logits 位级一致。

### 阶段 0b：真实数据、小步数（单卡）

```bash
python -m minideepseek.train.run --config configs/pretrain/full_6b.yaml --mode pretrain
```

先用 `steps: 100`、`batch_tokens: 8192` 的临时覆盖跑 100 步，看：

- loss 后 20 步均值比前 10 步低 ≥ 5%；
- `routing_entropy_normalized > 0.9`，`experts_used_fraction = 1.0`；
- `swiglu_gate_clamp_rate < 5%`；
- 无 NaN，`grad_norm` 没有持续 > 10× 中位数。

### 阶段 0c：目标硬件吞吐/显存校准

```bash
# 单卡
python scripts/benchmark.py --config configs/pretrain/full_6b.yaml \
    --seq-len 4096 --batch-size 1 --steps 10 --json-out bench_single.json

# 8 卡（FSDP2）
torchrun --nproc_per_node=8 scripts/benchmark.py \
    --config configs/pretrain/full_6b.yaml --distributed fsdp2 \
    --batch-size 1 --steps 10 --markdown-out doc/benchmark_results.md
```

通过条件：

- 实测 tok/s 与正式 run 的前 50 步差距 < 10%；
- 峰值显存 < 单卡容量的 90%（给 NCCL buffer 和碎片留余量）；
- 若 OOM：优先降 `seq_len`/batch，再考虑 activation checkpoint，**不要**先减 token 预算。

把实测值填进第 3 节的表，并提交 `doc/benchmark_results.md`。

### 阶段 1：0.5–1B token 试点

- 至少 8 个 val 点（`val_every` ≤ 总步数的 1/10）；
- 用 `scripts/scaling_law.py` 拟合，`R² > 0.95`；
- 若 `α > -0.02`，说明还在数据/学习率主导区，先别外推；
- 路由指标没有单调恶化（熵不持续下降、专家负载比不持续上升）。

### 阶段 2：主训练 6B token

- 4K 主训；末期 15–20% tokens 切 8K（`configs/pretrain/length_curriculum.yaml`）；
- 每 500 步留 checkpoint，`keep_last_ckpt: 8`，val 最优单独存 `best/`；
- 出现 NaN 时训练会写 `nan_step_*/nan_report.json` 后退出：先看最近 10 条 metrics 和
  当时的 `grad_norm`/`loss_spike`，再决定回滚还是调 AR。

### 阶段 3：后训练前检查

- 用 SFT 模型对同一 prompt 的 4 个候选打分，测 top-1 / pairwise accuracy；
  接近随机（top-1 ≈ 25%、pairwise ≈ 50%）时不要投 GRPO/OPD，先做 weighted SFT；
- GRPO 开跑前，`rollout ↔ actor` 的 logprob 一致性门禁必须通过。

---

## 3. 实测记录表（由 benchmark 填写）

> 跑完 `scripts/benchmark.py` 后把对应行替换成实测值；没有实测的写 `TBD`。

| 档位 | 设备 | 并行 | seq | batch | step ms | tok/s | 峰值显存 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 60m | 1×GPU | none | 4096 | 1 | TBD | TBD | TBD | 未测 |
| 145m | 1×GPU | none | 4096 | 1 | TBD | TBD | TBD | 未测 |
| 220m | 1×GPU | none | 4096 | 1 | TBD | TBD | TBD | 未测 |
| 220m | 8×GPU | fsdp2 | 4096 | 1 | TBD | TBD | TBD | 未测 |

---

## 4. 训练中盯哪些指标

`metrics.jsonl` / `report.html` 里按重要性排序：

| 指标 | 健康范围 | 异常时先看 |
|---|---|---|
| `loss` / `loss_ema` | 平滑下降 | 数据顺序、lr、是否 NaN |
| `val_loss` / `val_ppl` | 与 train 同步下降 | 过拟合、val 集污染 |
| `grad_norm` | 无持续尖峰 | loss spike、AR、clamp |
| `tokens_per_sec` | 与 benchmark ±10% | 数据加载、通信、碎片 |
| `mem_peak_mb` | < 90% 容量 | batch/seq、checkpoint 频率 |
| `expert_load_max_over_mean` | < 1.5 | 路由坍缩、bias 更新 |
| `routing_entropy_normalized` | > 0.9 | 同上 |
| `swiglu_gate_clamp_rate` | < 5% | 初始化、lr |
| `mtp_loss_d1` | 与 main 同步 | MTP 权重/学习率 |
| `indexer_loss` | 缓慢下降 | indexer 预热、top-k 调度 |

刷新可视化：

```bash
python scripts/plot_training.py checkpoints/pretrain_full/metrics.jsonl \
    --out checkpoints/pretrain_full/report.html \
    --meta model=mini_220m_a80m --meta run=pretrain_full
```

或在 YAML 里设 `train.report_every: 50`，训练过程中自动刷新。

---

## 5. 风险与回退

| 风险 | 触发信号 | 回退动作 |
|---|---|---|
| 显存不足 | benchmark OOM / >90% | 降 batch 或 seq；开 activation checkpoint |
| loss spike | `loss_spike > 0` 连续出现 | 开 AR / 回滚快照；降 lr |
| 路由坍缩 | 负载比 > 2、熵 < 0.8 | 提高 balance 系数、检查 bias 更新 |
| 吞吐不达标 | tok/s 低于 benchmark 10% | 查 dataloader、通信、碎片 |
| 后训练无信号 | 候选排序 ≈ 随机 | 停 GRPO/OPD，改 weighted SFT |
| NaN | 训练退出并留下 `nan_step_*` | 从最近 checkpoint 恢复，缩短 ckpt 间隔 |
