# 24GB 消费卡（3090 / 4090）训练预算

> 结论先行：**三档里只有 60m / 145m 能在 24GB 上按 4K 直接开训；220m 在 4K
> 一定会 OOM，加到几张卡都没用**（激活不随卡数切分）。加卡的唯一作用是缩短
> 时间，不能换来更长的上下文。
>
> 本文的显存数字来自 `scripts/measure_activation_budget.py` 在 CPU 上用
> bf16 autocast 实测的 saved-tensor 字节，再按 `T × top-k` 外推；时间数字是按
> 实测 FLOPs/token 加 MFU 假设算的**估算**。二者都要在目标卡上用同一脚本
> 复核（见 §6）。

---

## 0. 速查

| 配置 | 总参 / 激活 | 4K 一个 pack 的峰值显存 | 24GB 4K 能跑 | 24GB 2K 能跑 | 最少卡数 | 建议卡数 | 6B tokens（8×4090） |
|---|---|---|---|---|---|---|---|
| `mini_60m_dense_d768` | 60.9M / 60.9M | ~10.6 GiB | ✅ 1 张 | ✅ 1 张 | 1 | 2–4 | 4.5–9 h |
| `mini_145m_a60m` | 145.8M / 60.9M | ~12.3 GiB | ✅ 1 张 | ✅ 1 张 | 1 | 4 | 4.5–9 h |
| `mini_220m_a80m` | 216.7M / 82.2M | ~27.3 GiB | ❌ 需激活重算 | ✅ 1 张 | 1（@2K） | 8 | 7.5–15 h（@2K） |
| 任一配置 @8K 课程 | — | 27–78 GiB | ❌ | ❌ | — | — | 需 chunked attention |

“6B tokens” = `configs/pretrain/full_6b.yaml` 的 12000 step × 0.5M tokens。
时间区间对应 MFU 15%（保守，当前 torch 版 gather attention）到 30%（理想）。

---

## 1. 显存口径

单卡峰值 = **参数状态 + 激活 + 余量**。

### 1.1 参数状态（不随 seq_len 变）

优化器是 Muon + AdamW 混合：Muon 只存一份 momentum，AdamW 只覆盖 1D 参数
（norm / bias，量级可忽略）。训练时权重与梯度都是 fp32：

| 配置 | 权重 | 梯度 | Muon momentum | 合计（单卡 / 8 卡 FSDP2） |
|---|---|---|---|---|
| 60m | 0.23 GiB | 0.23 | 0.23 | 0.68 / 0.09 GiB |
| 145m | 0.54 | 0.54 | 0.54 | 1.63 / 0.20 |
| 220m | 0.81 | 0.81 | 0.81 | 2.42 / 0.30 |

FSDP2（`distributed.mode=fsdp2`）会把这三份按卡数切分，所以卡数对这部分有效；
但**激活是每个 rank 私有的，切不动**。

### 1.2 激活：gather attention 是唯一的显存大头

`model/attention.py::gather_attention` 是 correctness-first 的 torch 实现：
它把每个 query 选中的 KV **实体化**成 `[T, window + top-k, head_dim]`，再在
fp32 下做 softmax。autocast 下这条路径会留下两份 bf16 拷贝（两次 matmul 各
一次 cast），所以：

```
attention 显存 ≈ 2 × T × Σ_layers K_l × head_dim × 2 B
K_l = window(128) + 选中条目
    CSA(4)  : min(index_topk, T/4)
    HCA(32) : T/32
    SWA(0) / MTP : 0
```

`index_topk` 由模型文件派生（4K→1024、8K→2048），训练前 95% 的 step 都是
满 top-k，只有最后 5% 才衰减到 64（`IndexTopKSchedule`），**峰值出现在训练
开始时**，不能按收尾的 64 来估。

`head_dim` 是决定性变量：220m 是 16×256，60m/145m 是 12×64，同样的
`T × K` 下 220m 的 attention 激活是另外两档的 **4 倍**。

### 1.3 实测（`scripts/measure_activation_budget.py`，1 doc，bf16 autocast）

| 配置 | seq=1024 | 每 token | 其中 attention | 其余（FFN/MTP/mHC/logits） |
|---|---|---|---|---|
| 60m | 1.87 GiB | 1.866 MiB | 0.45 | 1.41 |
| 145m | 2.07 GiB | 2.069 MiB | 0.45 | 1.62 |
| 220m | 3.79 GiB | 3.791 MiB | 1.81 | 1.98 |

“其余”按 token 线性；“attention”按 `T × K` 增长。用这两项外推：

| seq_len | ΣK（含 MTP） | 60m | 145m | 220m |
|---|---|---|---|---|
| 2048 | 2688 | 4.1 GiB | 4.5 | 9.2 |
| 4096 | 4352 | 9.9 | 10.7 | 24.9 |
| 8192 | 7680 | 26.3 | 27.9 | 75.8 |

加上 §1.1 的参数状态（8 卡时几乎为 0），就是 §0 的峰值列。

> 注意：一个 pack 至少是 `seq_len` 个 token（`iter_batches` 里
> `size = max(1, batch_tokens // seq_len)`），所以 4K 训练时**每卡每个
> micro-batch 的下限就是 4096 token**，不能靠调小 batch 来省显存。

---

## 2. 吞吐与时长

### 2.1 每 token 的计算量（实测 + 外推）

`FlopCounterMode` 只统计 matmul；实测 forward：

| 配置 | @1024 | @4096（含更大 top-k） | 训练（≈3×forward） |
|---|---|---|---|
| 60m | 143.7 MFLOP | ~152 | **~0.45 GFLOP** |
| 145m | 143.8 | ~152 | **~0.45 GFLOP** |
| 220m | 216.1 | ~257 | **~0.77 GFLOP** |

60m 与 145m 几乎一样，是因为两者激活参数相同（60.9M），145m 多出来的
专家只是“存着不用”。

### 2.2 时长

`每卡 tok/s = 峰值 FLOPS × MFU / 每 token FLOPs`；4090 bf16 峰值 82.6 TFLOPS，
3090 为 71 TFLOPS（约慢 14%）。多卡按 95%/卡缩放（FSDP2 all-gather 在 PCIe 上
对本模型可忽略）。

| 配置 | 1×4090 | 4×4090 | 8×4090 | 8×3090 |
|---|---|---|---|---|
| 60m / 145m，6B tokens | 48 h | 12.5 h | 6.6 h | 7.7 h |
| 220m，6B tokens | 81 h | 21 h | 11 h | 13 h |

MFU 取 20% 为例；区间见 §0。参考点：1B tokens ≈ 上表 / 6.3，0.6B pilot
≈ 上表 / 10。

MFU 为什么不高：模型小、`gather_attention` 是逐 token gather + fp32 softmax，
并且没有 flash-attention / 激活重算；220m 在 4K 时 attention 的显存读写
（~10 MB/token）已经接近 4090 的带宽上限。

---

## 3. 卡数与硬件选择

- **60m**：1 张就能跑 4K；4 张是性价比拐点（12 h 量级）。
- **145m**：单卡 4K 约 12 GiB，能跑；建议 4 张。
- **220m**：24GB 只能在 2K 上单卡跑；要跑计划里的 4K 主训练，必须先加
  **按 block 的激活重算**（见 §4），之后 4K 峰值约 12–14 GiB，8 张是合理配置。
- 3090 vs 4090：显存同为 24GB、都支持 bf16；4090 每卡快约 14%，功耗更低。
  **不要混插**——FSDP 同步按最慢的卡走。两张以上都走 PCIe，无 NVLink，
  本模型的 all-gather 量（0.4 GB/step）不足以成为瓶颈。

`full_6b.yaml` 的 `batch_tokens_final: 65536` 是**全局**的；8 卡时每 rank 会拿到
8192 token = 2 个 4K pack，激活直接翻倍（60m→20.5 GiB，145m→23 GiB，贴边）。
24GB 卡上建议把 `batch_tokens(_final)` 设成 `4096 × 卡数`，用 `grad_accum`
补回每 step 的 token 数。

---

## 4. 阻塞项与修复优先级

| 问题 | 影响 | 修复 | 成本 |
|---|---|---|---|
| 无激活重算（`torch.utils.checkpoint` 未接） | 220m@4K OOM；所有配置@8K OOM | 每个 transformer block 包一层 `checkpoint(..., use_reentrant=False)` | 小（~1 天），训练变慢 25–35% |
| `gather_attention` 实体化 `[T,K,D]` | 8K 即使重算也放不下（220m 单层 ~18 GiB） | 按 token 分块 attention（chunk 512–1024），或接 flash-attn 的 varlen kernel | 中 |
| 无 CP/SP | 激活无法跨卡切分 | 上下文并行 / 序列并行 | 大 |
| 8K 阶段 `index_topk=2048` | K 翻倍，显存 ~2× | 降低 top-k 或改用分块 | 配置级 |

优先级：**先加激活重算**（解锁 220m@4K、60m/145m@8K 的大部分场景），再考虑
chunked attention（解锁 220m@8K）。

---

## 5. 复核命令

上卡第一件事（比跑长任务便宜得多）：

```bash
# 目标卡上测真实 allocator 峰值；OOM 就先降 seq_len
.venv/bin/python scripts/measure_activation_budget.py --device cuda \
    --flavor mini_220m_a80m --seq-len 2048 4096

# 真实一步的 tok/s 与峰值（走 trainer 同一条路径）
torchrun --nproc_per_node=8 scripts/benchmark.py \
    --config configs/pretrain/full_6b.yaml --distributed fsdp2 \
    --batch-size 1 --steps 10
```

把 `benchmark.py` 的 `tokens/s` 代回 §2.2 的公式，就能把 MFU 换成实测值；
把 `peak_mem_mb` 和本文 §1 对比，偏差超过 20% 时以实测为准。

---

## 6. 未验证与误差

- 显存是 **CPU 上 bf16 autocast 的 saved-tensor 字节**，不是 CUDA allocator
  峰值：CUDA 还有临时张量、碎片和 cuBLAS workspace，实测通常再高 10–20%。
- attention 的“两份拷贝”是 autocast 的行为，已在 CPU 上验证（fp32 不 autocast
  时是同样字节数的一份 fp32），但 CUDA 侧未实机确认。
- 时间估算依赖 MFU，误差可达 ±2×；`gather_attention` 的 elementwise 开销
  没有计入 FLOPs（只计 matmul），所以 MFU 取下限更稳。
- 3090/4090 的 NCCL 扩展性、8 卡 PCIe 拓扑未实测（`doc/testing.md` 已列）。
- 本文不涉及 SFT / GRPO / OPD 的时长：那三段的瓶颈是 rollout 吞吐，当前
  `rl/rollout.py` 是全量重算、无 KV cache，不能按本文的 pretrain 公式推。
