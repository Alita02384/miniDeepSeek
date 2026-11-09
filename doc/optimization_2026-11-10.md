# 145M 训练吞吐优化记录（2026-11-10）

> 目标：不改变模型结构、不改变每个 optimizer step 的 262,144 token 预算，
> 把 1×4090 上的训练吞吐从 8.05k tok/s 提上来。
> 相关提交：`931be68`（micro-batch / bf16 attention / MoE 排序）、
> `f770267`（训练循环去同步 / mHC 编译）。

## 0. 结论

| 指标 | 优化前 | 优化后 |
|---|---|---|
| `benchmark.py` 4K×1（含 optimizer） | 6,220 tok/s | **7,357 tok/s（+18%）** |
| `benchmark.py` 4K×2（含 optimizer） | 9,304 tok/s | **11,987 tok/s（+29%）** |
| 相对原始 4K×1 的端到端 | — | **1.93x** |
| 2K 训练稳态 | 8.05k tok/s | 待测（benchmark 推算 13–15k） |
| 2.35B token 的训练时间 | ~3.4 天 | **~2 天（预估）** |
| 4K 峰值显存 | 6.5 GiB | 11.9 GiB（24 GB 卡仍有余量） |

## 1. 基线：钱花在哪

4K、单 pack、bf16 autocast、fused attention 的分解（`torch.profiler` + CUDA event）：

| 项 | 每步耗时 | 说明 |
|---|---|---|
| fwd+bwd 总计 | 364–389 ms | 4,096 token |
| attention（fused） | ~40–84 ms | 5 个 sparse 层；kernel 实际收到 **fp32** 张量 |
| MoE dispatch | ~48 ms | 6 次 `x[idx]` + `index_add_`，每个专家一次 `.any()` 同步 |
| mHC / Sinkhorn / elementwise | ~80–100 ms | 20 次迭代 × 16 个模块，单步上万个小 kernel |
| Muon+Adam | 309 ms / optimizer step | grad_accum=64 摊薄后 ~1% |
| CPU 间隙 | 墙钟约比 kernel 总时长多 30% | 每 micro-batch 数百次 `.item()` |

关键发现：**attention 实际在跑 fp32**。RoPE/RMSNorm 把 q/kv 升到 fp32，而 attn_gym
的 fp32 路径不用 tensor core——同一个 kernel bf16 0.27 ms、fp32 20 ms（75x）。

## 2. 改动清单

### 2.1 micro-batch 4096 → 8192（最大的一笔）

- 24 GB 卡实测：4K×2 = **11.6–11.9 GiB**；4K×4（16384 token）在 20.4 GiB 处 **OOM**。
  所以 8192 token/rank 是安全上限，8K 阶段本来也是 8192/rank。
- `scripts/autoconfig_145m.py` 的 `--per-rank-tokens` 默认改为 8192，三阶段
  `batch_tokens=8192 / grad_accum=32`，每步 token 数仍为 262,144。
- benchmark 4K×1 6,220 → 4K×2 9,304（含 optimizer，+50%）；纯 fwd+bwd 口径 +22%
  （trainer 里 optimizer 已被 32 步累积摊薄，所以训练端收益接近 +22%）。

### 2.2 attention 边界 cast bf16

- 只在调 attn_gym 前把 q/local_kv/sparse_kv cast 成 bf16，输出 cast 回调用者 dtype；
  softmax/LSE 仍在 kernel 内 fp32，残差流和优化器状态不变。
- 模型内 A/B：364 → 327 ms（+10%）；新增 fp32 输入下的 fused-vs-reference 等价测试。
- 依据：上游 torchtitan 的 V4 attention 也是把 bf16 的 q/kv 直接交给 `gather_attn`。

### 2.3 MoE 排序 dispatch

- `argsort(token→expert)` + `bincount` + 单次 `index_add_`，替代 6 次
  mask/index_select/index_add；`cumsum().tolist()` 一次同步替代每个专家一次
  `mask.any()` 同步。
- 数学等价（新增按 token 显式循环的 fp32/bf16 等价测试，含空专家分支）。

### 2.4 训练循环去同步

每 micro-batch 的 host 同步从 **~230 次降到 ~4 次**：

| 位置 | 改法 |
|---|---|
| `swiglu_monitor` | 每次激活调用的 4 次 `.item()` → 设备侧累加，monitor 退出时一次 `tolist()` |
| `routing_metrics` | 每层 3 次 `float()` → 全部 stack 后一次 `tolist()` |
| trainer 标量 | loss/mtp/balance/indexer 的 8 次 `float()` → 一次 `tolist()` |

### 2.5 mHC Sinkhorn 走 torch.compile

- 只编译 `_sinkhorn_normalise`（20 次行列归一化），`dynamic=True`，带 `try/except`
  和 eager 回退；不编译整模型（MoE 动态 shape 会重编译）。
- 单次调用 1.21 ms → **0.557 ms（2.17x）**，数值误差 1.8e-7；但整步只 +0.6%
  （mHC 在 4K 整步里占比太小），属于“顺手做掉”。

## 3. 没有采用 / 试过无效的方案

| 方案 | 结论 |
|---|---|
| 整模型 `torch.compile` | 之前实测 **0.78x**，MoE 动态 shape 触发 64 次重编译上限 |
| CUDA graphs | MoE 每步 token 分布不同，需 padding/分桶，风险高；上游也只在固定 shape/EP 特例下支持 |
| FlashAttention | 不支持“每个 query 不同 KV 集合”和 attention sink；且 cu130 无预编译 wheel |
| DeepGEMM / FlashMLA / TileKernels | 全部 SM90+，4090（SM89）不可用 |
| FlexAttention | top-k 选择无法用 block mask 表达，会退化为逐元素比较 |
| `torch._grouped_mm` | 4090 可用，MoE 的 6 次 GEMM 3.69 vs 5.39 ms（1.46x），但整步只 ~0.7%，暂不引入 |
| 更大模型 | 不是本项目目标 |

## 4. 本地开源仓库调研

- **torchtitan（V4 参考实现）**：attention 同样调 `attn_gym.gather_attn(impl="fused")`，
  且直接把 bf16 的 q/kv 传进去——验证了 2.2；MoE 用 `argsort` + `torch._grouped_mm`
  （我们只差 grouped GEMM）；mHC 是和我们完全一样的 20 次 eager 循环；compile 是可选项。
- **minimind**：标准 causal + `F.scaled_dot_product_attention`，可选 `torch.compile`；
  它的效率来自“dense + 固定 shape”，我们的非标准 attention 拿不到这个待遇。
- **DeepSeek-V4-Flash**：TileLang 推理 kernel（FP8/FP4、sparse attention），只支持
  Hopper/Blackwell、只有前向，4090 不可用。
- **verl / LlamaFactory**：RL/SFT 流程，对预训练吞吐没有直接帮助。

## 5. 剩余空间与风险

1. **8K 阶段未实测**：8192/rank + activation checkpointing 的显存在 24 GB 上还没跑过。
2. **Triton 编译时间**：每个新 dtype/shape 组合首次编译 10–20 分钟（`make_ttgir`），
   换 stage 会再付一次；建议每阶段开训前先跑一次同 shape 的 benchmark 预热缓存。
3. **mHC 真正的融合**：`torch.compile` 只拿到 2.17x 单模块收益；要再进一步需要
   手写 Triton（20 次迭代在寄存器里完成），预计整步 +5% 左右。
4. **MoE grouped GEMM**：接 `torch._grouped_mm` 预计整步 +0.7%，收益低于维护成本。
5. **CPU 侧仍有 ~30% 间隙**：进一步要靠 CUDA graphs 或把 mHC/norm/rope 一起 compile。

## 6. 复现

```bash
# 生成 8192/rank 的三阶段配置
.venv/bin/python scripts/autoconfig_145m.py --gpus 1

# benchmark（4K×2 是训练口径）
.venv/bin/python scripts/benchmark.py --flavor mini_145m_a60m_topk64 \
  --seq-len 4096 --batch-size 2 --steps 3 --warmup 2 --device cuda

# 训练
NPROC_PER_NODE=1 bash scripts/train_curriculum.sh \
  configs/pretrain/generated/curriculum_145m_g1.yaml
```
