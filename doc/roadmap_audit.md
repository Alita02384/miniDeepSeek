# 路线图实现审计（2026-11-04）

> 对照 `../miniDeepSeek_technical_roadmap.md` 逐项检查“代码是否落地”，
> 不把“代码存在”等同于“GPU 上验证通过”。验证环境仍是 CPU + debug 模型。
>
> 图例：✅ 已实现并有 CPU 测试/证据；⚠️ 部分实现、可选路径或未在 GPU 验证；
> ❌ 未实现，或路线图明确列为非目标。

## 1. 结论

核心主线（模型、数据、预训练/SFT、GRPO、OPD、一致性、评测、FSDP2、HF 导出）
都已经有代码和测试。**但不是路线图 100% 完成**，主要缺口集中在：

- 数据侧：段落级去重、加权混料、语言识别/vendored/二进制/数学公式过滤；
- 训练侧：sequence-length curriculum、EP、激活重计算、分布式 RL；
- 推理侧：vLLM 自定义模型/kernel、verl GPU 实跑、vLLM 权重版本校验；
- 模型侧：teacher/rollout 的 bf16 权重（目前训练有 autocast，eval/RL 没有）；
- 规模侧：没有 6B 数据，也没有真实训练 mini-60M（按项目要求）。

## 2. 模型与训练技术

| 路线图项 | 状态 | 代码/证据 | 备注 |
|---|---|---|---|
| DeepSeekMoE 6 routed + 1 shared，top-2 | ✅ | `model/moe.py` | 与参考实现逐层对齐 |
| hash routing 前 3 层 | ✅ | `model/moe.py` | tid2eid 持久 buffer |
| SqrtSoftplus / route_norm / route_scale | ✅ | `model/moe.py` | |
| loss-free expert bias | ✅ | `train/losses.py` | DP 下计数 all-reduce |
| sequence-wise balance loss | ✅ | `train/losses.py` | 权重 1e-4 |
| mHC hc_mult=4 + Sinkhorn 20 | ✅ | `model/mhc.py` | |
| CSA(4) + HCA(32) + SWA | ✅ | `model/attention.py`, `compressor.py` | 三种层型 |
| CSA indexer + top-k + 蒸馏 loss | ✅ | `model/compressor.py` | indexer loss 有梯度 |
| MTP depth 1 + loss | ✅ | `model/model.py`, `wrapper.py` | 含多文档 label 边界 |
| SwiGLU clamp 10 + 监控 | ✅ | `model/nn.py`, `train/metrics.py` | |
| Muon + AdamW | ✅ | `train/optimizer.py` | FSDP2 DTensor 路径已对齐 |
| Anticipatory Routing | ⚠️ | `model/anticipatory.py` | stale-gate probe + replay，非 dataloader 预取；默认关 |
| loss spike 检测 / 自动 AR | ✅ | `train/spike.py` | |
| rollback | ⚠️ | `train/spike.py` | 单进程内存快照；distributed 下显式拒绝 |
| 4K dense-over-compressed | ✅ | `train/scheduler.py` | index_topk 调度 |
| 8K top-k 64 收紧 | ✅ | `train/scheduler.py` | 相对步数比例 |
| 2K→4K→8K sequence curriculum | ❌ | — | 需要分阶段数据/配置，未做单次训练调度 |
| FSDP2 | ✅ | `train/distributed.py` | CPU/Gloo 2 进程；NCCL 未验证 |
| DDP fallback | ✅ | `train/distributed.py` | 手动梯度平均 |
| EP（专家并行） | ❌ | — | 当前 FSDP2 复制专家分片，无 all-to-all |
| TP / CP | ❌ | — | 未实现 |
| 激活重计算 | ❌ | — | 未接 checkpoint_wrapper |
| bf16 训练 | ✅ | `train/trainer.py` | CUDA autocast；CPU 用 fp32 |
| eval/RL/teacher bf16 | ⚠️ | — | 模型仍 fp32；只有训练器 autocast |
| FP8 / FP4 / 1M context | ❌ | — | 路线图非目标 |

## 3. 数据技术

| 路线图项 | 状态 | 代码/证据 | 备注 |
|---|---|---|---|
| MiniMind tokenizer / chat template | ✅ | `tokenizer/` | hash 记录进 checkpoint |
| response-only mask | ✅ | `tokenizer/tokenizer.py` | 多轮/工具格式 |
| pretrain tokenize + packing | ✅ | `data/pretrain.py` | sample-level 边界 |
| SFT packing | ✅ | `data/sft.py` | |
| GRPO/OPD schema | ✅ | `data/grpo.py`, `opd.py` | |
| MinHash 文档级去重 | ✅ | `data/dedup.py` | |
| 段落级去重 | ❌ | — | `dedup.py` 已注明只做文档级 |
| 质量过滤（长度/符号/重复/模板） | ✅ | `data/quality.py` | |
| 语言识别 | ❌ | — | 未接 fastText/langid |
| code license / 自动生成过滤 | ✅ | `data/quality.py` | vendored 目录/二进制未做 |
| math 公式密度 / 答案完整性 | ❌ | — | 只在 GRPO 转换时要求 reference |
| 13-gram 污染检查 | ✅ | `data/contamination.py` | |
| fertility | ✅ | `data/fertility.py` | |
| manifest 权重/epoch 校验 | ✅ | `data/manifest.py` | |
| 加权数据混料 sampler | ❌ | `manifest.py` 只有 plan | 未接进 `iter_batches` |
| train/val/eval 无交集 | ⚠️ | GRPO 有 split；pretrain 只校验 manifest | 无自动去重交集检查 |
| 6B 语料 / 8K 长文数据 | ❌ | — | 当前只有 MiniMind 小样本 |

## 4. 预训练 / SFT

| 路线图项 | 状态 | 代码/证据 | 备注 |
|---|---|---|---|
| 预训练循环 | ✅ | `train/trainer.py` | |
| LR warmup/cosine/wd/clip | ✅ | `train/scheduler.py`, `config.py` | |
| batch-token 线性 warmup | ✅ | `data/pretrain.iter_batches` | 前 10% steps |
| MTP 权重 0.3→0.1 | ✅ | `train/trainer.py` | |
| 梯度累积 | ✅ | `train/trainer.py` | |
| val_path 验证集 | ✅ | `train/run.py` | 本轮补上 |
| 预训练 0.6B/6B 实跑 | ❌ | — | 按项目要求不训练 |
| SFT response-only | ✅ | `data/sft.py` | |
| 三个 specialist 从 base 分出 | ✅ | `configs/sft/*`, resume | |
| SFT 领域+通用混料 | ⚠️ | 配置/manifest | 无加权 sampler |
| 输出格式正确率 >95% 验收 | ⚠️ | reward/eval 统计 | 无专门格式评估器 |

## 5. 后训练（GRPO / OPD）

| 路线图项 | 状态 | 代码/证据 | 备注 |
|---|---|---|---|
| GRPO group/advantage/clip/KL | ✅ | `rl/grpo.py` | |
| math verifier reward | ✅ | `rl/reward.py` | |
| code unit-test reward | ✅ | `rl/reward.py` | 子进程执行 |
| world rubric reward | ✅ | `rl/reward.py` | 启发式默认 |
| world HTTP judge | ⚠️ | `rl/judge.py` | 可选，未真实调用 |
| local GRM | ❌ | — | 未实现 |
| medical safety reward | ✅ | `rl/reward.py` | |
| rollout↔actor logprob 一致性 | ✅ | `consistency/checks.py` | 9.5e-7 |
| router replay | ✅ | `model/routing.py` | torch 路径 100% |
| vLLM rollout | ⚠️ | `rl/backends/vllm.py` | 可选；未 GPU 验证 |
| vLLM router replay | ❌ | — | vLLM 不暴露路由 |
| 权重版本号 + logits 双校验 | ❌ | — | vLLM 更新只重建 engine |
| OPD on-policy rollout | ✅ | `rl/opd.py` | |
| OPD full-vocab reverse KL | ✅ | `rl/opd.py` | 6400 词表 |
| 多教师 hard key / 权重 | ✅ | `rl/opd.py`, `configs/opd` | |
| teacher no_grad/eval | ✅ | `rl/opd.py` | bf16 未做 |
| verl GRPO/OPD | ⚠️ | `rl/verl.py`, `scripts/export_verl.py` | 接口/配置，未 GPU 实跑 |
| verl full-vocab reverse KL | ❌ | — | verl 模板用 forward_kl_topk；canonical 是 `rl/opd.py` |

## 6. 一致性 / 评测 / 发布

| 路线图项 | 状态 | 代码/证据 | 备注 |
|---|---|---|---|
| tokenizer hash | ✅ | checkpoint meta | |
| logprob 门禁 | ✅ | `consistency/checks.py` | <1e-3 |
| routing 门禁 | ✅ | `consistency/checks.py` | ≥99.9% |
| checkpoint 门禁 | ✅ | `consistency/checks.py` | 位级 |
| 权重同步版本校验 | ❌ | — | 见上 |
| dtype/autocast 统一 | ⚠️ | 训练有 autocast；eval/RL 无 | |
| math/code/world 自写评估 | ✅ | `eval/` | |
| lm-eval 通用基准 | ⚠️ | `eval/lm_eval.py` | 可选依赖，未实跑 |
| MMLU/ARC/C-Eval 实际分数 | ❌ | — | 未产出 |
| 域条件专家使用率 | ❌ | `routing_metrics` 只有整体指标 | 未按 domain 分组 |
| HF 导出 | ✅ | `model/hf/` | logits 逐位一致 |
| torch/safetensors/DCP checkpoint | ✅ | `model/checkpoint.py` | DCP 仅单进程验证 |
| 分布式 DCP | ⚠️ | 函数存在 | 未在 FSDP2 下验证 |

## 7. 明确非目标（不算缺口）

按 roadmap §1.2：1M context、FP4 训练、MegaMoE/EP 大规模、32T 数据、
原生 129K 词表；按 §5：DSpark / 推测解码第一版不做。

## 8. 验证记录（本机）

```text
ruff check src tests scripts              All checks passed
pytest -q                                 114 passed (CPU debug model)
scripts/debug_smoke.sh                    exit 0
scripts/test_distributed.py world=1/2     sha256 一致 (FSDP2 / DDP 各自)
scripts/verify_reference.py               block rel <= 1.6e-4
HF export → from_pretrained               max |Δ| = 0
```

## 9. 建议优先级

1. 数据：加权混料 sampler、段落级去重、语言/vendored/二进制/数学过滤；
2. 训练：sequence-length curriculum、teacher/rollout bf16、激活重计算；
3. 推理：vLLM 权重版本号+logits 校验、目标版本 GPU smoke；verl GPU smoke；
4. 并行：EP（6 专家在 60M 上收益有限，可后置）、分布式 RL；
5. 评测：lm-eval 实跑、domain-conditioned 路由指标、格式评估器。
