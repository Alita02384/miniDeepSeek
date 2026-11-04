# 路线图实现审计（2026-11-04）

> 对照 `../miniDeepSeek_technical_roadmap.md` 与
> `DeepSeek-V4_Towards Highly Efficient Million-Token Context Intelligence.pdf`
> （arXiv 2606.19348v1）逐项检查“代码是否落地”，不把“代码存在”等同于
> “GPU 上验证通过”。验证环境仍是 CPU + debug 模型。
>
> 图例：✅ 已实现并有 CPU 测试/证据；⚠️ 部分实现、可选路径或未在 GPU 验证；
> ❌ 未实现，或路线图明确列为非目标。

## 1. 结论

核心主线（模型、数据、预训练/SFT、GRPO、OPD、一致性、评测、FSDP2、HF 导出）
都已经有代码和测试。**但不是路线图 100% 完成**，剩余缺口集中在：

- 数据侧：长文档/多语言语料、FIM / token-splitting、真实 6B 数据；
- 训练侧：EP/CP/激活重计算（本轮明确不做）、teacher/rollout bf16、分布式 RL；
- 推理侧：vLLM 自定义模型/kernel、verl GPU 实跑；
- 后训练：teacher hidden-state 缓存/offload、>10 teachers、GRM（本轮明确不做）；
- 模型侧：vLLM/verl 的 GPU 数值验证；
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
| Muon + AdamW | ✅ | `train/optimizer.py` | 已按官方 Algorithm 1：8+2 hybrid NS、gamma=0.18、AdamW eps=1e-20；FSDP2 DTensor 路径 |
| Anticipatory Routing | ⚠️ | `model/anticipatory.py`, `train/trainer.py` | 已加一步预取队列 + 缓存索引；probe 仍同步执行，未做异步 overlap；默认关 |
| loss spike 检测 / 自动 AR | ✅ | `train/spike.py` | |
| rollback | ✅ | `train/spike.py` | FSDP2 gather/broadcast 快照；不恢复 dataloader 位置 |
| 4K dense-over-compressed | ✅ | `train/scheduler.py` | index_topk 调度 |
| 8K top-k 64 收紧 | ✅ | `train/scheduler.py` | 相对步数比例 |
| 2K→4K→8K sequence curriculum | ✅ | `train/curriculum.py` | 分阶段 runner：独立 2K/4K/8K 数据 + 累计 step + resume；不是单次动态长度 |
| FSDP2 | ✅ | `train/distributed.py` | CPU/Gloo 2 进程；NCCL 未验证 |
| DDP fallback | ✅ | `train/distributed.py` | 手动梯度平均 |
| EP（专家并行） | ❌ | — | 官方 §3.1；本项目 FSDP2 复制专家分片，无 all-to-all；60M/6 专家下收益有限 |
| TP / CP | ❌ | — | 官方 §3.4.3 CP 面向 1M context；本项目 8K 未实现 |
| 激活重计算 | ❌ | — | 官方 §3.4.4 是 tensor-level checkpointing；本项目 60M 未接 |
| bf16 训练 | ✅ | `train/trainer.py` | CUDA autocast；CPU 用 fp32 |
| eval/RL/teacher bf16 | ⚠️ | `model/wrapper.py` | CUDA 上 forward 统一 autocast bf16；权重仍 fp32（teacher 内存未省） |
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
| 段落级去重 | ✅ | `data/dedup.ParagraphDeduper` | 跨文档段落去重已接入 pretrain（`--no-paragraph-dedup` 可关） |
| 质量过滤（长度/符号/重复/模板） | ✅ | `data/quality.py` | |
| 语言/脚本识别 | ✅ | `data/quality.py` | 启发式已知 script 比例，非 fastText |
| code license / 自动生成 / vendored / binary | ✅ | `data/quality.py` | vendored 与二进制为启发式 |
| math 公式密度 / 答案完整性 | ✅ | `data/quality.py` | 默认只统计；可配置为拒绝 |
| 13-gram 污染检查 | ✅ | `data/contamination.py` | |
| fertility | ✅ | `data/fertility.py` | |
| manifest 权重/epoch 校验 | ✅ | `data/manifest.py` | |
| 加权数据混料 sampler | ✅ | `data/mix.py`, `train/run.py` | `data.sources` 权重混料；manifest 仍管 license/epoch |
| train/val/eval 无交集 | ⚠️ | GRPO 有 split；pretrain 只校验 manifest | 无自动去重交集检查 |
| 6B 语料 / 8K 长文数据 | ❌ | — | 当前只有 MiniMind 小样本 |
| FIM / token-splitting | ❌ | — | 官方 §4.1 明确继承；roadmap 未列，尚未实现 |
| 长文档 / 多语言语料 | ❌ | — | 官方 §4.1 强调；当前数据只有 MiniMind 小样本 |

## 4. 预训练 / SFT

| 路线图项 | 状态 | 代码/证据 | 备注 |
|---|---|---|---|
| 预训练循环 | ✅ | `train/trainer.py` | |
| LR warmup/cosine/wd/clip | ✅ | `train/scheduler.py`, `config.py` | 官方是 warmup→保持→末期 cosine；已加 `decay_start_ratio` |
| batch-token 线性 warmup | ✅ | `data/pretrain.iter_batches` | 前 10% steps；官方 §4.2.2 同样有 batch scheduling |
| LR flat→cosine | ✅ | `train/scheduler.py` | 官方 §4.2.2：大部分时间保持峰值，末期 cosine |
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
| local GRM | ❌（暂缓） | — | 官方 §5.1.1；用户明确先不做世界知识专家/GRM，等确定替代领域后再做 |
| medical safety reward | ✅ | `rl/reward.py` | |
| rollout↔actor logprob 一致性 | ✅ | `consistency/checks.py` | 9.5e-7 |
| router replay | ✅ | `model/routing.py` | torch 路径 100% |
| vLLM rollout | ⚠️ | `rl/backends/vllm.py` | 可选；未 GPU 验证 |
| vLLM router replay | ❌ | — | vLLM 不暴露路由 |
| 权重版本号 + logits 双校验 | ✅ | `rl/rollout.py`, `rl/backends/vllm.py` | 版本计数 + 可选 logprob 门禁；未在真实 vLLM 上跑 |
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
| 权重同步版本校验 | ⚠️ | `rl/rollout.py` | 代码门禁有；真实 vLLM 未跑 |
| dtype/autocast 统一 | ✅ | `model/wrapper.py` | 训练/rollout/eval/teacher 共用 wrapper autocast；CPU 保持 fp32 |
| math/code/world 自写评估 | ✅ | `eval/` | |
| lm-eval 通用基准 | ⚠️ | `eval/lm_eval.py` | 可选依赖，未实跑 |
| MMLU/ARC/C-Eval 实际分数 | ❌ | — | 未产出 |
| 域条件专家使用率 | ✅ | `routing.domain_routing_metrics` | GRPO 可开 `log_domain_routing` 记录 |
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

1. 数据：长文档/多语言语料、FIM/token-splitting（官方 §4.1；roadmap 未展开）；
2. 训练：teacher/rollout bf16、分布式 RL；EP/CP/激活重计算按用户要求暂不做；
3. 推理：目标 vLLM 版本 GPU smoke、verl GPU smoke；权重版本+logprob 门禁已有代码；
4. 后训练：teacher hidden-state cache/offload、>10 teachers；
5. 评测：lm-eval 实跑、格式评估器；GRM 按用户要求先不做。


## 10. 对照官方报告后的归类

### 10.1 官方有、roadmap 明确不做 / 降规模

- 1M context、32T/33T tokens、129K vocab、256/384 experts、top-6；
- FP4 QAT（官方 §5.2.1）、MegaMoE/EP 大规模（官方 §3.1）、CP（官方 §3.4.3）；
- Quick Instruction、`|DSML|` tool schema、interleaved thinking、三种 reasoning effort
  （官方 §5.1.1；roadmap 明确沿用 MiniMind tool 格式）；
- preemptible/fault-tolerant rollout（官方 §5.2.3）、batch-invariant kernels（§3.3）。

这些不作为 60M 研究模型的验收项；在 README / decisions 中已说明。

### 10.2 官方有、roadmap 要求、但本项目仍缺 / 部分

| 项 | 官方位置 | 现状 |
|---|---|---|
| Anticipatory Routing 的 prefetch + cache | §4.2.3 | 一步预取队列已实现；probe 未异步 overlap |
| 自动短 rollback + AR | §4.2.3 | 检测/AR/rollback 已有；rollback 不恢复 dataloader 位置 |
| sequence-length curriculum | §4.2.2 | 分阶段 2K→4K→8K runner 已实现 |
| EP | §3.1 | 未实现；FSDP2 只做参数分片 |
| 激活重计算 | §3.4.4 | 未接 |
| FIM / token-splitting | §4.1 | 未实现 |
| 长文档 / 多语言数据 | §4.1 | 未接入 |
| GRM | §5.1.1 | 只有启发式 + 可选 HTTP judge |
| 教师 hidden-state 缓存/offload | §5.2.2 | 未做（3 teacher/6.4K 规模下非必需） |
| 数据加权混料 | §4.2 | ✅ 已接 `data.sources` |
| 段落级去重 | roadmap §4.2.3 | ✅ 已接入 pretrain |
| vLLM / verl GPU 实跑 | — | 仅接口/配置，未验证 |

### 10.3 已按官方报告修正的差异

- **Muon**：从“NS-5 + 旧缩放”改为官方 Algorithm 1 的
  **8 步快速收敛 + 2 步稳定**、`sqrt(max(n,m)) * 0.18` 更新缩放；
- **AdamW eps**：从 1e-8 改为官方 1e-20；
- **mHC 参数分组**：`hc_base`/`hc_scale` 走 AdamW，`hc_fn` 走 Muon；
- **batch size scheduling**：补上线性的 batch-token warmup；
- **验证集**：补上 `val_path` 接线。
