# 262k batch 与领域同质 batch

> 记录两件事：把有效 batch 提到 262k 后数据量怎么定；领域同质 batch 在四个阶段分别怎么落地。
> 代码部分已实现（`data.lockstep`、`rl.teacher_lockstep`、`domain_id` 日志），见 §5。
>
> 配套：`doc/data_plan_145m.md`（数据源）、`doc/moe_specialization.md`（分化机制）。

---

## 1. 262k 怎么配（只改累积，不改显存）

```yaml
data:
  batch_tokens: 16384       # 全局 micro-batch：4 卡 × 4096，显存不变
train:
  grad_accum: 16            # 16384 × 16 = 262,144 token/step
```

**关键**：`batch_tokens` 是 micro-batch（决定显存），`grad_accum` 才是放大有效 batch 的旋钮。
显存不变，变的只是每 step 的累积次数。

---

## 2. 要多少数据

batch 不决定数据量，只决定"多少步走完"。数据量由 `tokens / 参数` 决定。

| 总 token | step 数（262k/step） | tokens/总参数 | 评价 |
|---|---|---|---|
| 2B | 7,630 | 13.7 | Chinchilla 以下，偏少 |
| **3B** | **11,444** | **20.6** | 甜点，推荐 |
| 4B | 15,259 | 27.4 | val loss 还在降就加 |
| 6B | 22,888 | 41.1 | 该规模收益递减 |

判断加不加：看 2.5B 时的 val loss 曲线，flat 了就停。

---

## 3. 课程三段的步数要重算

改 batch 后每步 token 变了，`until_step` 必须按新 token/step 重排（沿用原 15% / 55% / 30% 比例）：

| 阶段 | token | 步数 @262k |
|---|---|---|
| 2K | 0.45B | 1,717 |
| 4K | 1.65B | 6,295 |
| 8K | 0.90B | 3,434 |
| 合计 | 3B | **11,446** |

建议开 batch ramp：前 10% 步数从 131k 涨到 262k（`batch_tokens_final` + `batch_warmup_ratio`）。

---

## 4. 四个阶段：哪里需要领域同质 batch

先纠正一个常见误解：**专家模型不是用 OPD 训的**。

```text
预训练（混合三域）          ← 唯一长期"混"的阶段
   ├── SFT code    → code specialist      ← 单域，天然纯
   ├── SFT tool    → tool specialist      ← 单域，天然纯
   └── SFT general → general specialist   ← 单域，天然纯
        ↓
   OPD（多 teacher 蒸馏）→ 统一主模型      ← 混合
```

| 阶段 | 是否混域 | 要不要领域 batch |
|---|---|---|
| **预训练** | ✅ 混合 | ✅ **主战场** |
| 专家 SFT | 单域（若加 25% general 则混） | ⚠️ 加 general 时按 batch 交替，别塞进同一个 pack |
| 专家 GRPO | 单域 | ❌ 天然纯 |
| **OPD** | ✅ 多 teacher | ⚠️ 可选，默认关（见 §6） |

---

## 5. 预训练：已实现的接法

### 5.1 配置

```yaml
data:
  tokenized_path: null
  sources:
    - {path: data/tokenization/pretrain_general_4k.npz, weight: 0.40, domain: general}
    - {path: data/tokenization/pretrain_code_4k.npz,    weight: 0.15, domain: code}
    - {path: data/tokenization/pretrain_tool_4k.npz,    weight: 0.03, domain: tool}
    - {path: data/tokenization/pretrain_mixed_4k.npz,   weight: 0.42, domain: mixed}
  lockstep: true        # 每个 optimizer step 只来自一个 domain
train:
  grad_accum: 16
```

### 5.2 行为

- `lockstep: true` 时，`WeightedBatchMixer` 的配额按 **step** 分配（而不是按 micro-batch），
  每个 slot 连续产出 `grad_accum` 个同源 batch → **一个 step 的 262k token 全部同域**；
- 每个 step 的 `metrics.jsonl` 记录 `domain_id`（= `data.sources` 里的下标，-1 表示单源）；
- 因此每步的 `expert_load_max_over_mean` / `routing_entropy_normalized` 天然就是该领域的指标；
- `lockstep: false` 退回原来的"micro-batch 级交错"，用于对照实验。

### 5.3 数据目录要调整的地方

**原始 jsonl 不用动**，只需要把 `tokenization/` 按域拆开：

```text
data/
├── pretrain_t2t.jsonl                 # 不变（MiniMind 通用）
├── sft_general.jsonl / sft_tool.jsonl # 不变
├── grpo_*.jsonl                       # 不变
├── tokenization/
│   ├── pretrain_general_2k.npz …      # 新增：按域打包
│   ├── pretrain_code_2k.npz …
│   ├── pretrain_tool_2k.npz …
│   └── sft_*.npz                      # 不变（单域训练仍用单文件）
└── manifests/
    └── pretrain_3b.yaml               # 新增：四源权重 + domain
```

打包命令（每个域跑一次）：

```bash
python -m minideepseek.data.prepare pretrain \
    --input data/pretrain_code.jsonl \
    --output data/tokenization/pretrain_code_4k.npz \
    --seq-len 4096 --domain code
```

> `domain_id` 的顺序 = `data.sources` 的书写顺序，改顺序会改日志语义，别随意调换。

---

## 6. OPD：不改，保持混合池

**结论：不采用"每步只用一个 teacher"。**

- KL 按 teacher 分组是**正确性要求**（每个 teacher 有自己的 logits/mask），
  现有实现 `_group_by_teacher` 已经满足；
- "每步只用一个 teacher"没有公开证据支持它更好，而且和 OPD 的目标相悖：
  OPD 要的是把多个老师的信号融进同一个学生，混 batch 反而更接近"平均多个老师"；
  teacher-pure 的每步只朝一个方向走，老师之间可能来回拉扯。
- 曾实现过 `rl.teacher_lockstep` 开关，2026-11-08 已回滚，避免留下未验证的行为。

**保留的只有统计口径**：per-teacher 的 KL 统计（`kl_by_teacher`）继续记录，
用于诊断是哪个老师在退化；这不需要改采样方式。

## 7. 必须配套的两件事

1. **分域验证集**：general / code / tool 各一份，每 N 步全跑。只看混合 val loss 会互相抵消，
   看不出谁退化。
2. **训练后的路由报告**：把 `domain_id` 和 `expert_load_*` 交叉统计，确认
   "不同 domain 用不同专家"，而不是"所有 domain 都均匀"或"都塌缩到同一个专家"。

---

## 7.1 分域验证怎么用

```yaml
train:
  val_every: 200
  val_steps: 10
  val_sources:
    - {path: data/tokenization/val_general_4k.npz, weight: 1.0, domain: general}
    - {path: data/tokenization/val_code_4k.npz, weight: 1.0, domain: code}
    - {path: data/tokenization/val_tool_4k.npz, weight: 1.0, domain: tool}
```

每 200 步会在 `metrics.jsonl` 追加一条记录，除整体的 `val_loss`，还有：

```text
val_loss_general  val_main_loss_general  val_tokens_general
val_loss_code     val_main_loss_code     val_tokens_code
val_loss_tool     val_main_loss_tool     val_tokens_tool
```

课程三段各自指向本长度的 val pack（`val_*_2k/4k/8k.npz`），所以 2K 阶段不会喂 4K 的包。

## 8. 当前状态

| 项 | 状态 |
|---|---|
| 262k 配置 | ✅ `configs/pretrain/pretrain_145m.yaml`（`batch_tokens 16384` × `grad_accum 16`，11,444 步） |
| 课程 until_step | ✅ 1717 / 8011 / 11444（0.45B / 1.65B / 0.9B，三段数据不重叠） |
| `lockstep` 预训练混料 | ✅ 代码 + 单测（`test_lockstep_mixer_keeps_each_step_on_one_source`） |
| `domain_id` 日志 | ✅ 代码 + 单测（`test_trainer_logs_domain_id_from_mixer`） |
| 按域打包的 npz | ✅ `data/tokenization/pretrain_{general,code,tool}_{2k,4k,8k}.npz`（`scripts/build_pretrain_packs.sh`） |
| 分域验证集 | ✅ `data/tokenization/val_{general,code,tool}_{2k,4k,8k}.npz`（各 1M token，训练跳过） |
| OPD teacher 锁定 | ❌ 已回滚（不采用，见 §6） |
| 训练侧 val 接线 | ✅ `train.val_sources` + `CurriculumStage.val_sources`，每域单独评估 |

### 8.1 流式打包

`build_pretrain_dataset` 会把整份语料读进内存（2B token ≈ 18 GB），这台机器只有
16 GB，所以新增 `scripts/pack_domain.py`：

- 逐文档 tokenize，按 `pack_documents` 的语义累积（≤`seq_len`、每文档 position 从 0 开始）；
- 直接追加写 `.npy`，最后回填 header，内存占用 O(seq_len)；
- 输出与 `load_flat_dataset` 完全兼容；
- 多输入时**按文档轮转**（general = MiniMind + 外部语料），避免预算被第一个文件吃满。
