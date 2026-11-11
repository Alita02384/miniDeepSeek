# 后训练（SFT + GRPO + OPD）实现与运行计划

> 目标环境：先单卡 4090（SM89）开发，最终多卡 CUDA。模型为 `mini_145m_a60m`
> （145.8M total / 60.9M active，V4 同构：MoE + mHC + CSA/HCA + MTP）。
>
> 相关记录：`doc/decisions.md` D82（vLLM 不可用与自研 KV-cache 引擎）、
> D83（GRPO old logprob 由 actor 重算）、`doc/sources.md`（增量 compressor 来源）。

---

## 1. 结论先行

1. **vLLM 0.33 无法服务这个模型**：它的 DeepSeek-V4 实现只有 FlashMLA
   （要求 SM90，kernel 内 `head_dim_v=512`、FP8 KV）和 FlashInfer（SM100/120）
   两类后端；我们的 4090 是 SM89，模型是 `head_dim=64` 的 bf16。H100 也不行
   （head_dim 不匹配）。详见 D82。
2. 因此 rollout 走**项目内自研 KV-cache 引擎**：数学与训练前向一致，
   slot 寻址、prefill/group-fork/批量 decode、CUDA graph 重放。
   单卡 4090 实测 decode **5.4 ms/step @ batch 64 ≈ 12k tok/s**（eager 49 ms）。
3. 多卡按 **DDP** 开发：每个 rank 一份完整模型 + 一份 rollout 引擎，
   prompt 按 rank 分片；145M 上 DDP 比 FSDP2 快（D81）。FSDP2 需要额外的
   gathered rollout 副本，暂未接。
4. GRPO 的 old logprob **由 actor 重算**（D83）。rollout 引擎与全量前向在
   bf16 下逐 token logprob 差约 4e-2，直接用采样 logprob 会给 ratio 引入
   系统性偏差；on-policy 的 actor 重算让 ratio 从 1 开始。

---

## 2. 流水线

```text
base checkpoint (pretrain)
  ├── sft_general_145m ──> grpo_general_145m ─┐
  ├── sft_code_145m    ──> grpo_code_145m    ├─> opd_three_teacher_145m
  └── sft_tool_145m    ──> grpo_tool_145m    ┘
```

- **SFT**：response-only mask，领域数据 70–90% + general 10–30%；
  每个 opt step 8,192×16 = 131,072 input token。2026-11-11 用重建后的数据实测：
  general 280 步（≈1.0 target epoch）、code 115 步（≈1.0）、tool 95 步
  （≈2.0）。`scripts/sft_step_budget.py` 可复算。MTP loss 0.3→0.1，
  sequence-wise balance 1e-4，indexer top-k 64。
  **`data.sources` 的 weight 按 loss token 计**：实现里用每个源的
  `target/input` 比例把 input 权重折算成 loss-token 权重，实测
  general/code/tool 的 loss 份额为 0.900/0.744/0.720。tool 源先按
  `require_tools=True` 剔除文件尾部的 general 切片。
- **GRPO**：group 6、16 prompts/step、max_new 512、temperature 0.8、
  CISPO/GRPO clip 0.2、KL 0；`rollout_backend=cached`。
  **注意**：`recompute_old_logprobs=true` 时 old logprob 由 actor 重算，ratio
  恒为 1，clip 实际不生效，等价于 on-policy REINFORCE + group baseline；
  不要把 `clip=0.2` 当成有效 trust region。
- **OPD**：student 自己采样，三个 teacher 对同一轨迹算 **full-vocabulary
  reverse KL**（6400 词表），按 `teacher_key` 路由，权重 1/3 each；
  student 不做 GRPO（与 V4 一致）。loss 的分母是**固定** `Σw_i`，某步缺
  teacher 不会放大剩余权重；老师和学生都用 T=0.8 采样，实际最小化的是
  `KL(π_s^T‖π_t^T)`，不是 untempered KL。

---

## 3. 关键代码

| 文件 | 作用 |
|---|---|
| `src/minideepseek/model/cache.py` | `RolloutCache` / `LayerCache`：slot 寻址的 KV ring、compressed KV、indexer K、partial compressor state |
| `src/minideepseek/model/compressor.py` | `project_tokens` / `capture_state` / `decode_step`：official 推理状态机的移植；`Indexer.query` / `compress_decode` |
| `src/minideepseek/model/attention.py` | `DeepSeekV4Attention.decode`（local window + top-k compressed + sink）；prefill 时写 cache；reference attention 按 query chunk |
| `src/minideepseek/model/model.py` | `forward(cache=..., slot=...)` 与 `decode(tokens, positions, cache, slots)` |
| `src/minideepseek/model/wrapper.py` | `prefill` / `decode_step` / `logits_at`（只算 response 位置的 logits） |
| `src/minideepseek/rl/cached_rollout.py` | 批量 prefill、group fork、活跃 slot 压缩解码、stop/max_new |
| `src/minideepseek/rl/graph_decode.py` | 分桶 CUDA graph；padding 行用专用 scratch slot |
| `src/minideepseek/rl/grpo.py` | old logprob 重算、CISPO/GRPO loss、routing 指标 |
| `src/minideepseek/rl/opd.py` | 多教师 full-vocab reverse KL、response-only logits |
| `scripts/test_rl_distributed.py` | 2 rank DDP + cached 的 GRPO/OPD smoke |

### 3.1 KV cache 布局

每层：`kv[S,W,D]`（ring，只留 local window）、`cmp[S, L/R, D]`、
`idx[S, L/R, index_dim]`（CSA）、`comp_state_*`、`idx_state_*`。
prefill 一次写入；group 采样用 `copy_slot` 分叉；decode 每步追加并更新状态。
compressor 的 overlap（R=4）状态与 official `inference/model.py` 的
`kv_state/score_state` 同构：`[2R, 2D]`，完成一个 block 时
`cat([state[:R,:D], state[R:,D:]])` 做 softmax 加权，然后左移。

### 3.2 为什么 CUDA graph 是必须的

eager decode 每步约 49 ms，其中 GPU 只有约 5 ms，其余是 ~1000 个 kernel 的
Python/launch 开销（mHC Sinkhorn 40 次迭代 × 7 层 × 2）。CUDA graph 把整步
重放压到 **5.4 ms**。分桶（1,2,4,...,512）避免为每个 active 数重捕；
padding 行指向 scratch slot，不会污染真实序列。capture 期间关闭
`MINIDEEPSEEK_COMPILE_MHC`（Sinkhorn 走 eager），保证图内无 host sync。

### 3.3 已知数值差异

decode（bf16 bmm/不同 batch 形状）与 actor 全量前向的 logprob 差约 **4e-2**；
这是 kernel 顺序/批量形状导致的 bf16 误差，不是 cache 数学错误（fp32 CPU
逐 token 对齐到 2e-6）。因此：

- GRPO 的 old logprob 由 actor 重算（`rl.recompute_old_logprobs=true`）；
- 一致性门禁对 cached 引擎不再比较 sampling logprob，而是比较
  `decode` 与 `prefill/full-forward` 的 logits（`tests/test_cache.py`）。

---

## 4. 实测（单卡 4090，145M bf16）

| 项目 | 数值 | 条件 |
|---|---:|---|
| SFT | 6.1–7.0k tok/s | 4K、micro 8192、grad_accum 16、MTP 开 |
| prefill | ~81 ms / 561 token | 一次 forward + 写 cache；torch reference attention |
| decode eager | 49 ms/step | batch 64 |
| decode graph | **5.4 ms/step** | batch 64，图重放 |
| rollout（16 prompt × group 4 × 32 token） | 1.4–1.5 s | 含 16 次 prefill；≈1.4k tok/s |
| 峰值显存 | 1.45 GB（rollout 64 slot） | cache + 145M |
| SFT 峰值显存 | 12.7 GB | micro 8192、seq 4096 |
| GRPO step（16 prompt × group 6 × 512 token） | **18.6 s** | rollout 6.3 / reward 3.1 / actor fwd+bwd 8.9 / opt 0.4 |
| GRPO 峰值显存 | 9.0 GB | actor mini-batch 16、activation checkpointing、1.8B RM 同驻 |

> 注 1：rollout 的端到端数字被 prefill 拖累（16×81 ms）；decode 本身已是
> 12k tok/s 量级。下一步是 chunked prefill（把 prefill 也走 cache 路径）。
>
> 注 2：GRPO 第一次跑全配置时 OOM，根因是 `Indexer.select` 的
> `[T, H, S]` 分数张量（49k × 8 × ~12k ≈ 7 GB）与无 checkpoint 的 actor
> 反传。修法：indexer 按 query chunk 512 分块；actor 按 16 sample
> mini-batch 反传（`rl.ppo_mini_batch_size`）；`train.activation_checkpointing=true`；
> RL 模型永久关闭 indexer aux loss（本来就不反传，且会让 checkpoint 重算图不一致）。

---

## 5. 运行顺序

```bash
# 0) 环境
bash scripts/setup_env_gpu.sh                 # torch 复用镜像，装 [dev,hf]
bash scripts/build_pretrain_packs.sh          # 已有 2K/4K/8K pack 时跳过

# 1) SFT 数据（每域一次）
.venv/bin/python - <<'PY'
from minideepseek.data.sft import build_sft_dataset
from minideepseek.tokenizer import load_tokenizer
tok = load_tokenizer()
# 全量扫描 -> 会话去重 -> 按唯一记录数比例分配 -> seed 打乱行号
for domain, n in [("general", 50000), ("code", 30000), ("tool", 8000)]:
    build_sft_dataset([f"data/sft_{domain}.jsonl"],
                      f"data/tokenization/sft_{domain}.npz",
                      tokenizer=tok, seq_len=4096, max_records=n, seed=1234)
# 等价的 CLI：python -m minideepseek.data.prepare sft --input ... --output ...
PY

# 2) 三个 specialist SFT（顺序跑，单卡约 30–40 min/个）
bash scripts/train_sft.sh configs/sft/general.yaml
bash scripts/train_sft.sh configs/sft/code.yaml
bash scripts/train_sft.sh configs/sft/tool.yaml

# 3) 三个 GRPO（依赖 SFT final）
bash scripts/train_grpo.sh configs/grpo/general.yaml
bash scripts/train_grpo.sh configs/grpo/code.yaml
bash scripts/train_grpo.sh configs/grpo/tool.yaml

# 4) 三教师 OPD
bash scripts/train_opd.sh configs/opd/three_teacher.yaml
```

多卡：`NPROC_PER_NODE=2 bash scripts/train_sft.sh ...`；
GRPO/OPD 使用 `distributed.mode=ddp`，每 rank 一份 cached 引擎，
prompt 按 `records[rank::world]` 分片。

---

## 6. 未完成 / 风险

1. **tool 多轮 rollout 仍走全量重算**：`ToolRolloutEngine` 在每次 tool
   observation 后重新 prefill。需要给 cached 引擎加 `prefill_append`
   （把 observation token 逐个 `decode_step` 追加，再继续采样）。
2. **prefill 仍是 torch reference**（81 ms/561 token）；chunked prefill
   可以把 prefill 也走 cache 路径，预计降到 ~10 ms。
3. **RL/OPD 全程用 torch reference attention（D84）**：attn_gym 按 packed
   布局重编译，OPD teacher forward 会卡在 Triton 编译；`rl/run.py` 与
   `train_grpo.sh`/`train_opd.sh` 已默认 `MINIDEEPSEEK_ATTN_BACKEND=torch`。
4. **FSDP2 + cached 未接**：需要 gather 一份完整权重给 rollout；当前只支持
   DDP，多卡规模上限受 24 GB 单卡限制。
5. **OPD 的 teacher forward 仍是全量前向**；teacher 很小（145M），可接受；
   若 teacher 变大，需要 chunked forwarding 或单独进程。
6. **GRPO 的 reward**：general 依赖 `internlm2-1_8b-reward`（3.6 GB fp16）；
   单卡与 actor 同驻时峰值约 16–20 GB，需要实测。
7. **一致性门禁**：bf16 下 rollout/actor 的 logprob 差 ~4e-2，不能按
   roadmap 的 1e-3 卡；已改为 actor 重算 old logprob + fp32 cache 单测。

---

## 7. 实跑状态（2026-11-11）

| 阶段 | 状态 | 关键数字 |
|---|---|---|
| SFT data | ✅ | general 50k/33.1M tok、code 30k/17.6M、tool 8k/5.1M |
| SFT general | ✅ `checkpoints/sft_general_145m/final` | 120 step，main loss 5.5→2.7，6–7k tok/s，12.7 GiB |
| GRPO general | ✅ `checkpoints/grpo_general_145m/final` | 50 step，49k resp tok/step，reward −1.8~−2.2，grad 0.6，18.6 s/step，9.0 GiB |
| OPD smoke（同 teacher） | ✅ | KL 0，grad 0（同权重应然） |
| OPD smoke（pretrain teacher） | ✅ | KL 0.315，grad 13.2（不同权重，验证 KL 路径） |
| SFT code / tool | ✅ `sft_code_145m/final`、`sft_tool_145m/final` | code 90 step / tool 80 step，顺序跑 |
| GRPO code | ✅ 10 step `checkpoints/grpo_code_145m/final` | reward mean −0.085（std 0.040），grad 0.27，24.6k resp tok/step |
| GRPO tool | ✅ 10 step `checkpoints/grpo_tool_145m/final` | reward mean −2.235（std 0.827），grad 0.46，49.2k resp tok/step（cached tool 引擎） |
| 三教师 OPD（SFT teacher） | ✅ 20 step `checkpoints/opd_three_teacher_145m/final` | KL total mean 0.065；general 0.041 / code 0.052 / tool 0.100 |
| 三教师 OPD（全 GRPO teacher） | ✅ 20 step `checkpoints/opd_three_grpo_145m/final` | KL total mean 0.065（first 0.066 / last 0.067）；general 0.040 / code 0.054 / tool 0.098；grad mean 0.87 |

### 7.1 已知现象

- GRPO general 的 response 全部顶到 `max_new=512`，reward 在 −2 附近；说明
  SFT 后模型还不愿主动收尾，或者 512 太短。真实训练应把 `max_new_tokens`
  提到 768/1024，并检查 stop token 命中率（`rollout` 已记录 `finished`）。
- reward 的 think/length 分量很小，主项是 1.8B GRM（−2.1 左右）；group 内
  std ~0.63，advantage 有信号。
- 单卡 4090 上 GRPO 的 18.6 s/step 里：rollout 6.3 / RM 3.1 / actor fwd+bwd
  8.9 / opt 0.4。rollout 已不是唯一瓶颈，actor 的 reference attention 仍是
  大头；进一步提速需要 chunked prefill + 更大 query chunk + 更快的
  gather attention（H100 上可用 attn_gym/FlashAttention）。
- 三教师 OPD 有两版：`opd_three_teacher_145m`（code/tool 用 SFT teacher）和
  `opd_three_grpo_145m`（三个 teacher 都是 GRPO final）。两版 20 step 的 KL
  total 都在 0.065 左右、没有单调下降；lr 5e-6、20 step 远不足以判断收敛。
  tool teacher 的 KL 最大（0.098），说明 tool specialist 与 student 分布差最大。
- code/tool GRPO 只跑了 10 step：reward 仍很低（code −0.085、tool −2.24），
  不能说明 specialists 变强；它们作为 OPD teacher 的“格式/行为来源”仍可用。

### 7.2 磁盘

后训练阶段只需要 `data/sft_*.jsonl` 与 `data/tokenization/sft_*`。
hn01 上从旧机带过来的 `pretrain_*` tokenization（26 GB）已删除；预训练
pack 仍在 hb01；需要时可从 JSONL 重新生成。中间 checkpoint 只保留
final/base：清理后数据盘 **32 GB free**、系统盘 **27 GB free**。
`/tmp` 下的 smoke checkpoint 每次 ~1 GB，测完即删；正式 run 的
`keep_last_ckpt` 建议保持 2。
