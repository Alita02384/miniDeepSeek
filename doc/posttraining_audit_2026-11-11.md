# 后训练代码审计（2026-11-11）

> 范围：SFT / GRPO（含 CISPO）/ 多教师 OPD / rollout（torch + KV-cache）/ reward /
> eval / pipeline / configs。**预训练进程与预训练配置未改动**；hb01 上的训练 job
> 未触碰。
>
> 审计基线代码：`bf6a670`；本文档的 commit：`e6b3c13`（审计）、`84721bf`（PROCESS 记录）。
> 服务器实跑数据：hn01 上已有的 `checkpoints/{sft,grpo,opd}_*_145m` 与
> `data/tokenization/sft_*`。
>
> 方法：逐文件阅读 + 与 `verl` 参考实现对照 + 在 CPU debug 模型上写小实验验证
> 可复现的怀疑；对已有实跑 metrics / 数据做统计复核。本文只记录**有证据**的
> 问题；未复现的猜测单独标为“未验证”。

---

## 0. 结论摘要

| 级别 | 数量 | 说明 |
|---|---:|---|
| **P0** | 1 | 多卡 SFT（以及共享 `Trainer` 的其它 stage）在 DDP/FSDP2 下，各 rank 用**自己的** target token 数做梯度缩放，rank 间权重会发散 |
| **P1** | 15 | tool 多轮 continuation 的模式/上下文/open_thinking/max_turns、GRPO 多卡配置不可运行、eval 路径/指标/`tests` 丢失、SFT 数据取文件前缀 + 训练量不足、SFT 域混料按 input token、reward clip 饱和、tool reward 的 unfinished/alignment/shaping、pipeline 与实跑路径脱节、HCA 长上下文 decode/forward 不一致 |
| **P2** | 20+ | 指标只在 rank 0、无 stop-rate、OPD teacher renormalize、GRPO per-rank token 归一化、3-teacher pipeline 实际 2 teacher、温度口径、resume、数据校验、reward/eval 小口径、文档/基准 checkpoint 等 |
| **P3** | 10+ | 指标口径、重复保存、`.gitignore` 符号链接、计算浪费等 |

**最重要的三条**：

1. **多卡 SFT 的梯度缩放 bug（P0）**：`trainer.py:845-853` 先
   `all_reduce_gradients`（求平均），再用**本 rank** 的 `target_tokens` 去除。
   target 数不同则各 rank 的缩放因子不同，参数在一步后就不同。2 进程 CPU
   debug 模型可复现（见 §3.1）。单卡不受影响；hb01 的预训练如果每个 rank 的
   target 数相等则也恰好无影响（满 pack）。
2. **tool 多轮 continuation（P1）**：它跑在 `model.train()` 模式下
   （`cached_tool.py:89-119` 位于 `super().rollout()` 恢复 train 之后），
   eager decode 会污染 MoE `tokens_per_expert_E`；如果 batch=1 的 CUDA graph
   恰好在此时首次捕获，counter 的加法会被录进图，之后每次 replay 都污染。
   同时 continuation 的每轮生成长度没有按剩余 context 截断，长 prompt +
   observation + 768 token 会直接 RoPE 越界崩溃（`IndexError`）。
3. **SFT 训练数据是文件前缀切片**：`build_sft_dataset` 的 `seed` 参数从未
   被使用，`max_records` 按文件顺序取前 N 条。general 只用了 877,651 条里的
   50,000 条（一个 86% 多轮、单一 prompt 重复 5.5% 的块），code 的 30,000 条
   只有 `magicoder_oss`/`self_oss_instruct`（漏掉 62,864 条 `codefeedback`），
   tool 8,000 条只有 5,414 个唯一会话。叠加训练步数后实际只有
   **0.41 / 0.48 / 1.39 epoch**，不是文档写的“约 1 epoch”。

另外还有一个容易被忽略的 P1：general/tool 的 `reward_clip=3.0` 低于
“RM(±3) + shaping(最大 +1.75)”的上限，**好组会被 clip 成完全相同的 3.0**，
`group_advantages` 返回全 0，梯度静默消失（§4.10）。

---

## 1. 代码同步状态（本地 / GitHub / hb01 / hn01）

### 1.1 本次实际执行

| 位置 | 审计前 | 审计后 | 操作 |
|---|---|---|---|
| 本地 `main` | `2cfb34c`（+ 并发会话的未提交 dashboard/pretrain 修改） | 代码基线 `bf6a670` + docs commit（`e6b3c13`/`84721bf`/`9e28786`） | 只读/写文档 |
| `origin/main` | `ad66d08` | 代码基线 `bf6a670` + docs commit（`9e28786`） | `git push origin main`（正常 fast-forward，无 force） |
| hn01 | `c946f5b`，`.venv` 外无改动 | `bf6a670`（代码；docs 同步的 fetch 被网络中断） | `git fetch` + `git merge --ff-only origin/main` |
| hb01 | `e30ef7c`（工作树 14 个 modified + 5 个 untracked） | 未动 | 只读检查；训练 job 未触碰 |

### 1.2 hb01 的真实状态

- `HEAD=e30ef7c`，比审计基线代码 `bf6a670` 落后 **33 个 commit**（当前 `main` 另有 2 个 docs commit）。
- 工作树有 14 个 modified 文件，其中 **11 个与本地 `main` 逐字节相同**
  （`model/wrapper.py`、`train/trainer.py`、`train/distributed.py`、
  `train/metrics.py`、`train/run.py`、`model/routing.py`、`viz/report.py`、
  对应 tests 等）；剩下 3 个 pretrain 配置/脚本与本地 `main`
  **去掉注释后完全一致**（`configs/pretrain/pretrain_145m.yaml`、
  `configs/pretrain/curriculum_145m_4x4090.yaml`、
  `scripts/build_pretrain_packs.sh`）。差异只是 commit/decision 编号的注释
  （hb01 写 D82/D83/D84，本地写成 D85/D86/D87）。
- 结论：**hb01 未提交的 pretrain 工作已经全部包含在本地 `main` 里**；不需要
  为它单独保留 patch。但 hb01 的 `src/minideepseek/rl/` 是**旧版本**
  （缺少 KV-cache engine、cached tool、CISPO 等），不能直接在 hb01 上跑当前
  后训练。
- 建议：等当前 pretrain job 结束、确认 checkpoint 落盘后，在 hb01 上
  `git stash`（或先把 WIP 提交到本地分支）→ `git fetch origin` →
  `git merge --ff-only origin/main` → 再决定是否 `stash pop`（会只剩注释冲突，
  可直接丢弃）。**不要在训练进行中 pull**，避免下次 auto-resume 用混合版本。
- hb01 的 `configs/pretrain/generated/curriculum_145m_g2.yaml` 是
  `scripts/autoconfig_145m.py` 生成物，`.gitignore` 已忽略；不在版本控制内。

### 1.3 两个小问题

- 服务器上的 `.venv` 是指向 `/opt/mds-venv` 的**符号链接**；`.gitignore`
  写的是 `.venv/`（只匹配目录），所以 `git status` 一直显示 `?? .venv`。
  建议改成 `.venv`（不带斜杠）或 `.venv*`，避免误提交。
- hn01 在同步前，后训练相关代码（`rl/`、`configs/{sft,grpo,opd}`、
  `tests/test_{grpo,opd,cache,cached_tool,tool_agent}.py`）与本地**已经逐字节
  相同**；差异只在 `trainer.py`/`metrics.py`/`viz/report.py` 和 pretrain
  configs。同步后完全一致，`pytest` 后训练 34 项全过。

---

## 2. 验证环境与已跑命令

```bash
# 本地
.venv/bin/python -m pytest tests/test_grpo.py tests/test_opd.py \
  tests/test_cached_tool.py tests/test_cache.py tests/test_tool_agent.py -q
# -> 34 passed

# hn01（同步后，CPU）
.venv/bin/python -m pytest tests/test_grpo.py tests/test_opd.py \
  tests/test_cached_tool.py tests/test_cache.py tests/test_tool_agent.py -q
# -> 34 passed

# 2 进程 DDP smoke
.venv/bin/python scripts/test_rl_distributed.py --method grpo --dist-mode ddp --rollout-backend cached
.venv/bin/python scripts/test_rl_distributed.py --method opd  --dist-mode ddp --rollout-backend cached
# -> PASS grpo / PASS opd
```

未跑全量 `pytest`（本地约 313 项，包含 pretrain/dashboard 的大测试）；未做真实
多卡 GPU 训练（hn01 是单卡，hb01 正在预训练）。

---

## 3. P0：多卡 SFT 的梯度缩放会让 rank 间权重发散

### 3.1 位置与机制

`src/minideepseek/train/trainer.py:845-853`：

```python
if self.dist.enabled and self.dist.mode == "ddp":
    all_reduce_gradients(self.model.model, self.dist.world_size)   # 求平均
# ...
target_tokens = sum(int(metric.tokens) for metric in micro_metrics)  # 本 rank
if target_tokens > 0:
    scale_gradients(self.model.model, 1.0 / target_tokens)          # 本 rank 的因子
```

`all_reduce_gradients` 是 `SUM / world_size`。所以每个 rank 的梯度是
`global_sum / world`，然后再除以**自己的** target token 数。当
`target_tokens` 不同，两个 rank 的最终梯度相差 `target_B / target_A`，参数在
一步后就不同；DDP 没有额外的参数 broadcast，下一步继续在发散后的权重上算
梯度，误差累积。

SFT 的 response-only mask 让每个 rank 的 target 数天然不同（general 的 target/input
≈ 0.86，code ≈ 0.37，tool ≈ 0.18，且每个 micro-batch 的响应长度不同）。这不是
统计口径问题，是**DDP 正确性**问题。

### 3.2 复现（CPU debug 模型，2 进程 Gloo）

脚本：`/tmp/ddp_scale_probe.py`（临时文件，不提交；核心逻辑见下）。两个 rank
用同一个 debug 模型、同一个 batch 序列，只让 target mask 的 True 数不同：

```python
n_target = 18 if rank == 0 else 4     # target token 数不同
# ... Trainer(...).train_step()
dist.all_gather_object(gathered, model.state_dict())
max(|rank0 - rank1|)
```

结果：

```text
targets 18 vs 4   -> max weight diff = 2.127e-05, 79 / 106 params differ
targets 18 vs 18  -> max weight diff = 0.0,     0 / 106 params differ
```

同一个模型、同一个数据，只因为 target 数不同，一步后就分叉。lr=1e-2 用于放大；
真实 SFT lr=2e-5，单步差会小两个数量级，但每步都会再放大一次，且 120 步
后 `final` 只保存 rank 0 的权重，rank 1 的差异不会报错。

### 3.3 影响范围

- **单卡**：`dist.enabled=False`，走 `1/target_tokens`，没有第二个 rank，不受影响。
  所以 hn01 上的现有 SFT/GRPO/OPD 结果不受此 bug 影响。
- **多卡 SFT**：`configs/sft/*.yaml` 都是 `distributed.mode: ddp`；`train_sft.sh`
  在 `NPROC_PER_NODE>1` 时启用。**会命中**。
- **多卡预训练**：如果每个 rank 的 batch 是满 pack（`batch_tokens/seq_len` 个
  文档），每个 rank 的 target 数相同（= batch_tokens − 文档数），则缩放因子恰好
  相同，**当前预训练大概率未受影响**。但只要 pack 填充率/文档数不同（例如
  `full_packs` 重切、某个 rank 的尾部 pack 不满），就会命中。修复对“target 数
  相同”的情况是 no-op，因此不需要在修复时停预训练。
- **FSDP2**：梯度在 backward 内已由 FSDP2 归约，之后仍用本 rank 的
  `target_tokens` 缩放；同样的发散问题，只是归约发生在更早的位置。

### 3.4 最小修复

对 DDP 路径，把 token 数也做 all-reduce，并按**全局** token 数缩放：

```python
local_tokens = sum(int(m.tokens) for m in micro_metrics)
if self.dist.enabled:
    count = torch.tensor(float(local_tokens), device=self.device)
    self.dist.all_reduce_sum(count)          # 已用于 expert usage，API 存在
    global_tokens = float(count)

if self.dist.enabled and self.dist.mode == "ddp":
    all_reduce_gradients(self.model.model, self.dist.world_size)   # mean
    factor = self.dist.world_size / max(global_tokens, 1.0)
elif self.dist.enabled:                                          # fsdp2：梯度已平均
    factor = self.dist.world_size / max(global_tokens, 1.0)      # 需确认 FSDP2 是 AVG
else:
    factor = 1.0 / max(local_tokens, 1.0)
scale_gradients(self.model.model, factor)
```

`global_tokens == local_tokens * world` 时，`factor = 1/local_tokens`，与旧行为
逐位等价（前提是参数初始一致），所以对满 pack 预训练是 no-op。FSDP2 的分支
需要先确认 `fully_shard` 的 reduce 是 AVG 还是 SUM（torch 默认 AVG；若为 SUM，
factor 应为 `1/global_tokens`）。建议同时加一个 2 进程回归测试：
**故意给两个 rank 不同数量的 target token，跑一步后 burst state dict 必须逐位
相同**。

**这项修复在共享的 `trainer.py`，会触及预训练代码。当前 hb01 的 job 正在跑，
本地 `trainer.py` 还有并发会话的未提交修改，所以本次只写进报告、没有直接改。**
建议在预训练 checkpoint 落盘后、或经你确认后，单独一个 commit 修。

---

## 4. P1：会跑错 / 跑挂 / 结果不可信

### 4.1 tool 多轮 continuation 跑在 train 模式，污染 expert usage

位置：`src/minideepseek/rl/cached_tool.py:89-119`（在 `super().rollout()` 之后），
`src/minideepseek/rl/cached_rollout.py:174-175, 243-247`（`was_training=...;
eval(); ... finally: train()`）。

`super().rollout()` 结束时把模型恢复成 `train()`；随后 `_continue_tool` 的
`_feed()`（`cached_tool.py:123-141`）直接 `self.model.decode_step(...)`，没有
再切 eval。`MoE.Router.forward` 里 `if self.training: tokens_per_expert_E.add_(...)`
（`model/moe.py:114-116`）会把这些 observation/generation token 计进 expert
usage。最终 `GRPO/OPD` 的 `_update_expert_biases()`（`losses.py:128-145`）用这些
计数更新 router bias，等于把 rollout token 分布混进训练 token 的负载均衡。

复现（CPU debug 模型，直接调用 `_continue_tool`，模拟 `super().rollout` 返回后
的 train 状态）：

```text
training: True -> True
counter sums per MoE layer: [0,0,0,0] -> [369,369,369,369]
```

更隐蔽的是 CUDA graph：`GraphDecodeRunner._capture()`（`graph_decode.py:60-77`）
在**首次遇到 bucket=1** 时捕获。tool continuation 正是 batch=1；如果 bucket 1
尚未在 `super().rollout()` 的 eval decode 中捕获过，它会在 train 模式下捕获，
`tokens_per_expert_E.add_` 被录进图，之后每次 replay 都污染计数。CPU 测试和
`use_cuda_graph=False` 的路径直接污染；GPU 路径取决于 bucket 1 是否已被捕获，
行为不确定。

修复：

```python
# CachedToolRolloutEngine.rollout 的 tool 分支
was_training = self.model.training
self.model.eval()
try:
    # 所有 _continue_tool(...) 调用
finally:
    if was_training:
        self.model.train()
```

（`_feed` 内部也可断言 `not self.model.training`。）补一个测试：continuation
后所有 MoE 的 `tokens_per_expert_E` 仍为 0，且 `model.training` 与调用前一致。

### 4.2 tool continuation 没有 context 预算，会越界崩溃

位置：`cached_tool.py:223-252`。`_feed` 把 `prefix[cache_len:] + observation_ids`
一次性喂进去，没有任何 `max_context` 检查；生成循环固定跑 `max_new_tokens`，
也不看剩余 context。`cache.length[slot]` 可以超过 `RolloutCache.max_context`，
最终在 `rope.py:83` 抛 `IndexError: index N is out of bounds for dimension 0
with size 128`（我们用一个 seq_len=64 的 debug 模型复现了同样的错误）。

这不是理论风险：hn01 上 `data/grpo_tool_train.jsonl` 的 prompt 长度
（真实 tokenizer，前 2,000 条）

```text
max = 2590, p99 = 2323
```

而 `configs/grpo/tool.yaml` 是 `max_new_tokens: 768`、`max_turns: 3`。只算
prompt + 3 轮生成 = 2590 + 2304 = 4,894 > 4,096，还没算 observation。
OPD 的 tool prompt 最大 1,754，加上 3×512 也在边缘。当前 10-step tool GRPO
没崩，是因为生成提前停或命中了较短的 prompt，并不是代码有保护。

修复（最小）：

- 在 `_continue_tool` 里每轮开始前算
  `remaining = self.max_context - int(cache.length[slot])`；
- observation 超过 remaining 时截断（并标记 `unfinished=True`），或直接结束该轨迹；
- 每轮生成 `turn_limit = min(max_new_tokens, remaining - len(obs))`，`<=0` 就停；
- `_feed` 开头断言 `start_pos + len(tokens) <= self.max_context`，给清晰报错；
- 回归测试：用小 `max_context` + 长 observation，断言不抛 IndexError、轨迹被
  标记 unfinished。

### 4.3 GRPO 的三份 config 在 `world_size>1` 下无法运行

`configs/grpo/{general,code,tool}.yaml` 都是：

```yaml
rl:
  rollout_backend: cached
distributed:
  mode: fsdp2
```

`build_grpo_trainer`（`rl/grpo.py:521-530`）在 `context.enabled` 且
`rollout_backend in {cached,kv}` 且 `mode != ddp` 时直接
`raise ValueError("... use distributed.mode=ddp ...")`。单卡
（`WORLD_SIZE=1`，`context.enabled=False`）能跑，所以现有单卡结果没问题；
但 `NPROC_PER_NODE=2` 会立刻报错。`train_grpo.sh` 的注释还写着
“torchrun + FSDP2 when NPROC_PER_NODE>1”，`README` 又写多卡 DDP，三处互相矛盾。

修复：把三份 GRPO config 改成 `distributed.mode: ddp`（与 OPD configs 一致），
修正 `train_grpo.sh` 注释；如果要保留 FSDP2，需要先实现 gathered rollout 副本，
不是改个 config 能解决的。

### 4.4 eval 配置的 checkpoint 路径全是旧的

hn01 上实际存在的是 `checkpoints/grpo_{general,code,tool}_145m/final`，但：

```text
configs/eval/general.yaml:9  checkpoint: checkpoints/grpo_general/final   -> 不存在
configs/eval/code.yaml:9     checkpoint: checkpoints/grpo_code/final      -> 不存在
configs/eval/tool.yaml:9     checkpoint: checkpoints/grpo_tool/final      -> 不存在
```

`scripts/eval.sh configs/eval/general.yaml` 会直接 `FileNotFoundError`。
`configs/pipeline/*.yaml` 与 `doc/server_runbook.md:80` 也有同样的旧路径。
修复：统一改成 `_145m` 后缀，或改成 eval 的统一入口（见 4.5）。

### 4.5 eval 对 general/tool 用了错误的 reward

`src/minideepseek/eval/evaluate.py:143-149`：

```python
if config.domain == "math" or record.verifier == "math_exact_match":
    result = math_reward(...)
elif config.domain == "code" or record.verifier == "unit_test":
    result = code_reward(...)
else:
    result = world_reward(...)          # 只对 world 域正确
```

`configs/eval/general.yaml` 的 `reward.kind: general` 且
`reward_model_path: internlm2-1_8b-reward`，`configs/eval/tool.yaml` 的
`reward.kind: tool`，但两者都走到 `world_reward` 的 heuristic judge：
general 的 1.8B RM 从未被加载，tool 的 alignment/gt/unfinished 规则从未被使用。
`reports/eval_*.json` 的数字与 GRPO 训练 reward 不可比，不能用来判断 specialist
是否变强。

修复：让 eval 复用 `rl.reward.make_reward_fn(settings)` + `dispatch_reward`，
general 加载 RM，tool 走 `tool_reward`（注意 eval 目前不执行 tool，因此 tool
指标含义要单独定义，至少不要假装是完整 agent 评估）；math/code 保持现状。
加一个 `test_eval_general_uses_reward_model` / `test_eval_tool_uses_tool_reward`
（可用注入的 fake reward_model 避免加载 1.8B）。

### 4.6 SFT 取文件前缀，`seed` 完全没用

`src/minideepseek/data/sft.py:128-131`：

```python
for path in paths:
    for record in iter_jsonl(path):
        if max_records is not None and used >= max_records:
            break
```

`seed: int = 1234`（`sft.py:116`）只被写进 meta（`sft.py:169`），**没有 shuffle、
没有采样、没有去重**。实际数据：

| 文件 | 全文件条数 | 构建使用 | 占比 | 前缀的问题 |
|---|---:|---:|---:|---|
| `sft_general.jsonl` | 877,651 | 50,000 | 5.7% | 前 50k 有 42,950 条多轮（86%）；一个 prompt「你今天过得怎么样？」出现约 2,745 次（5.5%）；首条 message 唯一率 53%（50k 之后约 85%）；100k–500k 的单轮长回答块完全没被用到 |
| `sft_code.jsonl` | 148,723 | 30,000 | 20.2% | 只有 `magicoder_oss` 24,753 + `self_oss_instruct` 5,247；`codefeedback` 62,864、`python_code_instructions_18k` 10,485、`leetcode`/`mbpp` 全部没进 SFT |
| `sft_tool.jsonl` | 99,552 | 8,000 | 8.0% | 8,000 条只有 5,414 个唯一会话（32% 完全重复）；全文件 66,000/99,552 唯一 |

这解释了为什么 `sft_general` 的生成很偏（大量自我/身份式回答、重复）而
`code`/`tool` specialist 弱：不是模型容量先到顶，而是**数据只看了文件的一个
连续块**。`seed` 参数看起来像“随机采样”，实际是摆设。

修复：

- 构建前用 `random.Random(seed)` 打乱行号，或按 `source` 分层/round-robin 抽样；
- 去重（至少完全相同的 conversation）；
- 把选中的行号范围/哈希写进 meta，保证可复现；
- code 数据如果要用 `leetcode`/`mbpp`，必须先与 eval 集做 decontamination
  （当前 GRPO code train 与 eval code 无精确 prompt 重叠，但要按这个问题重查）。

### 4.7 SFT 训练量不足：0.41 / 0.48 / 1.39 epoch

`configs/sft/{general,code,tool}.yaml`：`steps=120/90/80`，
`batch_tokens=8192`，`grad_accum=16`，即 **131,072 input token/step**；
三个 config 的 domain 权重是 0.90/0.75/0.70。用 tokenized meta 和实跑
`metrics.jsonl` 的 target token 数核对：

| 专家 | 实跑 target token | 数据 target token | target epoch | input epoch（subagent 复核） |
|---|---:|---:|---:|---:|
| general | 12,127,481 | 28,421,650 | **0.427** | 0.410 |
| code | 5,593,745 | 6,580,522 | 总 target 中 code 约 56.7% → **0.482** | 0.482 |
| tool | 3,862,197 | 916,652 | 总 target 中 tool 约 32.9% → **1.386** | 1.389 |

`doc/posttraining_plan.md:38-40` 写“120 步（general/code）/80 步（tool），约
1 epoch”，与实测不符；`doc/specialist_data_145m.md:93-99` 更旧（写的是
500×32,768×8=131M token/专家，实际是 15.7/11.8/10.5M input token），差约 10 倍。

按当前每步 target 数，做到 1 epoch 约需：

- general：`28.42M / 101.1k ≈ 281` 步（config 120）
- code：`6.58M / 35.3k ≈ 186` 步（config 90）
- tool：1 epoch ≈ 58 步；文档建议 2–3 epoch → 116–174 步（config 80）

**建议**：把 steps 提到约 280/190/120，或相应提高 `batch_tokens`；先把 4.5 /
4.6 修好再重训，否则只是把同一批前缀数据多看几遍。

### 4.8 SFT 的域混料按 input token 加权，不是按 loss token

`response-only mask` 让不同域的 `target/input` 差很多（general 0.86、code 0.375、
tool 0.181）。`data.sources` 的 weight 是 **input token** 权重。折算到实际
loss token：

| config | 名义 input 权重 | 实际 loss-token 占比 |
|---|---|---|
| general | 0.90 general / 0.07 code / 0.03 tool | **96.1% / 3.3% / 0.7%** |
| code | 0.75 code / 0.25 general | **56.7% / 43.3%** |
| tool | 0.70 tool / 0.30 general | **32.9% / 67.1%** |

所以 `sft_tool` 名义 70% tool，实际只有 33% 的 loss token 来自 tool；general
specialist 名义 7% code + 3% tool，实际 4% 左右。如果目标是“tool 专家主要学
tool”，现在不是。

修复：要么在混料时按每个 source 的 target/input 比例折算权重，要么在
`iter_batches` 里按 target token 数配额（而不是 input token），并在 meta/日志
里记录实际 loss-token 比例。

### 4.9 pipeline 与当前实跑路径脱节

`src/minideepseek/pipeline.py` + `configs/pipeline/*.yaml` 是旧命名/旧逻辑，
`doc/posttraining_plan.md` 也承认 `run_pipeline.sh` 未完全对齐，但具体问题：

- `_stage_done`（`pipeline.py:234`）找
  `checkpoints/curriculum_145m/stage_02_seq8192/final`；当前 3B 预训练是
  `curriculum_145m_3b/stage_00_seq2048/...`，pipeline 会认为预训练没完成。
- `_sft_stage`（`pipeline.py:160-181`）从**单个** `sft_<name>.npz` 的
  `_meta_tokens` 算 steps，但 override 了 `data.tokenized_path`，而
  `run.py:164` 在 `data.sources` 存在时**优先用 sources、忽略 tokenized_path**。
  于是 steps 按一个文件算，训练却用 sources 混料。
- `_preflight`（`pipeline.py:258-270`）对 OPD 要求
  `data/opd_prompts_<teacher_key>.jsonl`（如 `opd_prompts_code.jsonl`），
  当前实际是 `data/opd_prompts_three.jsonl`。
- pipeline configs 的 `resume`/`out_dir` 是 `checkpoints/sft_general` /
  `grpo_general` / `curriculum_145m` 等旧名，实际是 `*_145m` / `_3b`。
- `_sft_stage` 的 `tokens * epochs / (batch*grad_accum)` 不计 SFT 的
  target/input 比例（对 tool 会高估 steps）。

修复：要么把 pipeline 对齐到当前命名（`_145m`/`_3b`、`opd_prompts_three`、
按 sources 算 steps），要么明确标注 `run_pipeline.sh` 仅用于旧 2-teacher 流程，
README/文档不要把它当一键复现入口。建议加一个 `--dry-run` 断言：所有
resume/out_dir/data 路径存在，否则 preflight 直接失败。

### 4.10 reward clip 饱和：好组的 advantage 被静默清零

位置：`src/minideepseek/rl/reward.py:296-300`：

```python
if settings.reward_clip > 0:
    clipped = max(-settings.reward_clip, min(settings.reward_clip, total))
```

`configs/grpo/general.yaml` 与 `tool.yaml` 都是 `length_bonus: 0.5`、
`think_bonus: 1.0`、`reward_clip: 3.0`。RM 本身已经把分数夹在 [-3, 3]
（`rl/reward_model.py:62-63`），shaping 最多再加
`0.5 (length) + 1.0 + 0.25 (think)` = **+1.75**。因此没有 repetition penalty
时 RM ≥ 约 1.25 就会被压成 3.0；即使有 -0.5 的 repetition penalty，阈值也只有
约 2.25。**好组的 reward 会变成完全相同的 3.0**；
`group_advantages`（`reward.py:448-460`）对常数 group 返回全 0，这一组
**没有任何梯度**。

实测（真实 `dispatch_reward` + 假 RM，shipped general 设置，response 的
repetition penalty 为 -0.5）：

```text
RM 2.0 -> total 2.75            (pre_clip 2.75)
RM 2.9 -> total 3.0             (pre_clip 3.65)
RM 3.0 -> total 3.0             (pre_clip 3.75)
group_advantages([3,3,3]) = [0,0,0]
```

当前 general GRPO 50 步的 reward mean 是 −1.8 ~ −2.2，还没有大量饱和；
但模型变好（或换 RM/数据）后，**最好的组会先停止学习**，而且 metrics 会显示
`reward_mean=3.0, reward_std=0, grad_norm≈0`，看起来像“收敛”，实际是
clip 抹掉了组内差异。tool 配置同样。

修复（任选）：

- clip 只作用于 RM 分数，再叠加 shaping；或
- 把 general/tool 的 `reward_clip` 提到 ≥ 5.0（RM ±3 + shaping 1.75）；或
- shaping 不计入 clip；
- 加测试：RM 分数 2.0/2.9/3.0 的组，`group_advantages` 不能全 0。

### 4.11 tool continuation 的 `open_thinking` 推断错误，且 `max_turns` 少一轮

**(a) `open_thinking`**：`cached_tool.py:200-210` 用“re-render 后前缀是否等于
`prefix`”来猜第一轮用的是 open 还是 closed thinking：

```python
for open_thinking in (True, False):
    full_ids = self.tokenizer.encode(render_chat(..., open_thinking=open_thinking))
    if full_ids[:len(prefix)] == prefix:
        observation_ids = full_ids[len(prefix):]
        break
```

问题：两种 render 的差异在 **generation prompt/observation tail**，不在 prefix
（prefix 到 response 结束就截断了）。用真实 tokenizer 验证：

```text
open_thinking True  prefix matches: True  tail '\n<|im_start|>assistant\n<think>\n'
open_thinking False prefix matches: True  tail '\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

两个候选都匹配，循环**永远选 True**。tool/OPD 的 `thinking_ratio` 是 0.1：
当第一条 prompt 实际是 closed thinking 时，continuation 会以 open-thinking
的 scaffold 继续，轨迹不是 chat template 对这段对话的渲染；`ToolRolloutEngine`
（reference）用的是记录下来的 flag，所以两条引擎不一致。若 `max_turns=4`
（需要第二次 observation feed），第二次 re-render 会直接
`RuntimeError: multi-turn chat template is not prefix-stable`。shipped
tool config 是 `max_turns=3`，只错 scaffold 不崩，但会污染 tool 轨迹。

修复：在 `CachedRolloutEngine.rollout` 生成 `prompts` 时把 `open_thinking`
记录到 `RolloutSample`（或返回给 tool 分支），continuation 直接使用它，不要
trial-render。

**(b) `max_turns` off-by-one**：`cached_tool.py:175` 是
`for turn in range(1, self.max_turns)`，循环里先 `parse_tool_calls`、执行 tool，
再 `if turn >= self.max_turns - 1: unfinished=True; break`。第一次 response
是 `super().rollout` 生成的（turn 0），所以 `max_turns=3` 时实际只生成
**2 个 assistant turn**（turn 0 + turn 1），turn 2 的 tool call 执行完就
break，**不生成最后回答**；reference `agent_rollout.py:134
for turn in range(self.max_turns)` 会生成 3 个。结果：cached path 的
`unfinished` 更频繁、`tool_turns` 少一轮，reward 的 unfinished penalty 与
reference 不可比。

修复：把“最后一轮也生成”的 loop 结构与 reference 对齐（先 generate，再
parse/execute，最后才判断 unfinished），并加一个“模型总是调用 tool”的
max_turns=2/3 对照测试。

### 4.12 tool reward：unfinished 仍然给 gt 分，alignment 用答案个数当调用数

`reward.py:360-400` 的 `tool_reward` 在 `unfinished=True` 时只扣
`tool_unfinished_penalty=0.5`，**仍然计算并加上 `gt_match`（权重 2.5）**。
MiniMind 的 `train_agent.py` 在 unfinished 时把 `final_text` 置空，跳过 gt。
实测（构造 rubric `tool_turns` 含一个有效调用 + 最终答案，`unfinished=True`）：

```text
unfinished total 0.5  components: tool_alignment -1.0, gt_match 2.5, unfinished -0.5
finished   total 1.0
```

即“工具没走完但猜中答案”仍拿到 2.5，远大于 0.5 的截断惩罚；应该
`if unfinished: 跳过 gt（以及 think/length shaping）`。

`reward.py:387`：

```python
gap = abs(valid_calls - len(gt)) + max(0, len(calls) - valid_calls)
```

`gt` 是**答案列表**，不是“期望调用次数”。`data/grpo_tool_train.jsonl` 里
24,119/33,960 条记录 `len(gt)>=2`（例如天气问题有两个答案值，却只需要
一次 `get_current_weather`）。于是 `len(gt)=3` 时“一次正确调用 + 正确
答案”会被 `gap=2` 罚成负分；`len(gt)=1` 同样回答却是 +3.0。应增加
`expected_calls` 字段，或只惩罚 invalid/extra call，不要拿 `len(gt)` 当调用数。

另外，`dispatch_reward` 在 tool 分支之后仍会走 `_finalize`，而
`configs/grpo/tool.yaml` 设了 `length_bonus:0.5 / think_bonus:1.0 /
repetition_penalty:0.5`；MiniMind 只在 no-tool 分支加 length/think。实测
“错误答案 + 64 字 think”得到 2.25，其中 shaping 1.75、alignment 0.5，
**shaping 占 gt 权重（2.5）的 70%**。建议 tool 有调用时不做 length/think
shaping。

### 4.13 eval code 的 `tests` 字段被 load_records 丢掉 → pass_rate 恒 0

`data/eval_code.jsonl` 的记录是**顶层** `tests`：

```text
keys ['task_id', 'prompt', 'tests', 'domain', 'source', 'license']
```

`data/grpo.py:206-228` 的 `load_records`（以及 `convert_record`）只保留
`rubric=item.get("rubric") or {}`，**不把顶层 `tests` 放进 rubric**；eval_code
记录甚至没有 `verifier`，会走 `convert_record` 的默认 `verifier="none"`。
`code_reward` 只读 `record.rubric["tests"]`（`reward.py:144`）。实测：

```python
r = load_records("data/eval_code.jsonl")[0]
# rubric keys [], verifier none
code_reward(code, r, RewardSettings(kind="code"))
# {'total': 0.0, 'details': {'error': 'record has no rubric.tests'}}
```

所以 `configs/eval/code.yaml` 的 `pass_rate` 永远是 0，和模型无关。
修复：`convert_record`/`load_records` 把顶层 `tests`/`test_code` 映射进
`rubric`，并把 verifier 设成 `unit_test`；补 eval code 用例。

### 4.14 HCA decode 与 packed forward 在 `max_context > seq_len` 时不一致

`attention.py:402-411` 的 packed 前向：HCA 只保留**最旧**的
`ceil(max_doc_tokens/ratio)` 个压缩块（代码注释写“keep the most recent
entries”，实际 `torch.where(cmp < limit, cmp, -1)` 选的是 0..width-1）。
`attention.py:564-568` 的 decode：`k = max_comp`，`valid = selected < n_comp`，
看到的是**所有**已完成块。当 `max_context=1024`、`seq_len=256`、prompt=640
时，decode 看到 5 个块、packed forward 只看到 2 个，logits 最大差 **0.134**
（测试容差 1e-5）。

当前 145M RL 的 engine 用 `max_context=int(config.model.seq_len)=4096`
（`grpo.py:570`、`opd.py:507`、`backends/__init__.py:30-34`），所以
`max_context == seq_len` 时两者一致；**一旦用 8K 上下文做 eval/长上下文
rollout，或直接 `CachedRolloutEngine(max_context=None)`（默认
model.max_context_length=8192），就会命中**。`test_cache.py` 全部用
`max_context==seq_len`，掩盖了它。

修复：把 HCA 的窗口改成“最近 width 个块”，并让 decode 使用同一个 width；
或者从缓存构造时强制 `max_context == seq_len` 并在超长时明确报错。补一个
`max_context>seq_len` 的对照测试。

### 4.15 pipeline 的具体断点（比 §4.9 更细）

除 §4.9 的命名/override 问题外：

- **pretrain 数据布局**：`pipeline.py:105` 读 `data/pretrain_t2t.jsonl`（不存在；
  实际是 `data/pretrain_{general,code,tool}.jsonl`）；`pipeline.py:131` 找
  `pretrain_{2k,4k,8k}.npz`（实际是
  `pretrain_{general,code,tool}_{2k,4k,8k}.npz`）。`_pretrain_stage` 实测
  直接 `SystemExit: missing tokenized pretrain datasets`。
- **pretrain 预算 2×**：`pipeline.py:144`
  `grad_accum = source.get("train", {}).get("grad_accum", 8)`；curriculum YAML
  没有顶层 `train`，于是用 fallback 8，而 base config 是 16
  （`pretrain_145m.yaml:54`）。3B token 会算成 3433/12588/6866 步，而不是
  1717/6294/3433，**多训一倍**。
- **SFT 数据缺失**：`pipeline.py:114` 默认 `sft_domains=[general, tool]`，
  但 145M 的 `configs/sft/general.yaml` 混了 7% `sft_code.npz`；pipeline
  自己不会生成它，SFT general 会因缺 `sft_code.npz` 失败。
- **OPD student/teacher 不一致**：pipeline configs 让 OPD 从
  `checkpoints/curriculum_145m/stage_02_seq8192/final`（pretrain）初始化；
  OPD configs 实际是 `checkpoints/sft_general_145m/final`；真实 OPD 跑的是
  `sft_general final`（`doc/PROCESS.md:1480`）。teacher 路径
  `grpo_*_145m/final` 与 pipeline GRPO 写出的 `grpo_*` 也不一致；
  `_rl_stage` 只在 stage 显式有 `teachers` 时才 override（`pipeline.py:194`），
  pipeline configs 没有 → `FileNotFoundError`。
- **3-teacher 实际只有 2 个**：pipeline 3teacher 的 OPD stage 用
  `data/opd_prompts_mixed.jsonl`（只有 general 17,460 + tool 19,536），
  没有 code；`configs/opd/three_teacher.yaml` 的
  `data/opd_prompts_three.jsonl` 在 hn01 数据盘上存在（gitignore 不跟踪），
  但 pipeline 没有用它。code teacher 权重 1/3 永远不会被路由，叠加 §5 P2-1
  的 renormalize，实际是 0.5/0.5。
- **preflight 不完整**：只检查 SFT JSONL、GRPO/eval data、以及过期的
  `opd_prompts_<key>.jsonl`；不检查 pretrain 输入、SFT npz、GRPO resume、
  teacher checkpoint。`_stage_done` 硬编码 `stage_02_seq8192`。

结论：`run_pipeline.sh` 现在**不能**端到端复现 145M 后训练；README /
posttraining_plan 应把它标成“旧 2-teacher 模板”，实际入口用
`scripts/train_*.sh` + 手动 config（§7 的修复顺序）。

---

## 5. P2 / P3

### P2

1. **OPD 用“这一步实际出现的 teacher 权重和”做归一化**
   （`rl/opd.py:246-254`）：`loss = Σ_present w_k·mean_t KL_k / Σ_present w_k`。
   文档/注释写的是固定 `L = Σ_i w_i E_t[KL_i]`；如果某一步只出现一个 teacher，
   它的权重会被放大成 `w_i/Σ_present w`（例如 0.5/0.5 只剩 code 时，梯度是
   固定权重意图的 2×；0.7/0.3 只剩 code 时是 3.3×）。单卡 16 prompt 时几乎
   每个 teacher 都会出现，影响小；DDP 4 卡时每 rank 只有 4 prompt，实测约
   13% 的 rank-step 会缺 teacher。修复：除以**固定的**
   `sum(self.teacher_weights.values())`，或断言本 step 所有 teacher 都出现。
   注意：`_kl_for_group` 的“每 teacher 一个 token mean”本身**不是** bug——它
   正是 `Σ w_i E_t[KL_i]` 的估计量；不要按 token 数去重加权。
2. **GRPO 的 token 归一化是 per-rank，然后 DDP 对 rank 求平均**
   （`rl/grpo.py:258,262` 用 local `total_tokens`；`:406-407`
   `all_reduce_gradients`）。per-rank token 数不同（变长 response + rank 分片），
   实际梯度是 `(1/W) Σ_r S_r/T_r`，不是文档写的全局 token mean
   `ΣS_r/T_global`；每个 token 的权重是 `1/(W·T_r)`。这**不会**导致 rank 发散
   （缩放发生在 all-reduce 之前），但结果依赖 DP 尺寸，且因为
   `run.py:32` 忽略 `data.shuffle/data.seed`、rank 分片是文件顺序的 stride，
   可能变成系统性偏差。verl 明确禁止这种写法（`core_algos.py:1170-1175`）。
   修复：all-reduce `T_local` 得到 `T_global`，本地 loss 乘
   `world·T_local/T_global`（或在 all-reduce 后按 `1/T_global` 缩放，参考
   §3.4 的写法），并在分片前 shuffle。
3. **3-teacher OPD pipeline 实际只跑 2 个 teacher**
   （`configs/pipeline/minimind_3teacher.yaml:60-65` 把 data override 成
   `opd_prompts_mixed.jsonl`，只有 general/tool；`pipeline.py:191-197` 生效）。
   code teacher 被加载、权重 1/3，但永远不会被路由；叠加 P2-1 的 renormalize，
   实际是 0.5/0.5。修复：指向含 code 的 `opd_prompts_three.jsonl`，并在
   `_group_by_teacher` 里对“配置了但本 step 没出现”的 teacher 告警/失败。
4. **温度口径**：rollout/actor/OPD student+teacher 都用同一个
   `config.rl.temperature`（`wrapper.py:298,359,714-738`），所以 ratio 恒 1 的
   结论成立。但 OPD 实际最小化的是 `KL(π_s^T ‖ π_t^T)`（T=0.8），不是文档里的
   `KL(π_s ‖ π_t)`；GRPO 的梯度尺度也约 1/T。verl 的 distillation loss 是
   未加温度的。建议在 `posttraining_plan.md` 写明“在 T=0.8 下蒸馏”，或把
   teacher/student 的 logits 除以 T 后再恢复（即用 untempered KL），避免把
   温度当成了纯采样参数。
5. **GRPO/OPD 的 metrics 只在 rank 0 记录**（`grpo.py:126-128`、
   `opd.py:118-120`），reward/KL/response_tokens 都是 rank-local；SFT 有
   `Trainer._globalize`，RL 没有。多卡日志会误导。建议在 logger 前对
   scalar 做 token-weighted all-reduce（与 §3.4 一起做）。
6. **没有 stop-rate / finished 指标**。`RolloutSample`（`rl/rollout.py:38-70`）
   没有 `finished` 字段；`CachedRolloutEngine.rollout` 的 `finished` 是局部变量
   （`cached_rollout.py:169`）。`doc/posttraining_plan.md` 说“rollout 已记录
   `finished`”是不准确的。当前 GRPO general 的 `response_tokens` 约 49,152
   = 96×512、OPD 每步 8,192 = 16×512，**几乎全部顶到 max_new**，但没有一个
   指标能直接告诉你 stop 命中率。建议给 `RolloutSample` 加 `finished`/`stop_hit`，
   在 stats 里记 `stop_rate` 和 `max_new_hit_rate`。
7. **GRPO/CISPO 的 clip 在当前配置下是死代码**。`recompute_old_logprobs: true`
   时 `old_logprobs = actor_logprobs.detach()`（`grpo.py:219-220`），ratio 恒为 1；
   实跑 metrics 也确认 `ratio_mean=1.0`、`clip_fraction=0`、`approx_kl=0`；
   实测 cispo 与 grpo 的梯度相对差 3.6e-7、`epsilon_high` 1 vs 5 无差异。
   这等价于 on-policy 的 REINFORCE-with-group-baseline，本身没问题，但
   `clip=0.2` / `loss_type=cispo` / `epsilon_high=5` 的配置和文档会让人以为
   有 trust region；`test_grpo.py:147` 的 `approx_kl≈0` 断言也因此是空的。
   建议：要么删掉这些误导项/在文档写清“无 clip”，要么实现真正的 PPO 多 epoch
   （需要保留 rollout 的 old logprob，并处理 D83 的 4e-2 偏差）。
8. **SFT `auto_resume: true` 是死代码**（`configs/sft/*.yaml` 同时设了
   `train.resume`）。`trainer.py:1662-1669` 只在 `resume is None` 时才看
   auto_resume。重跑同一 config 会从 pretrain base 重新开始、写同一个
   `final`，并把 `metrics.jsonl` 再追加一段 0..119。建议在 `out_dir/final`
   已存在且 `resume` 指向别处时拒绝启动（或要求 `--force`）。
9. **`resume_data` + `resume_step: false` 的默认组合会重放数据流开头**。
   `trainer.py:1620-1623` 的 replay 循环会消耗 mixer 的 slot，但 mixer 是有限
   调度（`total_batches = steps*grad_accum`）；用完后 `_pull_batch` 会重建
   factory 并从 slot 0 开始，SFT 前 120*16 个 micro-batch 与 replay 的前缀
   完全重合。SFT configs 显式设了 `resume_data: false` 躲过它，但默认值是坑。
10. **SFT 的 target_mask 完全不校验**（`run.py:235-259` 只查 tokenizer hash、
    vocab、seq_len）。一个全零 mask 的 pack 会跑满 120 步、loss=0、不报错。
    建议校验 meta `format == minideepseek-sft-v1`、`0 < target_tokens < tokens`、
    每个 document 至少 1 个 target，并在第一步断言 `main_token_count>0`。
11. **SFT 没有任何 validation**。三个 config 都没设 `val_every`/`val_path`/
    `val_sources`，`final` 没有 held-out 数字；GRPO/OPD 又都从这个 `final` 出发。
    建议至少从 `data/eval_*.jsonl` 抽一小份 val_sources，每 40 步跑一次。
12. **SFT base 是 266M token / 8.9% 的 pretrain**。三个 config 都 resume
    `curriculum_145m_3b/stage_00_seq2048/step_001015`（1015×262,144 = 266M，
    约 3B 计划的 8.9%、2K 阶段的 59%）。所有 SFT/GRPO/OPD 质量结论都是在这个
    base 上测的；路径还是硬编码的旧 step，预训练推进后会过期。建议
    pretrain 结束后改到最终 checkpoint，并在 meta 里断言 base 的
    step/token 数。
13. **`OPD`/`GRPO` 没有 auto-resume**；`configs` 的 `resume` 永远指向 SFT base，
    crash 后重跑会从 base 重新开始并覆盖 out_dir 的 step_*/final。
    `TrainSettings.auto_resume`/`keep_last_ckpt` 在 `rl/*.py` 里完全没被读取
    （SFT/pretrain 的 `trainer.py:1663` 有）。建议复用 `latest_checkpoint(out_dir)`：
    如果 out_dir 已有 checkpoint 且用户没有显式要求从头开始，就从最近
    checkpoint resume（只加载模型，`step_index` 恢复），并 honor `keep_last_ckpt`。
14. **SFT 的 `mtp_weight` 最后一步是 0.108，不是配置的 0.1**
    （`trainer.py:541-543`，120 步时 progress=(119-96)/24=0.958）。
    影响很小，但会让人对“0.1→0.1”的调度产生疑问。
15. **`MetricsLogger`/报告对多卡不聚合**（同 5）；`doc/PROCESS` 的“308 passed”
    也没有覆盖多卡 SFT。
16. **math 的 fallback 取“文本里最后一个数字”是奖励 hack**
    （`reward.py:61-63`）：没有 `\boxed`/answer marker 时，最后出现的数字
    就被当答案。`"I cannot solve this, the numbers are 3 and 5"` 对 reference
    `5` 会判对；`"1,234"` 会取 234。建议没有明确答案标记就返回 `None`（不 parse），
    数字前先去掉逗号。
17. **tool 的 gt 匹配会扫描 think 文本**（`reward.py:391-392` 用 raw final
    turn；MiniMind 先按 `</think>` 切掉）。一个错误答案只要 think 里出现
    正确答案就能拿 `gt_match=2.5`。建议 `final_text = final_turn.split("</think>")[-1]`。
18. **eval 永远 `open_thinking=True`**（`evaluate.py:92`），而 RL 的
    general/tool 是 `thinking_ratio=0.9/0.1`；tool eval 还是单轮、不执行
    tool。即使修了 4.4/4.5/4.13，eval 与训练分布仍不一致。建议给
    `EvalConfig` 加 `thinking_ratio`，tool eval 走多轮 rollout。
19. **RM 看到的是带 think 的 raw response**（`reward.py:340/374`），MiniMind
    传的是 `</think>` 之后的答案；`general_reward` 在没有 RM path 时静默返回
    0（只靠 shaping），应当显式报错/告警。
20. **`unit_converter` 的温度换算是错的**（`tools.py:37,101`）：只乘
    `1.8`/`0.5556`，没有偏移；`10°C -> 18.0`（正确 50.0），
    `18°F -> 10.0008`（正确 -7.7778）。tool 数据里 1,394 条温度题因此
    无法被工具正确验证。改成 affine 公式。
21. **verl adapter 结构性问题**（可选路径）：`compute_score` 对 general
    没接 RM（永远 0）；`to_verl_record` 丢 `gt` list/tools/multi_turn；
    `build_verl_opd_config` 用 `teacher_key: "data_source"`，而
    `teacher_models` 的 key 是 general/tool/code，无法路由。
22. **最终一轮的 tool observation 被丢掉**：`agent_rollout.py:171-173` /
    `cached_tool.py:192-194` 在 append observation 之前就 `break`；中间轮的
    observation 在 sequence 里并正确 mask，最后一轮不在。MiniMind 会把最后
    一轮也 append。对 loss 无影响，但 reward/轨迹语义与 reference 不同。
23. **SFT 的 `train.resume` 是 mid-2K 路径**（`step_001015`），不是最终的
    8K stage；显式 resume 又让 auto_resume 失效。预训练推进后应改成
    `stage_02_seq8192/final`（或最终 checkpoint），否则直接跑 SFT 会从
    旧的 266M base 开始。
24. **code reward 的边界**（`reward.py:140-180`）：空 fenced block 会拿
    `format` 奖励且 `compiled=1.0`；syntax-error 路径 total=-0.1 但
    components 里 `format=+0.05`；no-tests 路径 total=0 但 `format` 在
    components 里。建议要求非空 code，并保证 `sum(components)==total`。
25. **tool 的 no-call 分支跳过 unfinished penalty**（`reward.py:371-377`），
    `max_tool_calls` 截断执行但 reward 仍解析所有 calls；这些边界会让
    unfinished/超额调用少扣分。

### P3

1. `flat_logprobs`（`rl/rollout.py:245-262`）按 `sample.response_length` 切
   actor logprob，但 `actor_logprobs` 只返回 mask=True 的 token；对多轮 sample
   会长度不匹配。目前只在 test 里用未 masked 的 sample，是潜伏 bug。
2. `CachedToolRolloutEngine` 截断到 `max_new_tokens` 时没有 `unfinished=True`，
   reward 的 unfinished penalty 不会触发；真实截断轨迹被当成正常结束。
3. `make_reward_fn` 只要 `reward_model_path` 有值就加载 1.8B RM
   （`rl/reward.py:421-430`），即使 `kind: tool` 只在“没有 tool call”时用；
   单卡上白占 3.6 GB、启动变慢。可以在 tool/general 需要时 lazy load。
4. `code_reward` 用 `subprocess.run([sys.executable, "-I", script])` 执行模型
   生成的代码，只有超时，没有文件系统/网络隔离；研究环境可接受，但别在共享
   机器上跑不可信数据。
5. `_validate_rl_settings` 在 `opd.py` 里的错误文案写的是 “OPD does not ...”，
   但 `grpo.py` 也调用它（各自一份），报错会误导。
6. `GRPO`/`OPD` 的 `save("final")` 在 `train()` 末尾和 `run_*.py` 各一次
   （`rl/grpo.py:433` + `rl/run.py:44`；OPD 同理），重复 ~1 GB I/O。
7. `SFT` 的 `positions.npy` 被保存（general 132 MB），但
   `SFTDataset.__getitem__` 又用 `torch.arange` 重算，完全不读。
8. `data.domain` 在 SFT config 里没有被 trainer 使用（真正 domain 来自
   `data.sources[].domain`），容易误以为它决定混料。
9. `.gitignore` 的 `.venv/` 匹配不到 symlink `.venv`（服务器上会显示
   `?? .venv`）。
10. `OPD` `kl_total` 实际是“除以 weight_sum 后的加权平均 KL”，命名像“总和”；
    `response_length_mean` 含 observation token，而 `response_tokens` 只含
    loss token，两个口径不一致。
11. **不要按 `distributed.clip_grad_norm` 的 docstring 去改 GRPO 的
    `torch.nn.utils.clip_grad_norm_`**：实测（torch 2.15，FSDP2 DTensor，
    local shard norm 1.84e-2 vs 2.13e-2）它在两个 rank 上返回了相同的
    global norm 4.17426634；不是 B40 那类 bug。DDP 下梯度也已一致。
12. `ToolRolloutEngine`（非 cached）对每条 plain record 单独 `generate`，
    且 seed 相同；慢且轨迹相关性高，只在 backend=torch 时用。
13. `GRPO` 的 `_merge_stats`/`OPD` 的 `response_length_mean` 用
    `response_tokens`（mask=True 的 loss token 数）做权重，但它本身是
    per-sample 均值（tool sample 还含 observation token）。实测
    (50 token, mean 5) + (10 token, mean 100) 合并成 20.83，真实 sample 均值
    13.64。只是日志。
14. `rl.method` 在 configs 里从未被读取（`run.py:95` 只按 argv 分发）；
    `data.shuffle`/`data.seed` 对 GRPO 也没生效（`run.py:32` 直接
    `load_records(path)`，`build_grpo_dataset` 不 shuffle）；所以多卡分片是
    文件顺序 stride，可能让某个 rank 只看到一种 domain。
15. 分布式 resume 跳过 `load_checkpoint` 的 `model_sha256` 校验
    （`grpo.py:501-505`、`opd.py:356-360`），checkpoint 损坏不会被发现。
16. `group_advantages` 的 `eps=1e-6` 比 MiniMind 的 1e-4 小：接近同分的
    group 会被放大成较大的 advantage；zero-std 返回全 0 本身是对的。
17. `_finalize` 只在 `key not in result.components` 时加 shaping
    （`reward.py:292-294`）；如果某个 reward 自己用了 `length`/`think`/
    `repetition` 作为 component，对应 shaping 会被静默丢掉。目前没有
    reward 这样做，属 latent。
18. `configs/pretrain/generated/pretrain_145m_g4.yaml:70` 仍是 `fsdp2`；
    `scripts/launch_curriculum.sh:16` 默认 `curriculum_145m_g1.yaml` 不存在。
    这些是预训练/dashboard 侧的旧产物，本次不动，但下次重启前要清理。

---

## 6. 明确验证为“不是 bug”的点

这些是审计里逐项验证过、不需要改的（避免后续重复排查）：

- **response-only mask**：`assistant_token_mask` 只标 assistant 内容 + 结尾
  `<|im_end|>\n`，不标 system/user/tool；88k 条 SFT 数据逐条检查
  `bad_start=0`、`no_footer=0`；`<|im_end|>` 与后续 `\n` 都是 target，
  EOS 有监督。
- **截断**：现有 88k 条没有一条超过 4096（general/code/tool 最大
  1,214/3,546/1,178），所以“右侧截断丢掉 EOS”目前不触发，只是潜在问题。
- **SFT 单卡的 token-weighted grad accumulation**：两个真实样本（147 vs 656
  target）累加梯度与拼成一个大 batch 的梯度相对误差 1e-4（n_mtp=0）/4.4e-4
  （n_mtp=1），naive 等权会有 34% 偏差；说明单卡实现是对的。**多卡才坏**（§3）。
- **SFT MTP 的 `shift_targets` / `_masked_ce`**：doc 边界不跨文档，MTP
  有效 target 数与 main 一致；MTP 权重缩放没有实际偏差。
- **SFT 混料比例（input 口径）**：1920 个 slot 的实际配额
  1728/134/58 = 0.9000/0.0698/0.0302，和配置一致；每个 rank 的
  `order[rank::world]` 分片不重叠。
- **SFT 数据加载/meta**：`skipped=0`，meta 的 tokens/target_tokens 与数组
  一致；dataset tokenizer hash/vocab/seq_len 校验存在。
- **MTP strict load**：pretrain/SFT 都是 n_mtp=1；GRPO/OPD 都是 n_mtp=0 并用
  `ignore_prefixes=("mtp_layers.",)`；没有 mismatch 路径。
- **KV-cache rollout**：prefill/decode 与全量 forward 在 fp32 CPU 上逐 token
  对齐（tests `test_cache.py`、`test_cached_tool.py`）；cached engine 的采样
  logprob 与 actor 全量 forward 一致（<1e-4）。
- **tool continuation 的数值**：除 observation 对齐外，新生成 token 的
  sampling logprob 与全量 forward 的最大差 9.5e-7；cache 长度/position 的
  replay 数学是对的。问题只在模式/上下文预算。
- **DDP GRPO/OPD 的梯度 all-reduce**：`all_reduce_gradients` 存在（B46 已修），
  2 进程 smoke PASS；不像 SFT 那样在 reduce 后做本 rank 缩放。
- **GRPO 的 advantage**：`group_advantages` 是组内 zero-mean/unit-variance，
  与 verl 的 `compute_grpo_outcome_advantage` 一致；std=0 时返回全 0。
- **CISPO 的 detached clamp**：与 CISPO 的“stop-gradient on clipped weight”
  一致；问题只是当前 on-policy 配置下 ratio=1，clip 不生效（§5 P2-7）。
- **奖励解析**：math 的 `\boxed`/answer marker、group advantages、code
  单元测试 pass/fail、tool gt/alignment 路径都有测试；未发现符号/重复计数。
- **OPD 的 KL 方向/尺度**：用同一组 student/teacher logits 与直接
  `KL(π_s‖π_t)` 对比，值一致（0.97457；forward KL 0.9816）；full vocabulary，
  per-token mean，方向正确。
- **GRPO 的 intra-rank mini-batch chunk**：所有 chunk 用同一个 local 分母，
  求和等于 local full-batch mean；advantage broadcast 正确。**问题只在多卡**
  的 per-rank 归一化（§5 P2-2），不是 chunking。
- **step/checkpoint 计数**：`step_N` 在 N 次 update 后保存且 step=N；
  `resume_step=true` 从 N+1 继续；scheduler 在 optimizer 前 step，与 pretrain
  trainer 一致；没有 off-by-one。
- **温度**：rollout/actor/OPD student+teacher/cached+tools 都用同一
  `config.rl.temperature`，ratio 恒 1 的结论成立。**但 loss 用的是 tempered
  distribution**（OPD 最小化 `KL(π_s^T‖π_t^T)`，GRPO 梯度尺度约 1/T），
  与文档的 untempered KL 不一致（§5 P2-4）。
- **`clip_grad_norm` 的 FSDP2 行为**：实测 torch 2.15 下
  `torch.nn.utils.clip_grad_norm_` 在两个 rank 返回相同 global norm
  （4.17426634），不是 docstring 警告的 local-shard 问题；不要按 P3-11
  去改。
- **Manual-DDP + MoE 的 all_reduce**：expert 是 stacked Parameter，未使用
  expert 仍产生 zero grad；不会出现某 rank `grad=None` 导致 collective
  不匹配。
- **reward model 的格式/设备**：本地 `internlm2-1_8b-reward` 在 CPU float16
  能加载并打分（-0.814），`[user, assistant]` 格式与 MiniMind 一致；
  dtype/device 不是 bug。
- **tool mock 的确定性**：除温度换算（§5 P2-20）外，mock 结果确定；
  observation token 在 sequence 里、loss mask 外（中间轮）正确。
- **eval 的 index_topk**：从 checkpoint `extra.index_topk` 读取
  （`evaluate.py:56`），不是 stale 假设；stale 的只有 checkpoint 路径。
- **cache slot/fork/stop**：`slot = local*group_size+offset`、scratch
  slot=`num_slots-1`、bucket padding、group fork、stop/max_new 都正确；
  cached sampling logprob 与 actor 全量 forward 约 5e-7。
- **`_sample_tokens`**：top-k/p 只影响采样 id，返回的 logprob 来自未过滤的
  temperature-scaled 分布，与 actor recompute 一致；不是 bug。
- **SFT 全量截断检查**：877,651 条 general 全部可编码、0 skipped、0 条因
  >4096 而丢 footer；code/tool 也 0 截断。现有数据不会触发右侧截断问题。


---

## 7. 建议的修复顺序

### 立刻（不影响预训练）

1. eval 路径（4.4）与 GRPO config `mode: ddp`（4.3）：纯配置，最安全。
2. tool continuation 的 eval 模式 + context 预算（4.1/4.2）+ `open_thinking`
   记录 + `max_turns` 对齐（4.11）+ 回归测试。
3. SFT 数据构建的 shuffle/dedup/分层与 `seed` 生效（4.6），重新 tokenize
   三份 SFT 数据；同时修正 steps（4.7）与混料口径（4.8）。
4. eval reward dispatch（4.5）、eval code tests 映射（4.13）——和 4.4 一起改。
5. tool reward 的 unfinished/alignment/shaping（4.12）：先修“unfinished 仍给
   gt”和“`len(gt)` 当调用数”，这两条会直接教模型错误的工具行为。
6. pipeline 对齐或明确标注（4.9/4.15）：至少修 pretrain 2× 预算、tokenize
   布局、SFT sources 与 3teacher 数据；否则 pipeline 不可用。
7. reward clip 饱和（4.10）：提高 general/tool 的 `reward_clip` 或把 shaping
   排除出 clip；这是“模型变好时反而没梯度”的定时炸弹。

### 需要你确认 / 等预训练结束

8. **P0 的 `trainer.py` 梯度缩放**（§3）：共享代码，建议等 hb01 当前 job
   落盘后单独修；修复对满 pack 预训练是 no-op，但需要你确认“动共享 trainer”
   可以接受。修完跑 2 进程“不同 target 数 → 权重逐位一致”的回归。
9. hb01 的代码同步（§1.2）：等 pretrain 结束，`stash` → `merge --ff-only` →
   丢弃只差注释的 stash。
10. SFT base checkpoint 更新到最终 pretrain（§5-P2-12）。

### 之后（多卡 / 质量）

11. 多卡 metrics 聚合、stop-rate、OPD teacher renormalize（P2-1/5/6）。
12. resume 语义（P2-8/9/13）、SFT target_mask 校验（P2-10）、SFT validation（P2-11）。
13. 文档：`posttraining_plan.md` 的 120/90/80 步与 epoch 说法、
    `specialist_data_145m.md` 的 10× 预算、`server_runbook.md` 的旧路径。

---

## 8. 未验证 / 风险 / 待确认

- **没有真实多卡 GPU 验证**：SFT P0 是 CPU Gloo 2 进程复现；DDP/FSDP2 的
  NCCL 行为、`all_reduce_sum` 的 device 放置需要 GPU 上再确认。
- **没有跑长 tool 轨迹**：context 溢出是用 debug 模型（seq_len=64 和
  max_context=1024）复现的；真实 4096 下是否一定崩取决于 prompt/observation
  长度，但 `max=2590 + 768×3` 已经超过 4096。
- **HCA 不一致只在 CPU debug 模型上验证**（max_context>seq_len）；GPU
  bf16、真实 4096/8192 上下文下未测，但机制与数值无关。
- **eval/RM 没有实跑**：eval 的 stale path、tests 丢失、reward dispatch
  是代码/数据层验证；没有加载 1.8B RM 跑完整 eval。
- **pipeline 没有端到端跑通**：断点是通过直接调用 `_pretrain_stage` /
  `_sft_stage` / `_preflight` 确认的；没有真的让 pipeline 跑完一个 stage。
- **没有验证 CUDA graph 捕获 counter 的 GPU 行为**：CPU eager 已证明污染；
  graph 路径取决于 bucket 1 首次捕获时机，需要在 4090 上用
  `use_cuda_graph=True` + `tokens_per_expert_E` 检查。
- **没有做 SFT 数据重采样后的质量 A/B**：当前“前缀块”结论有数据统计支撑，
  但重采样后能提升多少未测。
- **没有改任何代码**：本文只是审计；修复需要你确认顺序（尤其 §3 的共享
  `trainer.py`）。本地 `trainer.py` 还有并发会话的未提交修改，修之前要先
  协调。
- **RPO 未找到**：仓库里只有 GRPO / CISPO（`rl.loss_type`）和 OPD，没有
  `rpo.py`/DPO/RPO 实现。如果“RPO”指的是 DPO/RPO 偏好优化阶段，它目前是
  **缺失**的；如果只是笔误指 GRPO/CISPO，请忽略这一条。需要你确认后再决定
  是否补一个偏好优化 stage。

---

## 附录 A：关键证据的位置

| 证据 | 位置/命令 |
|---|---|
| SFT DDP 分叉 | `/tmp/ddp_scale_probe.py`（2 进程，18 vs 4 target：max diff 2.127e-5，79/106 参数；18 vs 18：0.0） |
| tool train-mode 污染 | `/tmp/tool_probe2.py`（counter 0→369；new-token logprob diff 9.5e-7） |
| tool context 溢出 | 同一脚本，seq_len=64 → `IndexError: index -94 out of bounds`；独立复现（max_context=1024、1397-token observation）→ `IndexError: index 1024 ...` |
| tool open_thinking 歧义 | 真实 tokenizer：open/closed 两个候选 prefix 都匹配，tail 分别 `...<think>\n` / `...<think>\n\n</think>\n\n`；循环总选 True |
| `max_turns` 少一轮 | 代码阅读：`cached_tool.py:175` + `192-194`；reference `agent_rollout.py:134` |
| reward clip 饱和 | 真实 `dispatch_reward`：RM 2.9/3.0 → total 3.0，`group_advantages([3,3,3])=[0,0,0]` |
| tool unfinished 仍给 gt | 构造 rubric：`unfinished=True` 总 0.5（gt_match 2.5，unfinished -0.5） |
| eval code tests 丢失 | `load_records("data/eval_code.jsonl")[0].rubric == {}`；`code_reward` 返回 `record has no rubric.tests` |
| HCA 长上下文不一致 | 独立复现：max_context=1024、seq_len=256、prompt=640，decode vs packed logits max diff 0.134 |
| SFT 数据统计 | `data/sft_*.jsonl` + `data/tokenization/sft_*.json`；first-N prefix / source 分布 / 唯一会话；全量 general 877,651 条 0 截断 |
| SFT 实跑 token | hn01 `checkpoints/sft_*_145m/runs/*/metrics.jsonl`（12.13M / 5.59M / 3.86M target） |
| GRPO/OPD 实跑 | hn01 `checkpoints/grpo_general_145m/metrics.jsonl`（ratio=1, clip=0），`checkpoints/opd_three_grpo_145m/metrics.jsonl`（8192 resp tok/step） |
| prompt 长度 | hn01 真实 tokenizer：tool max 2590 / p99 2323；OPD max 1754 |
| eval 路径缺失 | hn01 `checkpoints/grpo_general/final` 不存在，只有 `_145m` |
| pipeline 断点 | `_pretrain_stage` → `SystemExit: missing tokenized pretrain datasets`；`_sft_stage` → missing `sft_general.npz`；3teacher preflight → missing `opd_prompts_code.jsonl` |
| 测试 | 本地/hn01 后训练 34 passed；DDP GRPO/OPD smoke PASS；SFT/reward/pipeline 定向测试另计 23 + 25 passed |

## 附录 B：本次没有改动的文件

- 没有改 `src/`、`configs/`、`scripts/`、`tests/`；
- 没有改 `trainer.py`（并发会话有未提交修改，且 P0 修复需要你确认）；
- 没有碰 hb01 的 job；没有改预训练配置；
- 只新增本文档（以及 `doc/PROCESS.md` 的一条进度记录，若已提交）。
