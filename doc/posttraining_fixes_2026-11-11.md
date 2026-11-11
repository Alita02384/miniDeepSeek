# 后训练审计修复记录（2026-11-11）

> 对应审计：`doc/posttraining_audit_2026-11-11.md`。
> 范围：SFT / GRPO / CISPO / 多教师 OPD / rollout / reward / eval / pipeline。
> **没有修改预训练配置，没有触碰 hb01 上正在跑的 pretrain job。**
> 代码根：`/Users/cjh/Workspace/miniDeepSeek/miniDeepSeek`；验证机：本地 CPU +
> hn01（单卡 4090，空闲）。

## 0. 结论

| 级别 | 审计数量 | 本轮状态 |
|---|---:|---|
| P0 | 1 | 已修 + 2 进程 DDP/FSDP2 回归 |
| P1 | 15 | 14 项代码/配置已修；1 项（SFT 域混料口径）随 G1 修复并实测 |
| P2 | 25 | 核心项已修（多卡归一化、teacher 权重、resume、stop rate、metrics、eval、数据、pipeline）；少数留待后续 |
| P3 | 18 | 修了 1/2/3/7/9/13/14/15；其余记录为已知限制 |

**超出审计的新 bug**（本轮实测发现）：

1. `TrainSettings` 没有 `stage_index`，`Trainer._stamp` 在**任何独立 SFT/预训练**
   的第一步都会 `AttributeError`。只有 curriculum 动态属性路径碰巧能跑。
   已加字段并补回归（`tests/test_trainer.py` 30 项全过）。
2. `rollout.flat_logprobs` 按 `response_length` 切 actor logprob，但 actor 只返回
   `token_mask=True` 的 token；多轮 sample 会整体错位。已按 `num_loss_tokens`
   切分。
3. `code_reward` 的 components 与 total 不一致（syntax/no-tests/空 fenced block）；
   已让每个 component 带权重并保证 `sum(components)==total`，空代码块不再拿
   format/compiled 分。
4. HCA 的 packed 窗口选的是**最旧**的 `width` 个块，与注释和 decode 相反；
   `max_context>seq_len` 时 decode/packed logits 差 0.134。已统一为最近窗口。
5. `build_trainer` 的 `auto_resume` 只在 `resume is None` 时生效，而 SFT 配置都
   设置了 `resume`，所以重跑会从 pretrain base 覆盖 `final`。已让 out_dir 内
   的最新 checkpoint 优先，并恢复 step/data replay。

## 1. P0：多卡 SFT 梯度缩放

- 文件：`src/minideepseek/train/trainer.py`
- 旧逻辑：DDP/FSDP2 先按 AVG 归约，再除**本 rank** 的 target token 数。
  SFT 的 response-only mask 使各 rank target 数不同，参数一步后分叉。
- 新逻辑：先 all-reduce 本地 target 数得到 `T_global`，再乘
  `world_size / T_global`（单卡仍为 `1/T_local`）。target 数相等时逐位等价，
  因此对满 pack 预训练是 no-op。
- 回归：`scripts/test_sft_ddp_scaling.py`（Gloo，2 进程，rank0 4 个 target /
  rank1 2 个 target）→ DDP 与 FSDP2 的 `max weight diff = 0.000e+00`。
  `tests/test_sft_ddp_scaling.py` 把两种模式纳入 pytest。

## 2. P1 修复

### 2.1 tool 多轮 continuation（4.1/4.2/4.11）

`src/minideepseek/rl/cached_tool.py`、`cached_rollout.py`、`rollout.py`：

- `CachedToolRolloutEngine.rollout` 对所有 continuation 调用包在
  `model.eval()` / `finally: model.train()` 中；`_continue_tool` 入口断言非
  training。避免 MoE `tokens_per_expert_E` 和首次 CUDA graph capture 被 rollout
  污染。
- `RolloutSample` 新增 `open_thinking/finished/max_new_hit`；cached 与 torch
  engine 都记录。continuation 直接用记录的 flag，不再 trial-render（旧循环
  永远选 True）。
- 每轮先算 `room = max_context - len(prefix) - len(observation)`；observation
  放不下就结束轨迹并标 `unfinished`，不再让 decode 写到 RoPE 之外。`_feed`
  加越界断言，错误信息带 `max_context`。
- `max_turns` 对齐 reference：先让最后一轮生成，再 parse/执行它的 tool calls，
  然后 `unfinished=True`。`max_turns=3` 现在确有 3 个 assistant turn。
- 回归：`tests/test_cached_tool.py` 新增 eval/counter、open_thinking、
  长 observation、max_turns、越界 `_feed` 五项；相关 39 项通过。

### 2.2 GRPO config 与 eval 路径（4.3/4.4）

- `configs/grpo/{general,code,tool}.yaml`: `distributed.mode: ddp`（cached
  rollout 每 rank 一份完整模型，FSDP2 需要 gathered copy，尚未接）。
- `scripts/train_grpo.sh` 注释改为 manual DDP。
- `configs/eval/*.yaml` checkpoint 改 `_145m`；`doc/server_runbook.md` 同步。

### 2.3 eval reward dispatch / eval_code tests（4.5/4.13/P2-18）

- `evaluate()` 改为 `make_reward_fn` / `dispatch_reward`：general 加载 1.8B RM，
  tool 走 `tool_reward`，math/code 保持 verifier。旧 else 分支把 general/tool
  送进 heuristic world judge，报告与 GRPO reward 不可比。
- `data/grpo.py::_rubric_from_record` 把顶层 `tests/test_code/expected_calls`
  映射进 rubric，并在有 tests 时把 verifier 默认成 `unit_test`；eval_code 的
  `pass_rate` 不再结构性为 0。
- `EvalConfig.thinking_ratio`（general 0.9 / tool 0.1），details 记录
  reward kind/source、thinking_rate、`tool_execution=false`（tool eval 仍是
  单轮，不执行 tool，不能当完整 agent 评估）。
- 测试：`tests/test_eval.py` 5 项。

### 2.4 reward（4.10/4.12/P2-16/17/19/24/25）

`src/minideepseek/rl/reward.py`、`tools.py`：

- `reward_clip` 只夹 base reward（RM/verifier），shaping 在 clip 外叠加；被 clip
  时加 `clip` component，保证 `sum(components)==total`。RM 2.0/2.9/3.0 的组不再
  全被压成 3.0，`group_advantages` 非零。
- `tool_reward`：unfinished 不给 gt 分、不加 length/think shaping；
  `expected_calls` 存在时才按它算调用数，否则只罚 invalid/extra call，
  **不再把 `len(gt)` 当调用数**；gt 只在最终 turn 的 `</think>` 之后匹配；
  有 tool call 时不做 length/think shaping。
- `code_reward`：component 带权重、空 fenced block 0 分、syntax/no-tests 路径
  与 total 一致。
- math 无 `\boxed`/answer marker 时不再猜“最后一个数字”；`x = 42` 仍识别为
  显式答案；`1,234` 先去逗号。
- general RM 看 `</think>` 之后的答案；`make_reward_fn` 对 general 缺 RM 直接
  报错。
- `unit_converter` 温度改 affine：10C→50F，18F→-7.7778C。
- 测试：`tests/test_grpo.py` / `test_tool_agent.py` 新增 6 项。

### 2.5 SFT 数据与训练量（4.6/4.7/4.8/P2-10）

`src/minideepseek/data/sft.py`、`train/run.py`、`data/tokenize.py`：

- `build_sft_dataset` 不再取文件前缀：全量扫描 → conversation 去重 → 按每文件
  唯一条数比例分配 `max_records` → 用 seed 打乱行号 → 编码；meta 记录
  total/unique/duplicates/selected、`selection_sha256`。
- `data.sources` 权重按 `target/input` 折算成 loss-token 权重。
  实测新数据：general 0.900、code 0.744、tool 0.723（目标 0.90/0.75/0.70）。
- SFT 训练前校验 meta format、`0<target_tokens<tokens`、每个样本至少 1 个
  target；micro-batch target 为 0 直接报错。
- `tokenize.py` 同时构建 `sft_{general,code,tool}`（50k/30k/8k），修掉
  `sft_code.npz` 缺失；`prepare.py sft` 暴露 `--no-dedup/--no-shuffle`。
- `configs/sft/*.yaml` steps 用实测 target/step 重算：**210 / 100 / 80**
  （约 1.0 / 1.0 / 1.8 target epoch；`scripts/sft_step_budget.py` 可复算）。

hn01 重建结果（seed=1234，4096）：

| 源 | 全量 | 唯一 | 抽样 | input token | target token |
|---|---:|---:|---:|---:|---:|
| general | 877,651 | 830,933 | 50,000 | 25.59M | 21.63M |
| code | 148,723 | 129,791 | 30,000 | 12.31M | 5.69M |
| tool | 99,552 | 66,000 | 8,000 | 5.41M | 1.93M |

旧数据（前缀）为 28.42M / 6.58M / 0.92M target；新抽样把 code 的 5 个来源、
general 的中后段、tool 的长会话都纳入了。

### 2.6 HCA 长上下文一致性（4.14）

`src/minideepseek/model/attention.py`：packed `_hca_topk` 改为取
`[limit-width, limit)`，decode `_select_sparse` 取
`[n_comp-width, n_comp)`，都受 `hca_topk_max` 限制。回归：`seq_len=256,
max_context=1024, prompt=640`，prefill+3 步 decode 与全量 forward <1e-5。

### 2.7 pipeline（4.9/4.15）

`src/minideepseek/pipeline.py` + `configs/pipeline/*.yaml`：

- pretrain 的 `grad_accum` 从 base config 读（16，不再 fallback 8）；支持
  `pretrain_{domain}_{2k,4k,8k}.npz` 并按域求和；已有 `until_step/sources` 的
  curriculum stage 原样保留，不再改写成单个 `tokenized_path`。
- SFT steps 从**全部** `data.sources` 的 target/input 比例和 loss-token 混料
  计划算；有 sources 时不再 override `tokenized_path`。
- `_stage_done` 在 `out_dir/*/final` 里找；preflight 检查 pretrain/SFT 输入、
  GRPO resume、OPD data + teacher checkpoint，`--dry-run` 缺路径也直接失败。
- 配置改名到 `_145m` / `_3b`、`opd_prompts_three.jsonl`；3teacher 的 OPD 不再
  指向只有 general/tool 的 mixed 池。
- 测试：`tests/test_pipeline.py` 5 项。

### 2.8 RL 多卡与恢复（P2-1/2/5/6/13）

`src/minideepseek/rl/{grpo,opd,run}.py`、`train/distributed.py`：

- GRPO：all-reduce 本地 loss token 数，按 `world/global_tokens` 缩放；DDP 结果
  等于全局 token mean（旧为 per-rank mean 的再平均）。
- OPD：除以**固定**配置权重和 `Σw_i`；每个 teacher 的 KL 按
  `world*local_tokens/global_tokens` 缩放；某步缺 teacher 不再放大剩余权重。
- 两个 trainer 新增 `finished_rate/max_new_hit_rate/samples`；rank0 记录前用
  `all_gather_object` 合并（均值按 sample 或 token 加权，std 用方差合成）。
- `rl/run.py`：out_dir 有 checkpoint 时 auto-resume 并恢复 step；按
  `keep_last_ckpt` 裁剪；删掉 `train()` 之后的重复 `save("final")`。
- 分布式 resume 现在校验 `model.pt` 的 sha256。
- 配置启用 `auto_resume: true`、`keep_last_ckpt: 2`。

## 3. 验证

```bash
# 本地
.venv/bin/python -m pytest tests/test_grpo.py tests/test_opd.py \
  tests/test_cached_tool.py tests/test_cache.py tests/test_tool_agent.py -q
# -> 39 passed（新增 tool 回归后）

.venv/bin/python -m pytest tests/test_pipeline.py tests/test_data.py \
  tests/test_eval.py tests/test_trainer.py tests/test_resume.py -q
# -> 全部通过

# 2 进程回归
OMP_NUM_THREADS=1 .venv/bin/python scripts/test_sft_ddp_scaling.py --dist-mode ddp
# -> max weight diff = 0.000e+00 / PASS
OMP_NUM_THREADS=1 .venv/bin/python scripts/test_sft_ddp_scaling.py --dist-mode fsdp2
# -> 同上

# hn01（同步后）
.venv/bin/python scripts/sft_step_budget.py --config configs/sft/general.yaml --steps 400
# -> target/step 106,443；1 epoch 203 步；当前 120 步 = 0.59 epoch
#    code 59,114/96 步；tool 44,280/44 步
```

全量 pytest 结果见 `doc/PROCESS.md` 对应条目。

## 4. 明确未做 / 待预训练结束

1. **SFT validation（P2-11）**：三个 SFT config 仍无 held-out。建议从重建后的
   SFT 源里再抽 1% 做 `sft_*_val.npz`，并配 `val_every: 40`；本轮未写，
   因为要同时保证 val 不泄漏进 train，且 hn01 的 12GB checkpoint 盘正被旧
   run 占用。
2. **SFT base checkpoint（P2-12/23）**：仍是
   `curriculum_145m_3b/stage_00_seq2048/step_001015`（266M token）。hb01 的
   3B 预训练还在跑；**等落盘后再改到最终 8K checkpoint**，本轮不动预训练。
3. **hb01 代码同步（§1.2）**：hb01 仍 `e30ef7c` + 14 modified，pretrain job
   在跑；**没有 pull、没有 stash、没有重启**。等 job 结束再
   `stash → merge --ff-only origin/main`。
4. **温度口径（P2-4）**：OPD 实际最小化 `KL(π_s^T‖π_t^T)`（T=0.8），GRPO 梯度
   尺度约 1/T；这是有意保留的采样温度，但文档需写明。已在
   `doc/posttraining_plan.md` 加注，未改 loss。
5. **GRPO/CISPO clip 是死代码（P2-7）**：`recompute_old_logprobs=true` 时
   ratio 恒 1；本轮只把这一点写进文档，没有实现多 epoch PPO（需要保留 rollout
   old logprob 并处理 D83 的 4e-2 偏差）。
6. **tool eval 多轮 + 最后 observation（P2-18/22）**：eval 仍是单轮且不执行
   tool；`unfinished` 时不 append 最后 observation。两者都不影响当前 loss，
   但 evaluator 不能声称是完整 agent 评估。
7. **RPO**：仓库只有 GRPO/CISPO + OPD；用户确认 RPO 是笔误，未新增 stage。
8. **verl adapter（P2-21）**：`rl/verl.py` 的 general RM / `to_verl_record` /
   teacher_key 仍有结构性问题；这是可选路径，未在当前 4090 流程使用。
9. **FSDP2 + cached rollout**：仍不支持（需要 gathered rollout 副本）；GRPO
   configs 已固定 ddp。
10. **P3 剩余**：code 执行无文件系统/网络隔离；`group_advantages` eps=1e-6；
    `_finalize` 的 shaping key 冲突 latent；`tool_reward` 对超额调用只按
    invalid/expected 罚；pretrain generated config 的 fsdp2/launch 旧产物不在
    本轮范围。
