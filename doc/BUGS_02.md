# 审计问题清单 02（2026-11-06）

> 状态：`待修` / `已修` / `已确认非 bug` / `需设计决策`。
> 原则：每项先复现/确认，再最小化修复；修复后补回归测试，避免无关行为变化。
> 本轮审计基于 commit `17cbb0c`，`pytest -q` 基线 **156 passed**（CPU debug 模型）。
> 排查方式：通读 `src/` 全部代码并与 `torchtitan` 参考实现、`DeepSeek-V4-Flash/inference`
> 官方实现逐项对照；对可疑点写 `/tmp` 临时脚本做数值验证（不修改仓库）。
> 修复记录见 [FIXS_02.md](FIXS_02.md)。修复后：除 B53（设计缺口，保留）外全部已修；
> `pytest -q` 回归 **165 passed**。

## 结论摘要

- 核心模型数学（attention / compressor / indexer / mHC / MoE / MTP 标签对齐 /
  packed 多文档边界）**没有发现 bug**，已用独立数值实验验证（见 §4）。
- 训练/数据/RL 侧链路有 1 个 P0、5 个 P1、6 个 P2，以及若干低优先级/设计取舍项。
- 最严重的是 **FSDP2 恢复优化器状态后下一步必崩**：这条路径被文档与测试声称
  “已验证”，但现有测试在 resume 后没有继续 `optimizer.step()`，所以没有暴露。

## P0 — 会让正式训练续跑直接失败

### B40. FSDP2 恢复优化器状态后，下一步 optimizer.step() 必崩

- 状态：已修
- 位置：
  - `src/minideepseek/train/distributed.py::_load_single_optimizer`（约 L407）
  - `src/minideepseek/train/distributed.py::_full_to_local`（约 L313）
  - 触发点：`src/minideepseek/train/trainer.py::Trainer.resume`（dist 分支，约 L587）、
    `src/minideepseek/train/spike.py::restore`
- 问题：`_load_single_optimizer` 把优化器 state 里的每个张量用 `_full_to_local` 转成
  **普通 local Tensor**；但 FSDP2 的参数/梯度是 DTensor。AdamW 在下一步对
  `exp_avg`（普通 Tensor）和 `grad`（DTensor）做 `lerp_` → 混类型报错；Muon 的
  momentum buffer 被恢复成 local shard（形状 = 分片大小），而 `Muon.step` 用
  `parameter.grad.full_tensor()`（全量）做 `add_` → 形状不匹配。
- 复现（CPU/Gloo）：
  - 1 进程 FSDP2，训练 1 步 → save → 新 Trainer resume → `train_step()`：
    `RuntimeError: aten.lerp_.Scalar got mixed torch.Tensor and DTensor, need to
    convert all torch.Tensor to DTensor before calling distributed operators!`
  - 2 进程 FSDP2，同上（Muon 组先崩）：
    `RuntimeError: The size of tensor a (2) must match the size of tensor b (4)
    at non-singleton dimension 0`
  - rollback 路径（`snapshot` → `restore` → 再走一步）同样失败。
- 影响：
  - `full_6b`/`pilot_0.6b`/`dense_control_1b` 等同一次训练**中断后续训**
    （`resume_optimizer` 默认 true）→ 崩；
  - `length_curriculum` 的每个 stage 交接（`_stage_config` 继承 true）→ 第 2 个
    stage 第一步崩；
  - `loss_spike_rollback` 打开时的回滚 → 恢复后第一步崩。
  - SFT/GRPO/OPD 不加载优化器状态（`resume_optimizer=false` / 只加载模型），
    不受影响。
- 为什么没被发现：`tests/test_distributed.py::test_fsdp2_trainer_save_and_resume`
  只断言“至少一个子优化器有 state”，**resume 后没有再 step**；单进程 resume 测试
  同理。`doc/PROCESS.md` 里“2 进程 resume/rollback 临时脚本 PASS”未固化进测试。

## P1 — 静默改变训练结果 / 报告

### B41. sequence-wise balance loss 被施加到 hash 路由层

- 状态：已修
- 位置：`src/minideepseek/train/losses.py::sequence_balance_loss`（约 L39）
- 问题：遍历 `gate_scores` / `routing.layers` 时没有过滤 `layer.is_hash`。
  hash 层 `_select_experts` 只查 `tid2eid`，**完全不读 gate 分数**，所以 balance
  loss 无法改变路由；它唯一的梯度落点是 hash 层 router gate 权重，效果是改变
  routed expert 的加权系数。这与 `update_expert_biases` 里“hash 层不更新 bias
  （读不到）”的处理自相矛盾。
- 证据（debug 模型，15 token 的 batch）：
  - full balance loss = 3.32e-4，其中仅 hash 层就贡献 3.28e-4；
  - backward 后 hash 层 gate 梯度 6.4e-5 / 6.8e-5，learned 层 6.9e-5 / 7.2e-5，
    同量级；相对 LM loss 的 gate 梯度约为 0.1%–0.4%/步。
- 影响：mini-60M 前 3 层是 hash 层（3/12），会被一个不可能达成目标的项持续
  往“压低高负载专家加权”的方向推。单步很小，但方向是系统性的。

### B42. batch-token warmup 在每个 epoch / 数据迭代器重启时归零

- 状态：已修
- 位置：
  - `src/minideepseek/data/pretrain.py::iter_batches`（`batch_index = 0` 约 L310）
  - `src/minideepseek/data/sft.py::iter_batches`（约 L198）
  - `src/minideepseek/train/trainer.py::_next_batch`（迭代器耗尽后重建 factory，
    约 L183）
- 问题：warmup 进度由 `iter_batches` 内部的 `batch_index` 决定，每次新建迭代器
  都从 0 开始；Trainer 在数据耗尽时会重建迭代器，因此 warmup 重置。
- 证据：同一 dataset 用同一个 factory 连续创建两次迭代器，batch 序列完全一样
  `[(8,1),(16,2),(8,1)]`（warmup 没有继续）。
- 影响（旗舰配置估算）：`full_6b` 计划 `12000 steps × 8 accum = 96000` 个
  micro-batch；6B token 在 8 卡、64K tokens/micro-batch 下约
  `6e9 / 65536 ≈ 91500` 个 micro-batch，即约 95% 处数据回卷。回卷后 warmup 重置
  到 base 16K，per-rank 从 2 个 pack 掉回 1 个 pack，最后约 5% 的训练实际
  batch 减半，`tokens` 记账也对不上。
- 附带：验证集每次 `evaluate` 新建迭代器（`batch_index=0`），所以 val 永远用
  base batch_tokens（16K），不是 final 64K。这保持验证子集稳定，属于可接受行为，
  修复时保持。

### B43. `MetricsLogger.summary()` 一旦记录过验证集，就丢掉所有训练指标

- 状态：已修
- 位置：`src/minideepseek/train/metrics.py::MetricsLogger.summary`（约 L46）
- 问题：用 `set.intersection(*(set(item) for item in history))` 求公共 key；验证
  记录只有 `val_loss`，和训练记录没有交集，于是所有训练 key 被丢弃。
- 证据：写入一条训练记录 + 一条 val 记录后再写一条训练记录，
  `summary()` 返回 `{'steps': 3}`。
- 影响：任何 `val_every > 0` 的正式 run，最终 report 里的 lr / grad_norm /
  routing / swiglu 汇总为空；`steps` 还把 val 记录算成训练步。

### B44. GRPO 数据里顶层 `tools` 是 JSON 字符串时，会被拆成“字符列表”

- 状态：已修
- 位置：
  - `src/minideepseek/data/grpo.py::convert_record`（`tools=list(record.get("tools") or [])`，约 L162）
  - `src/minideepseek/data/grpo.py::load_records`（约 L181）
- 问题：MiniMind 惯例里 `tools` 是 JSON 字符串（项目已在 system message 上专门
  解析，见 `tokenizer._parse_json_field`），但顶层字符串在这里被 `list(...)`
  变成字符列表；`GRPORecord.resolved_tools()` 看到 list 就不再 `json.loads`。
- 证据：string tools 经 `convert_record` 后 `len(tools) == 124`，渲染出的
  `<tools>` 段每行一个单字符（`"["`、`"{"`、`"t"` ...），不报错、静默污染整条
  GRPO prompt。
- 影响：凡是“顶层 string tools”的数据集都会训出错误行为。

### B45. 一致性门禁无法直接加载含 MTP 的 checkpoint

- 状态：已修
- 位置：`src/minideepseek/consistency/check.py::main`（约 L71–L80）
- 问题：用 `--n-mtp-layers`（默认 0）建模型，然后 strict 加载、不忽略
  `mtp_layers.`。
- 证据：`python -m minideepseek.consistency.check --flavor debugmodel --seq-len 128
  --checkpoint checkpoints/smoke/final` 报
  `Unexpected key(s) in state_dict: "mtp_layers.0..."`；加 `--n-mtp-layers 1` 后通过。
- 影响：README 里对 `checkpoints/pretrain_full/final`（n_mtp=1）的旗舰命令会直接
  失败。eval/teacher/RL 都已显式忽略 MTP（D38），唯独这个验收门禁没有。

## P2 — 低严重度但确实是逻辑问题

### B46. GRPO/OPD 在 `distributed.mode=ddp` 下从不做梯度 all-reduce

- 状态：已修
- 位置：`src/minideepseek/rl/grpo.py::GRPOTrainer.step`（约 L281）、
  `src/minideepseek/rl/opd.py::OPDTrainer.step`（约 L228）
- 问题：pretrain Trainer 在 `mode == "ddp"` 时显式
  `all_reduce_gradients(...)`（trainer.py 约 L440），RL 两个 trainer 没有；
  `apply_parallelism` 的 ddp 分支返回裸模型，不会自动归约。
- 证据：2 进程 `mode=ddp` 跑 GRPO 1 步后，两个 rank 的权重 digest 不同
  （`42588915496b` vs `34f40e5c5244`）。
- 影响：`ddp` 是 `DistributedSettings` 明确支持的模式（audit 标 ✅），此模式下
  RL 实际是 `world_size` 个独立训练、权重发散。旗舰配置都用 fsdp2，未命中。

### B47. `data.sources` 混料时数据校验被静默跳过

- 状态：已修
- 位置：`src/minideepseek/train/run.py::load_dataset`（sources 分支返回 dict，
  约 L130）、`_validate_dataset`（约 L148）、`run()`（约 L182）
- 问题：`_validate_dataset` 用 `getattr(dataset, "meta", None)`，对 dict 得到
  None 直接 return；混料 run 不做 tokenizer hash / vocab / seq_len 校验。
- 影响：`configs/pretrain/mixed_pilot.yaml` 这类 run 失去数据门禁。

### B48. `tokens` 指标把 MTP 目标也算进去（depth 1 时约 2×）

- 状态：已修
- 位置：`src/minideepseek/model/wrapper.py::LossOutput.token_count`（约 L39）、
  `src/minideepseek/train/trainer.py`（约 L393、L529）
- 问题：`tokens = main_token_count + Σ mtp_token_count`。MTP depth 1 时 MTP 目标
  数约等于主目标数，日志/报告里的 `tokens` 约为真实 LM token 的 2 倍。
- 影响：如果验收口径是“训练到计划 token 数”（roadmap 的 6B），会系统性高报。
- 备注：如果原意是“compute tokens”，应在文档中改名；本轮按“LM 训练 token”修正。

### B49. `tokenize_prompts` 与 rollout 的 prompt 编码不一致

- 状态：已修
- 位置：`src/minideepseek/data/grpo.py::tokenize_prompts`（约 L255）
- 问题：渲染时没有 `tools=`、没有 `open_thinking=True`；rollout/eval 的
  `encode_prompt` 都传了。该函数被 `data/__init__.py` 导出。
- 影响：一旦被使用，prompt 形态与 GRPO 实际训练不一致（工具 schema 丢失、
  think 块被预先闭合）。

### B50. `judge.make_judge` 的环境变量重复

- 状态：已修
- 位置：`src/minideepseek/rl/judge.py::make_judge`（约 L113）
- 问题：`MINIDEEPSEEK_JUDGE_URL` 写了两次，第二个大概率本意是 fallback 变量。
- 影响：当前等价于少一个 fallback，不改变已有行为；属复制粘贴缺陷。

### B51. verl 的 `compute_score` 丢掉了 prompt

- 状态：已修
- 位置：`src/minideepseek/rl/verl.py::compute_score`（约 L76）、
  `to_verl_record`（约 L40）
- 问题：构造 `GRPORecord(prompt=[])`；world 的 `default_world_judge` 里
  `relevance = token_overlap(response, prompt_text)` 恒为 0。
- 影响：verl 路径的 world reward 结构性错误（只剩 factuality/length）。verl 是
  可选且未 GPU 实跑，但接口层应自洽。
- 修复方式：导出数据时把 prompt 放进 `extra_info`，`compute_score` 还原。

## 疑似 / 参考对齐 / 低优先级

### B52. Compressor 的 `wkv`/`wgate` 没有像参考实现那样强制 fp32

- 状态：已修（参考对齐）
- 位置：`src/minideepseek/model/compressor.py::Compressor.forward`（约 L112）
- 问题：直接在 wrapper 的 bf16 autocast 下做投影与 weighted-sum；参考实现用
  `torch.autocast(..., float32)` 包住这两个投影，注释明确 “The compressor runs
  these under fp32 autocast”，并在 pooling 后 `kv.to(dtype)` 再 norm。
- 影响：CUDA bf16 下与参考数值设计有偏差；`scripts/verify_reference.py` 只在
  CPU fp32 比对，抓不到。
- 备注：CPU fp32 下该改动是 no-op，低风险。

### B53. MTP 的无效（跨界填充）位置仍计入路由统计

- 状态：需设计决策，本轮不修
- 位置：`src/minideepseek/model/model.py::mtp_forward`（约 L199）
- 问题：参考实现给 MTP block 传 `padding_mask=~valid`，让 router 的
  `tokens_per_expert_E` / balance loss 排除这些位置；本实现只把
  `prev_hc_hidden` 置零。
- 影响：每篇文档末尾 1 个 token 以 embedding(0) 参与 MTP router 负载统计，
  量很小但系统性。
- 不修原因：需要把 padding mask 从 `MTPBlock` 一路传到 `MoE.forward` /
  router 计数与 balance loss 的 mask，改动面大、收益小，留作后续。

### B54. 验证集 loader 复用训练的 warmup 预算

- 状态：已确认非 bug（保持）
- 说明：每次 `evaluate` 都新建 val 迭代器（`batch_index=0`），因此 val 固定用
  base batch_tokens，保证验证子集稳定（B2 的修复目标）。B42 的修复不会改变
  这一行为。

### B55. AR 预取会推高 `_batches_consumed`

- 状态：已修
- 位置：`src/minideepseek/train/trainer.py::_prefetch_step`（约 L310）
- 问题：预取的 batch 计入 `extra.data_batches`，resume 时可能多跳过几步数据。
- 影响：AR 默认关，影响小；修复成本低。

## 4. 已验证且未发现问题的部分

- **多文档 packed 前向**：同一 packed batch 的 logits 与逐文档前向一致
  （CPU fp32 max |Δ| = 1.9e-6），SWA/CSA/HCA/MTP 的文档边界正确。
- **main / MTP 标签对齐**：用显式 `tokens[1:]` / `tokens[2:]` 独立手算 CE，与
  `compute_loss` 完全一致（单文档、双文档）；MTP 标签掩码与参考
  `roll(labels, shift=d)` 等价。
- **SFT response-only mask**：真实 `minimind/sft_t2t_mini.jsonl` 前 50 条逐条核对，
  mask 只覆盖 assistant 段（含 think/tool_call/`<|im_end|>`），
  `main_token_count == mask.sum()`。
- **数据打包**：现有 debug 数据的 positions 连续、每包 ≤ seq_len、文档首 token
  不是 target。
- **optimizer/scheduler**：Muon 8+2 hybrid NS、`sqrt(max(n,m))*0.18`、参数分组、
  warmup→flat→cosine 边界与文档一致。
- **indexer 蒸馏 loss**：teacher/student/归一化与参考 `SparseIndexerLoss` 对齐；
  top-k 排序在全局常数缩放下不变。
- **FSDP2 训练本身**（无 resume）：2 进程与 1 进程权重一致；
  `torch.nn.utils.clip_grad_norm_` 对 DTensor 计算的是全局 norm（实测 vanilla
  与自定义版本都返回 3256.0，因此 GRPO 用 vanilla 不是 bug）。
- **GRPO/OPD 主损失**：ratio、old/actor logprob 温度一致；反向 KL 方向、全词表、
  response-only 位置正确。
