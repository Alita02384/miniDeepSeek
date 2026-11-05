# 审计问题清单（2026-11-05）

> 状态：`待修` / `已修` / `已确认非 bug` / `需设计决策`。
> 原则：每项先复现/确认，再最小化修复；修复后补回归测试，避免无关行为变化。
> 本轮审计基于 commit `5f0d30c`，`pytest -q` 基线 130 passed（CPU debug 模型）。
> 修复后：P0/P1 及可复现的 P2 已修；标为“文档/策略/设计”的项是待产品决策的设计缺口，
> 不是当前可复现的静默逻辑错误。最终回归 156 passed。

## P0 — 会让正式训练跑不完、跑错或结果不可信

### B1. 数据迭代器耗尽直接崩溃

- 状态：已修
- 位置：`src/minideepseek/train/run.py`（`load_dataset` 返回 generator）、`src/minideepseek/train/trainer.py`（`_next_batch`）
- 问题：`Trainer._next_batch` 的“重启 iterator”只对 list/可重复迭代对象有效；`load_dataset` 传入的是 `iter_batches(...)` 返回的 generator，耗尽后第二次 `next()` 直接抛 `StopIteration`。
- 复现：debug pretrain 数据集 421 packs，`batch_tokens=4096, seq_len=128`，训练 200 步 → step 14 崩溃。
- 影响：多卡分片后每个 rank 数据更少，`full_6b` 等正式训练会在早期崩溃。
- 修复方向：让训练/验证数据入口返回可重建的 factory；`Trainer` 每轮用 factory 新建 iterator。

### B2. 验证集第二次开始 NaN / 每次用不同子集

- 状态：已修
- 位置：`src/minideepseek/train/trainer.py::evaluate`
- 问题：`_val_iter` 只创建一次，`evaluate` 消耗后不重置；耗尽后返回 `nan`。即使未耗尽，每次验证的是后续不同 batch，指标不可比。
- 复现：4 步、`val_every=1, val_steps=1` → `8.05, nan, nan, nan`。
- 修复方向：每次验证从 val factory 重新创建 iterator，并使用同一批验证样本；与 B1 共用入口。

### B3. 多卡 batch / prompts_per_step 被 world_size 放大

- 状态：已修
- 位置：`src/minideepseek/train/run.py::_batch_factory`、`src/minideepseek/rl/grpo.py::_sample_records`、`src/minideepseek/rl/opd.py::_sample_prompts`
- 问题：配置和文档按 global batch 写，但 loader/RL 按 per-rank 使用。
- 证据：`full_6b` 注释是 `12000 steps * 0.5M tokens`；8 卡时 `8 × 8 accum × 65536 ≈ 4.2M tokens/step`，约 50B tokens，而不是 6B。SFT 8 卡约 5 epoch；GRPO/OPD 全局 prompts 为 world 倍。
- 修复方向：在 `train/run.py` 按 `shard_world_size` 缩放 `batch_tokens`/`batch_tokens_final`；RL 按 world 缩放 `prompts_per_step`（world=1 行为不变）。

### B4. SFT → GRPO/OPD 生产配置链不匹配

- 状态：已修
- 位置：`configs/sft/*.yaml`、`configs/grpo/*.yaml`、`configs/opd/*.yaml`、`configs/opd/three_teacher.yaml`、`src/minideepseek/model/checkpoint.py`
- 问题一：SFT 三个配置没有 `train.resume`，specialist 实际从随机初始化开始。
- 问题二：SFT/pretrain 为 `n_mtp_layers: 1`，GRPO/OPD 为 `0`；strict load 报 unexpected `mtp_layers.*`。
- 问题三：`three_teacher` 的 world teacher 是 SFT ckpt（n_mtp=1），`TeacherSettings.n_mtp_layers` 默认 0，同样报错。
- 修复方向：SFT 配置补 `train.resume: checkpoints/pretrain_full/final`；RL/eval/teacher 加载时显式忽略 `mtp_layers.`（仅该前缀），主权重严格加载。

### B5. RL resume 误加载旧优化器状态并覆盖 RL 学习率

- 状态：已修
- 位置：`src/minideepseek/rl/grpo.py::GRPOTrainer.resume`、`src/minideepseek/rl/opd.py::OPDTrainer.resume`
- 问题：只要存在 `optimizer.pt` 就加载；`base_lr` 被恢复成 SFT/pretrain 的值，scheduler 随后使用旧 LR。
- 证据：GRPO 配 `2e-6`，从 SFT checkpoint resume 后 `base_lr=2e-5`；OPD 从 pretrain resume 会到 `6e-4` 量级。与 D22“不加载 RL 优化器状态”冲突。
- 修复方向：RL resume 只加载模型权重（并忽略 MTP），不加载 optimizer；LR 由当前 RL 配置决定。

### B6. context length 检查恒真

- 状态：已修
- 位置：`src/minideepseek/model/wrapper.py::attention_metadata`
- 问题：把当前 batch 的 `max(positions)+1` 作为允许上限传给 `_get_attention_metadata`，检查永远不触发；越界时在 RoPE cache 抛 `IndexError`。
- 复现：seq_len=32、prompt 25 + 生成 10 → `IndexError: index 32 is out of bounds`。
- 修复方向：使用 `self.model.max_context_length` 作为上限；长序列给出明确 ValueError。

### B7. FSDP2 下 indexer aux loss 被多除一次 world_size

- 状态：已修
- 位置：`src/minideepseek/model/wrapper.py::compute_loss`（`aux_loss_world_size`）、`src/minideepseek/train/trainer.py::build_trainer`
- 问题：indexer loss 已按 local tokens 归一化，FSDP 梯度平均后正好得到 global per-token；再除 world 会让 aux 项缩小 W 倍。
- 证据：world=2 时 `indexer_loss` 日志正好是 world=1 的一半（4.03e-6 → 2.01e-6），权重 digest 不一致。
- 修复方向：去掉 wrapper 里的额外除法及相关属性；修正 D27 文档。

### B8. SFT tool 数据崩溃 / tool schema 丢失

- 状态：已修
- 位置：`src/minideepseek/tokenizer/tokenizer.py::normalize_conversation`、`src/minideepseek/data/sft.py`、`src/minideepseek/rl/rollout.py::encode_prompt`、`src/minideepseek/eval/evaluate.py`
- 问题：MiniMind SFT 的 `tool_calls` 是 JSON 字符串，`normalize_conversation` 保留字符串，模板逐字符迭代 → `TypeError`；system message 的 `tools` 被丢弃；GRPO/eval 渲染 prompt 时也没有传 `record.tools`。
- 证据：`sft_t2t_mini.jsonl` 共 84,832 条 tool 记录，第一条 index 771,074；debug 前 50 条无 tool，所以测试全过。
- 修复方向：解析字符串 `tools`/`tool_calls`，保留并透传 `tools`；SFT build 对无效记录跳过；补 tool 记录测试。

### B9. 数据 npz 并非 mmap，6B 数据会整块进内存

- 状态：已修
- 位置：`src/minideepseek/data/io.py::save_flat_dataset/load_flat_dataset`、`src/minideepseek/data/pretrain.py`、`src/minideepseek/data/sft.py`
- 问题：`np.savez_compressed` + `np.load(mmap_mode="r")` 对 npz 成员实际解压为 ndarray，不是 memmap。
- 证据：6B tokens 的 input/positions/target_mask 约 54GB/rank；注释“keeps 6B-token datasets out of RAM”错误。
- 修复方向：保存为独立 `.npy` 侧车文件并用 `np.load(mmap_mode="r")`；保留旧 `.npz` 读取回退，避免现有 debug 数据失效。

## P1 — 静默行为漂移 / 结果系统性偏差

### B10. `index_topk` 未与 checkpoint/训练配置绑定

- 状态：已修
- 位置：`configs/eval/*.yaml`、`configs/grpo/*.yaml`、`configs/opd/*.yaml`、`configs/pretrain/length_curriculum.yaml`、`src/minideepseek/train/curriculum.py`
- 证据：GRPO 默认 1024、eval 写 64；同一权重 logits 最大差约 2.85。curriculum 8K stage 继承 1024，但 dense-over-compressed 应为 2048。
- 修复方向：显式写 GRPO/OPD/eval 的 topk；curriculum stage 增加 `index_topk` 覆盖；文档说明运行期设置需与 checkpoint 对齐。

### B11. HF 导出 CLI 静默关闭 SwiGLU clamp / 重置 index_topk

- 状态：已修
- 位置：`scripts/export_hf.py`、`src/minideepseek/model/hf/export.py::export_checkpoint`
- 问题：CLI 未传的 `--swiglu-limit/--index-topk` 以 `None` 传入，`setdefault` 不覆盖已存在 None；导出 config 变成 `swiglu_limit: null, index_topk: null`。
- 修复方向：只在参数非 None 时覆盖；flavor/seq_len 缺失时回退 meta。

### B12. 周期 checkpoint step 差一

- 状态：已修
- 位置：`src/minideepseek/train/trainer.py::train`、`src/minideepseek/rl/grpo.py::train`、`src/minideepseek/rl/opd.py::train`
- 证据：`step_000001/meta step=0`、`step_000002/meta step=1`、`final/meta step=2`；resume 会重跑一步。
- 修复方向：周期 save 的 step 记录为“已完成步数”(self.step+1)，或先递增再 save；保持 final 不变。

### B13. Trainer resume 不校验 tokenizer hash

- 状态：已修
- 位置：`src/minideepseek/train/trainer.py::resume`、`src/minideepseek/train/run.py`
- 证据：hash=`AAA` 保存、`BBB` 恢复不报错；训练数据集 `meta.tokenizer_hash/seq_len` 也未校验。
- 修复方向：resume 时比较 meta hash；训练前比较 dataset meta 与当前 tokenizer/model seq_len（缺失时仅 warning）。

### B14. 数学 reward 答案解析假阴性

- 状态：已修
- 位置：`src/minideepseek/rl/reward.py::_ANSWER_RE/extract_answer`
- 证据：`"因此答案是 42"` → extracted `"答案是 42"` → False；`"所以答案是42"`、`"因此 x = 42"` 同样 False。
- 修复方向：去除“因此/所以”作为 answer marker；优先具体标记；捕获后若含数字，取最后数字作为候选。

### B15. `format_bonus` 基本恒为 0

- 状态：已修
- 位置：`src/minideepseek/rl/rollout.py:73`、`src/minideepseek/eval/evaluate.py:96`、`src/minideepseek/rl/reward.py:114`
- 证据：chat template 的 generation prompt 已预先闭合空 think 块；响应里没有 `<think>`，现有 `reports/eval_debug_math.json` 的 `format_rate=0.0`。
- 修复：rollout/eval 使用 `open_thinking=True`；format bonus 检查响应中的 `</think>`；新增 `test_rollout_prompt_opens_thinking_and_format_bonus_is_reachable`。

### B16. Muon/AdamW 参数分组与官方 recipe 相反

- 状态：已修
- 位置：`src/minideepseek/train/optimizer.py::_is_adamw_parameter`
- 证据：官方 recipe 的 Muon 正则包含 `router.gate.weight`，`compressor.ape` 留给 AdamW；当前相反。
- 修复方向：移除 router.gate 的 AdamW 标记；把 `.ape` 归入 AdamW；补分组测试。

### B17. AR fallback off-by-one，`max_steps` 未生效

- 状态：已修（自动 AR 步数上限）
- 位置：`src/minideepseek/model/anticipatory.py`、`src/minideepseek/train/trainer.py`
- 问题：`observe_step` 在 optimizer 更新后记录，却按 `step-delta` 解读；delta=1 的 fallback 实际用当前 gate。`max_steps` 未被读取。
- 修复：在 optimizer.step 前记录 θ_t；自动 AR 步数取 `loss_spike_ar_steps` 与 `max_steps` 的较小值。

### B18. GRPO vLLM 路径不更新权重

- 状态：已修
- 位置：`src/minideepseek/rl/grpo.py`、`src/minideepseek/rl/backends/__init__.py`
- 问题：GRPO 从不调用 `engine.sync_weights`；engine 在 `resume` 之前构建，首次 rollout 用初始化权重，之后一直旧权重。
- 修复方向：resume 后同步一次；每一步 optimizer.step 后同步（与 OPD 一致）；torch engine 是 no-op。

### B19. OPD teacher 构建参数与 checkpoint 不一致

- 状态：已修
- 位置：`src/minideepseek/rl/opd.py::load_teacher`、`configs/opd/three_teacher.yaml`
- 问题：不传 `index_topk`，n_mtp 依赖默认 0；world teacher 是 SFT ckpt（n_mtp=1、末期 topk=64）。
- 修复方向：load_teacher 支持 `index_topk` 并忽略 MTP key；配置显式给出需要保留的运行期设置。

### B20. resume 不恢复数据位置 / batch warmup 重置

- 状态：已修
- 位置：`src/minideepseek/train/trainer.py::resume`、`src/minideepseek/train/run.py`
- 证据：训练 2 步后 resume，第一个 batch 仍是 `[1,2,3,4]`。
- 修复方向：短期在 checkpoint meta 记录已消费 micro-batch 数/epoch，resume 时跳过；或至少文档明确并拒绝中途 resume？优先实现 skip。

### B21. verl 适配三处硬问题

- 状态：已修
- 位置：`src/minideepseek/rl/verl.py`、`scripts/export_verl.py`
- 问题：`rollout.n=1` 使 GRPO 组内优势为 0；无 `--val` 时用训练集前 16 条做验证；custom reward 文件相对 import 在 verl 按路径加载时失败。
- 修复方向：`n` 参数化并默认 8；用 `split_records` 产生 val；reward 文件改绝对 import。

### B22. tied 模型 safetensors 往返失败

- 状态：已修
- 位置：`src/minideepseek/model/checkpoint.py::export_safetensors`
- 证据：`export_safetensors` 跳过共享 storage 的 `lm_head.weight`，`import_safetensors(strict=True)` 报 missing key。
- 修复方向：对共享 storage 的别名 key 写独立 clone（safetensors 要求），或显式展开 tied 权重。

### B23. `verify_reference.py` 因 D34 失效

- 状态：已修
- 位置：`scripts/verify_reference.py`
- 证据：reference state dict 含旧 `layers.0/1.feed_forward.expert_bias_E`，strict load 报 unexpected key。
- 修复方向：脚本过滤 hash 层旧 key（与 D34 迁移说明一致）。

### B24. `grpo_math` manifest 自身校验不过

- 状态：已修
- 位置：`data/manifests/grpo_math.yaml`
- 证据：`gsm8k_train` 4.0 epoch > 2；`math_train` 2.67 > 2。
- 修复方向：修正 `available_tokens` 或 `max_epochs`，使校验通过；manifest 接入训练入口另行评估。

### B39. 阶段切换 resume 继承 step/optimizer/data 位置

- 状态：已修
- 位置：`config.py`（`resume_data/resume_optimizer/resume_step`）、`train/trainer.py::resume`、`rl/grpo.py::resume`、`rl/opd.py::resume`
- 问题：SFT 从 pretrain checkpoint 初始化时会继承 pretrain 的 step（约 12000），导致 `while step < 500` 直接跳过 SFT；同时继承 pretrain 优化器 base_lr 与已消费 batch 数；GRPO/OPD 从 SFT/pretrain 初始化同样会跳过 RL。
- 修复：新增三个 resume 开关；SFT/GRPO/OPD 配置从 base 初始化时全部 false；curriculum stage 仅 `resume_data=false`（保留累计 step/optimizer）。
- 验证：新增 `test_stage_transition_keeps_fresh_optimizer_and_data`；debug SFT→GRPO→OPD 实跑通过。

## P2 — 配置空转 / 低优先级

- B25 `RLSettings.kl_coeff`、`min_response_tokens`、RL `grad_accum`/`val_path` 由 `_validate_rl_settings` 显式拒绝；`decay_start_ratio` 已接入 GRPO/OPD scheduler（状态：已修；RL 验证集仍未实现）。
- B26 GRPO/OPD 不更新 expert bias；`tokens_per_expert_E` 只涨不清（状态：已修，见 FIXES.md）。
- B27 `DistributedSettings.broadcast_buffers` 已生效；`DataSettings.max_records/max_tokens/tokenizer_dir/domain` 是 prepare 侧/预留字段，训练入口未消费（状态：部分修复/文档）。
- B28 `MetricsLogger.summary` 的 `tokens` 只统计 log step（状态：已修，见 FIXES.md）。
- B29 `from_padded` 丢一行内多文档边界（状态：已修）。
- B30 `data/grpo.validate_record` 用 `reference in rendered_prompt` 做泄漏判断，短数字答案易误杀（状态：已修，改为 answer marker/长串判断）。
- B31 `data/quality.code_license_ok` 对无 header 文件直接放行，与“剔除未知 header”描述不一致（状态：文档/策略）。
- B32 13-gram 污染检查、manifest epoch 校验未接训练入口（状态：设计缺口）。
- B33 `MTPBlock.collect_indexer_loss` 引用不存在的 `self.layers`（状态：已修，删除死方法）。
- B34 `duplicate_indices` 成对重复时可能把两个都标 duplicate（状态：已修）。
- B35 HF 导出 remote-code 文件仍 import `minideepseek.model.*`，非真正自包含（状态：文档/设计）。
- B36 `consistency.check --demo` 对 mini flavor 也取模 2048；`check_checkpoint_consistency` 不传 index_topk/tie/swiglu（状态：已修）。
- B37 `Trainer` grad accumulation 按 micro-batch 等权而非 token 等权；多卡主 loss 也按 rank 等权（状态：需设计决策）。
- B38 `generate` 在 `top_k/top_p` 非默认时返回过滤后 logprob，和 actor raw logprob 不一致（状态：已修）。
