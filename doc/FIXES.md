# 修复记录（2026-11-05）

> 对应问题清单：[BUGS.md](BUGS.md)。
> 每条记录：问题、修复、验证、剩余风险。修复均先复现/确认，再做最小改动。
> 本次修复 commit：`837a3da`。

## 修复位置索引（当前工作树行号）

| ID | 修了什么 | 关键位置 |
|---|---|---|
| B1/B2 | 数据/验证集改为可重建 factory；验证集每次重置；训练 generator 可循环 | `train/run.py:35`（`_batch_factory`）、`train/run.py:85`（`load_dataset`）、`train/trainer.py:148`（`_val_factory`）、`train/trainer.py:183`（`_next_batch`）、`train/trainer.py:491`（`evaluate` 调用） |
| B3 | 全局 batch token 按 rank 平分；RL prompt 预算按 rank 平分 | `train/run.py:28`（`_per_rank_tokens`）、`train/run.py:35`（使用处）、`rl/grpo.py:129`（`_sample_records`）、`rl/opd.py:121`（`_sample_prompts`） |
| B4/B19 | SFT resume 配置；加载显式忽略 MTP；teacher 运行期 top-k | `configs/sft/{math,code,world}.yaml:31`、`model/checkpoint.py:124`（`filter_state_dict_prefixes`）、`model/checkpoint.py:142`（`load_checkpoint(ignore_prefixes=...)`）、`rl/grpo.py:346`、`rl/opd.py:299`、`rl/opd.py:343`（`load_teacher`）、`eval/evaluate.py:55` |
| B5 | RL resume 只加载模型，不加载 optimizer | `rl/grpo.py:346`（`GRPOTrainer.resume`）、`rl/opd.py:299`（`OPDTrainer.resume`） |
| B6 | context 上限使用模型配置；generate 截断剩余 context | `model/wrapper.py:145`（`attention_metadata`）、`model/wrapper.py:301`（prompt/context 检查）、`model/wrapper.py:308`（`max_new_tokens` 截断） |
| B7 | 去掉 indexer aux loss 的额外 `/world_size` | `model/wrapper.py:223`（`compute_loss`）；删除 `build_trainer` 中的 `aux_loss_world_size` 赋值 |
| B8 | 解析工具 JSON、保留/透传 tools、工具 prompt 进 SFT/GRPO/eval | `tokenizer/tokenizer.py:215`（`normalize_conversation`）、`tokenizer/tokenizer.py:242`（`_parse_json_field`）、`data/sft.py:67`（`encode_sft_record`）、`data/sft.py:103`（`build_sft_dataset`）、`data/grpo.py:48`（`resolved_tools`）、`rl/rollout.py:72-79`（`encode_prompt`）、`eval/evaluate.py:96-100`、`data/opd.py`（prompt normalize） |
| B9 | 数据改存 `.npy` sidecar 并真正 mmap；旧 npz 回退 | `data/io.py:71`（`save_flat_dataset`）、`data/io.py:100`（`load_flat_dataset`）、`data/pretrain.py:35`、`data/sft.py:34` |
| B10 | top-k 配置对齐/curriculum 覆盖/运行期 top-k 入 meta 并被 eval/export/teacher 使用 | `train/curriculum.py:43,96`、`config.py:129`（TeacherSettings）、`train/trainer.py:559,578`（`index_topk` extra）、`rl/grpo.py:321,339`、`rl/opd.py:274,292`、`model/hf/export.py:110`、`eval/evaluate.py:55`、`rl/opd.py:343`、`configs/{grpo,eval,opd}/*.yaml:5`、`configs/pretrain/length_curriculum.yaml:26` |
| B11 | HF export 从 meta 回退可选参数，优先运行期 top-k | `model/hf/export.py:97`（`export_checkpoint`）、`model/hf/export.py:110-115` |
| B12 | 周期 checkpoint 记录已完成步数 | `train/trainer.py:486`（周期 save）、`train/trainer.py:533`（`save(step=...)`）、`rl/grpo.py:297`、`rl/opd.py:250` |
| B13 | resume 校验 tokenizer hash；训练入口校验 dataset meta | `train/trainer.py:583`（`resume`）、`train/run.py:148`（`_validate_dataset`）、`train/run.py`（run 中调用） |
| B14 | 数学答案解析：移除“因此/所以”，优先具体标记并取最后数字 | `rl/reward.py:27`（`_ANSWER_RE`）、`rl/reward.py:47`（`extract_answer`） |
| B15 | rollout/eval 打开 think；format bonus 检查 `</think>` | `rl/rollout.py:73-78`、`eval/evaluate.py:96-100`、`rl/reward.py:114` |
| B16 | router.gate → Muon；compressor.ape → AdamW | `train/optimizer.py:249`（`_is_adamw_parameter`）、`train/optimizer.py:256`（`.ape` 判定） |
| B17 | AR snapshot 在 optimizer.step 前记录 θ_t；自动 AR 步数受 max_steps 限制 | `train/trainer.py:264`（max_steps 上限）、`train/trainer.py:446`（`observe_step` 位置）、`model/anticipatory.py`（docstring） |
| B18 | GRPO 每步同步 vLLM；resume 后同步 | `rl/grpo.py:278`（step 内 sync）、`rl/grpo.py:428`（resume 后 sync）；OPD `rl/opd.py:440` |
| B20 | checkpoint 记录已消费 micro-batch；resume 跳过；curriculum 关闭跳过 | `train/trainer.py:146,189,559,578,619-623`、`config.py:226`（`resume_data`）、`train/curriculum.py:115` |
| B21 | verl `rollout_n` 默认 8；无 val 用 10% split；reward 绝对 import | `rl/verl.py:108`、`rl/verl.py:121,170`、`scripts/export_verl.py:24,52` |
| B22 | tied safetensors 别名 key 写 clone | `model/checkpoint.py:198` |
| B23 | verify_reference 过滤旧 hash-bias key | `scripts/verify_reference.py:66` |
| B24 | grpo_math manifest 预算自洽 | `data/manifests/grpo_math.yaml:4` |
| B26 | RL 前更新 expert bias/清零计数器 | `rl/grpo.py:242,286`、`rl/opd.py:183,230` |
| B27 | `broadcast_buffers` 配置生效 | `train/distributed.py:166,199` |
| B29 | from_padded 保留一行内多文档 | `model/batch.py:141`（`from_padded`） |
| B30 | reference 泄漏检查改为 answer marker/长串 | `data/grpo.py`（`_reference_leaked`/`validate_record`） |
| B39 | 阶段切换 resume 不继承 step/optimizer/data | `config.py`（三个 resume 开关）、`train/trainer.py::resume`、`rl/grpo.py::resume`、`rl/opd.py::resume` |
| B28 | summary tokens 统计所有 step | `train/trainer.py:147,433,508` |
| B33 | 删除 block 级不存在的 `collect_indexer_loss` | `model/model.py`（block/MTP 方法删除，仅保留 model 级实现） |
| B34 | duplicate_indices 只标记后出现的文档 | `data/dedup.py:97` |
| B36 | consistency 传递 index_topk/tie/swiglu；demo prompt 不再二次取模 | `consistency/checks.py:192-283`、`consistency/check.py`（`index_topk` 读取与传参） |
| B38 | 采样仍用 filtered 分布，但返回未过滤策略 logprob | `model/wrapper.py:495,516`（`_sample_tokens`） |

## B1 / B2 — 数据迭代器耗尽 / 验证集 NaN

- 状态：已修
- 改动：
  - `train/run.py`：`load_dataset` 现在返回可重建的 batch factory（单数据集、sources、val 都是），不再返回一次性 generator。
  - `train/trainer.py`：`val_batches` 接受 factory；每次 `evaluate` 重新创建 iterator，固定使用同一段验证集；训练 iterator 通过 `_batch_factory()` 重启。
- 验证：
  - 新增 `test_factory_loader_cycles_and_validation_is_stable`：2-batch factory 训练 4 步，4 次 val loss 全有限。
  - 新增 `test_resume_skips_consumed_batches` 等。
  - 手工复现原 bug：generator 训练 200 步在 step 14 崩溃；修复后不再出现。
- 风险：若外部调用仍传一次性 generator 给 `Trainer`，val 仍无法重置；项目入口已经统一传 factory。

## B3 — 多卡 batch / prompts 放大

- 状态：已修
- 改动：
  - `train/run.py::_per_rank_tokens`：`batch_tokens`/`batch_tokens_final` 按 world_size 平分（最小 1）；config 语义回到 global micro-batch。
  - `rl/grpo.py::_sample_records`、`rl/opd.py::_sample_prompts`：`prompts_per_step` 按 world_size 平分；world=1 不变。
- 验证：
  - 新增 `test_global_batch_tokens_are_split_by_world`、`test_grpo_prompt_budget_is_global`。
  - 手工：debug 数据 world=1/2 的 batch token 分别约 251/123，global 接近配置 256。
- 风险：`batch_tokens < world_size * seq_len` 时会退化为每 rank 1 个 pack，global 略超配置；属最小可训练约束。

## B4 — SFT / GRPO / OPD 配置链

- 状态：已修
- 改动：
  - `configs/sft/{math,code,world}.yaml` 增加 `train.resume: checkpoints/pretrain_full/final`。
  - `model/checkpoint.py::load_checkpoint` 增加显式 `ignore_prefixes` 和 `filter_state_dict_prefixes`；只允许忽略明确列出的前缀。
  - `GRPOTrainer.resume` / `OPDTrainer.resume` / `load_teacher` / `eval._load_model` 加载时忽略 `mtp_layers.`；主权重仍 strict。
  - `TeacherSettings.index_topk` + `load_teacher` 使用；OPD 配置显式设置 teacher top-k。
- 验证：
  - 新增 `test_load_checkpoint_can_ignore_mtp_heads`：n_mtp=1 → n_mtp=0 可直接加载，主 logits 完全一致；不加参数仍报错。
  - 手工用 n_mtp=1 SFT checkpoint 构建 GRPO，加载成功。
- 风险：忽略 MTP 只针对该前缀；其他 key 不匹配仍然报错。

## B5 — RL resume 覆盖学习率

- 状态：已修
- 改动：`GRPOTrainer.resume` / `OPDTrainer.resume` 只加载模型权重，不加载 `optimizer.pt`（与 D22 一致）；分布式路径也不再加载 optimizer。
- 验证：
  - 新增 `test_grpo_resume_does_not_override_rl_lr`：SFT ckpt base_lr=2e-5，GRPO 配 2e-6，resume 后仍为 2e-6。
- 风险：RL 不再继承 SFT 的优化器动量；这是原设计意图。

## B6 — context 检查恒真

- 状态：已修
- 改动：
  - `MiniDeepSeekModel.attention_metadata` 用 `self.max_context_length`，不再用当前 batch 最大长度。
  - `generate` 前检查 prompt 长度，并把 `max_new_tokens` 截到剩余 context；prompt 已占满时给明确 ValueError。
- 验证：
  - 新增 `test_forward_rejects_sequence_longer_than_context`、`test_generate_caps_new_tokens_to_context`。
  - 原 `IndexError: index 32 out of bounds` 变为受控错误/截断。
- 风险：长 prompt 不再静默越界；数据准备需按模型 context 截断。

## B7 — FSDP indexer aux loss 多除 world_size

- 状态：已修
- 改动：`MiniDeepSeekModel.compute_loss` 不再把 indexer loss 除以 DP size；删除 `aux_loss_world_size` 属性及 `build_trainer` 的赋值。
- 验证：
  - 手工 2 进程 `build_trainer`：world=1/2 的 model state digest 完全一致，indexer loss 日志均为 4.0274e-6（修复前 world2 是一半）。
- 风险：文档 D27 需要同步修正（见 BUGS/决策更新）。

## B8 — tool 数据崩溃 / schema 丢失

- 状态：已修
- 改动：
  - `tokenizer.normalize_conversation` 解析字符串形式的 `tools`/`tool_calls`，保留 tools 字段；非法 JSON 抛 ValueError。
  - `data/sft.encode_sft_record` 支持系统消息里的 tools；`build_sft_dataset` 跳过非法记录。
  - `GRPORecord.resolved_tools()`；`validate_record`、rollout `encode_prompt`、eval 生成都传 tools。
  - `data/opd.load_opd_prompts` 也做 normalize。
- 验证：
  - 新增 `test_sft_tool_record_keeps_tool_schema`：MiniMind 风格 tool 记录不再 crash，渲染文本包含 `# Tools`/函数名，SFT dataset 可构建。
  - 手工：真实 `sft_t2t_mini.jsonl` 第一条 tool 记录（index 771074）可编码。
- 风险：非法 tool JSON 会被跳过而非中断；若需要严格报错可加日志计数。

## B9 — npz 非 mmap

- 状态：已修
- 改动：
  - `data/io.save_flat_dataset` 改为写独立 `.npy` sidecar（`<stem>.input_ids.npy` 等）+ JSON meta；`load_flat_dataset` 用 `mmap_mode="r"`，旧 `.npz` 仍可回退读取。
  - `PackedPretrainDataset` / `SFTDataset` 使用统一 loader；每 batch 复制成可写 tensor，避免 memmap 只读 warning。
- 验证：
  - 新增 `test_flat_dataset_sidecars_are_memmapped`：四个数组均为 `np.memmap`。
  - 手工：legacy debug npz 仍能读；新数据 mmap。
- 风险：新建数据集不再生成 `.npz` 文件，路径仍沿用 `.npz`；外部直接用 `np.load` 的脚本需要改用 `load_flat_dataset`。

## B10 — index_topk 未绑定

- 状态：已修（配置/运行时元数据层面）
- 改动：
  - GRPO/OPD 配置显式 `index_topk: 1024`；eval 配置改为 1024（对应 GRPO ckpt）。
  - curriculum stage 增加 `index_topk` 覆盖字段；8K stage 设 2048。
  - Trainer/GRPO/OPD checkpoint meta 的 extra 记录运行时 `index_topk`；HF export、eval、teacher 优先使用该运行时值。
- 验证：新增 curriculum 测试；HF export meta 测试；手工 export config 值正确。
- 风险：旧 checkpoint 没有运行时字段时仍回退静态配置；SFT 训练末期 64、GRPO 1024 的差异通过显式配置/运行时字段处理。

## B11 — HF 导出静默丢设置

- 状态：已修
- 改动：`export_checkpoint` 仅在参数为 None 时回退 meta；优先使用 runtime index_topk；flavor/seq_len 也支持 meta 回退。
- 验证：新增 `test_export_checkpoint_uses_meta_runtime_settings`；手工无 flavor/seq 参数也能导出。
- 风险：无。

## B12 — 周期 checkpoint step 差一

- 状态：已修
- 改动：`Trainer/GRPO/OPD.save` 增加可选 `step`；周期保存传 `self.step+1`/`self.step_index+1`，未改变 final 行为。
- 验证：新增 `test_periodic_checkpoint_step_matches_name`；手工 `step_000001` meta=1。
- 风险：无。

## B13 — tokenizer hash / dataset 校验

- 状态：已修
- 改动：`Trainer.resume` 比较 checkpoint tokenizer hash；`train/run` 启动时校验 dataset 的 tokenizer hash、vocab、seq_len 与模型配置。
- 验证：新增 `test_resume_rejects_tokenizer_hash_mismatch`；完整测试通过。
- 风险：旧数据 meta 缺失时跳过；不阻止无 meta 的手工数据。

## B14 — 数学 reward 解析

- 状态：已修
- 改动：`_ANSWER_RE` 去掉“因此/所以”；捕获后若含数字取最后一个数字，支持“42 元”等写法。
- 验证：新增 `test_math_extraction_handles_connective_prefixes`。
- 风险：`答案是 3+4` 会取 4；这是“最后数字”启发式的已知取舍，可用 `\boxed{}` 避免。

## B15 — format_bonus 恒 0

- 状态：需设计决策，未改
- 说明：当前 chat template 的 generation prompt 预先闭合 think；是否让 rollout `open_thinking=True` 属于行为决策，暂不静默修改。

## B16 — Muon/AdamW 分组

- 状态：已修
- 改动：`_is_adamw_parameter` 不再把 `router.gate` 归 AdamW，改为 Muon；`.ape`（compressor/indexer compressor）归 AdamW。
- 验证：新增 `test_router_gate_and_compressor_ape_grouping`。
- 风险：与旧 checkpoint 的优化器状态分组无关（参数集合相同，只是优化器不同）；旧训练无法严格复现。

## B17 — AR snapshot 时序

- 状态：部分修复
- 改动：`Trainer.train_step` 在 `optimizer.step()` 之前 `observe_step`，使 `stale_snapshot(step-delta)` 语义正确。
- 验证：现有训练/AR 测试通过；未新增专门数值测试。
- 剩余：`AnticipatoryRoutingSettings.max_steps` 仍未实现（语义需确认）。

## B18 — GRPO vLLM 权重同步

- 状态：已修
- 改动：GRPO 每次 optimizer.step 后 `engine.sync_weights`；build 时 resume 后同步一次（torch backend no-op）。OPD 也补 resume 后同步。
- 验证：torch backend 测试通过；vLLM 仍无 GPU 实机。
- 风险：vLLM 重建/权重同步路径未在 CUDA 验证。

## B19 — teacher 运行期设置

- 状态：已修
- 改动：`TeacherSettings.index_topk`；`load_teacher` 优先使用 checkpoint 运行时 topk；忽略 MTP key。
- 验证：手工/后续测试计划；完整测试通过。
- 风险：teacher 的 `swiglu_limit` 仍硬编码 10.0（当前所有配置一致）。

## B20 — resume 数据位置 / warmup

- 状态：已修
- 改动：Trainer 记录已消费 micro-batch 数到 checkpoint extra；resume 时用同一 factory 跳过；curriculum stage 有独立数据集，`resume_data=False`。
- 验证：新增 `test_resume_skips_consumed_batches`、curriculum 测试；手工 resume 后第一个 batch 是第 3 个。
- 风险：AR 预取的下一批不保存；rollback 不恢复数据位置（原有近似）。

## B21 — verl 适配

- 状态：已修
- 改动：`build_verl_config` 增加 `rollout_n` 默认 8；`export_verl` 无 `--val` 时用 `split_records(0.1)` 而不是训练集前 16 条；reward 文件改绝对 import 以便 verl 按路径加载。
- 验证：`test_integrations` 9 passed。
- 风险：verl 本体未安装/未 GPU 实跑；OPD 的 teacher_key 映射仍是待验证设计。

## B22 — tied safetensors

- 状态：已修
- 改动：`export_safetensors` 对共享 storage 的别名 key 写 clone，保证 strict import 有 `lm_head.weight`。
- 验证：新增 `test_safetensors_roundtrip_tied`；手工 import 成功。

## B23 — verify_reference

- 状态：已修
- 改动：脚本用目标模型 key 集合过滤 reference state dict，兼容 D34 后旧 hash-bias key。
- 验证：`scripts/verify_reference.py --seq-len 160 --mtp` 通过（worst block 1.5e-4）。

## B24 — grpo_math manifest

- 状态：已修
- 改动：`total_tokens` 20M → 10M，使 target/available 与 max_epochs=2 自洽。
- 验证：`data.prepare manifest` 返回 ok。

## B38 — 采样 logprob 使用过滤后分布

- 状态：已修
- 改动：`_sample_tokens` 仍从 filtered 分布采样，但返回的 logprob 用未过滤的 `scaled` 分布，保证 GRPO 的 old logprob 是策略概率。
- 验证：默认 top-k/p（无过滤）行为不变；完整测试通过。
- 风险：非默认过滤只在 vLLM/额外配置中使用，尚未专门测试。

## 待办

- B25–B37 中剩余配置空转、RL expert bias、`from_padded` 多文档、metrics token 统计等尚未逐项处理。
- 需要补的测试：B18 的 fake generator、B19 teacher 运行时、B7 的多进程验证脚本固化、B17 数值测试。

## B28 — summary token 统计

- 状态：已修
- 改动：`Trainer` 维护 `_tokens_seen`，每个训练步累加 `StepMetrics.tokens`；summary 使用该计数，不再只统计 `log_every` 打印的 step。
- 验证：新增 `test_summary_tokens_count_all_steps`（4 步、log_every=2，summary tokens=20）。
- 位置：`src/minideepseek/train/trainer.py:147,433,508`。

## B33 — block 级 `collect_indexer_loss`

- 状态：已修
- 改动：删除 `DeepSeekV4TransformerBlock` / `MTPBlock` 上引用不存在 `self.layers` 的方法；`DeepSeekV4Model.collect_indexer_loss` 保留并继续被 wrapper 使用。
- 验证：完整测试通过（原方法无调用方）。
- 位置：`src/minideepseek/model/model.py`（删除两个 block 级方法，保留 model 级实现）。

## B34 — duplicate_indices 聚类

- 状态：已修
- 改动：`pairs` 已按 `(left,right)` 排序，右侧文档才是后出现的重复项；直接返回所有 `right`。
- 验证：`test_minhash_duplicate_detection` 通过。
- 位置：`src/minideepseek/data/dedup.py:97`。

## B36 — consistency 工具配置传输

- 状态：已修
- 改动：
  - `check_checkpoint_consistency` / `run_all_checks` 增加 `index_topk/tie_word_embeddings/swiglu_limit/indexer_loss_coeff` 并传给重建模型；
  - CLI 优先读 checkpoint 运行期 `index_topk`；
  - `--demo` 不再对 mini flavor 二次取模 2048（tokenizer 已处理 vocab limit）。
- 验证：`minideepseek.consistency.check --demo` 通过（checkpoint `max_abs_diff=0`）。
- 位置：`src/minideepseek/consistency/checks.py:192-283`、`src/minideepseek/consistency/check.py`。

## B38 — 采样过滤后的 logprob

- 状态：已修
- 改动：`_sample_tokens` 仍用 top-k/top-p 的 `filtered` 分布采样，但返回 `scaled`（未过滤策略）的 logprob；GRPO old logprob 与 actor 一致。
- 验证：默认无过滤时行为不变；完整测试通过（尚未补专门 top-k 数值测试）。
- 位置：`src/minideepseek/model/wrapper.py:483-516`。

## B15 — format bonus / think 开关

- 状态：已修
- 改动：
  - rollout/eval 的 generation prompt 使用 `open_thinking=True`，与 SFT 数据的 think 结构一致；
  - `math_reward` 的 format bonus 检查响应里是否出现 `</think>`（开标签在 prompt 中）。
- 验证：新增 `test_rollout_prompt_opens_thinking_and_format_bonus_is_reachable`；debug eval 可运行。
- 位置：`src/minideepseek/rl/rollout.py:73-78`、`src/minideepseek/eval/evaluate.py:96-100`、`src/minideepseek/rl/reward.py:114`。
- 风险：改变了 GRPO/eval 的 prompt 形态；这是为了修复 SFT 训练/rollout prompt 不一致。

## B29 — from_padded 多文档

- 状态：已修
- 改动：按每行 `positions` 的 0 重置切分多个文档，再交给 `from_sequences`；`target_mask` 同步切分。
- 验证：新增 `test_from_padded_keeps_multiple_documents_in_one_row`；原有单文档测试保持通过。
- 位置：`src/minideepseek/model/batch.py`（`from_padded`）。

## B39 — 阶段切换 resume

- 状态：已修
- 改动：`TrainSettings` 新增 `resume_data` / `resume_optimizer` / `resume_step`。
  - SFT 从 pretrain 初始化：三者 false（新数据、新优化器、step=0）；
  - GRPO/OPD 从 SFT/pretrain 初始化：`resume_step=false`（模型权重仍加载，优化器不加载）；
  - curriculum stage：`resume_data=false`，保留 optimizer 与累计 step。
- 验证：新增 `test_stage_transition_keeps_fresh_optimizer_and_data`；debug SFT → GRPO → OPD 实跑通过。
- 位置：`src/minideepseek/config.py:226-232`、`src/minideepseek/train/trainer.py:583-626`、`src/minideepseek/rl/grpo.py:346`、`src/minideepseek/rl/opd.py:299`、`configs/sft/*.yaml:31-34`、`configs/{grpo,opd}/*.yaml`。

## B26 — RL 阶段 expert bias / 计数器

- 状态：已修
- 改动：GRPO/OPD 在裁剪梯度后、optimizer.step 前同步 `tokens_per_expert_E`（分布式 all-reduce SUM）并调用 `update_expert_biases`；计数器每步清零。
- 验证：`test_grpo_syncs_generator_after_update` 额外断言所有 MoE 计数器为 0；完整测试通过。
- 位置：`src/minideepseek/rl/grpo.py:242,286`、`src/minideepseek/rl/opd.py:183,230`。
- 风险：RL 的 routing bias 现在会随 rollout actor 的负载变化；与 pretrain/SFT 机制一致。

## B27 — 空转配置

- 状态：部分修复
- 改动：`DistributedSettings.broadcast_buffers=false` 现在会跳过 buffer 广播（默认 true 行为不变）；`DataSettings.max_records/max_tokens/tokenizer_dir/domain` 是 prepare 侧/预留字段，训练入口未消费，已在 BUGS.md 标注。
- 验证：FSDP2 2 进程默认路径通过。
- 位置：`src/minideepseek/train/distributed.py:166,199`。

## B30 — reference 泄漏误杀

- 状态：已修
- 改动：短 reference 只有出现在 `final answer` / `answer is/[:]` / `答案是` / `答案为` / `\boxed{` 等 answer marker 之后才判泄漏；长度 ≥8 的 reference 仍按整串出现判断。
- 验证：新增 `test_reference_leak_does_not_reject_question_digits`；原 `"Answer is 2"` 泄漏测试仍通过。
- 位置：`src/minideepseek/data/grpo.py`（`_reference_leaked`、`validate_record`）。
- 风险：没有 marker 的短答案泄漏可能漏检，但避免大量 `1+2` 类问题被误杀。
