# 修复记录 02（2026-11-06）

> 对应问题清单：[BUGS_02.md](BUGS_02.md)。
> 每条记录：问题确认、修复方式、验证、剩余风险。
> 修复原则：先复现/确认，再最小改动；不改无关行为；每条补回归测试。
> 修复前基线：`17cbb0c`，`pytest -q` **156 passed**；修复后 **165 passed**。
> 涉及源码 16 个文件、测试 7 个文件、脚本 1 个。

## 修复位置索引

| ID | 修了什么 | 关键位置 | 回归测试 |
|---|---|---|---|
| B40 | FSDP2 optimizer state 恢复时保持参数 layout；Muon 恢复后把分片 buffer 还原为全量 | `train/distributed.py:343`（`_state_to_local`）、`train/distributed.py:429`（`_load_single_optimizer`）、`train/optimizer.py:135` | `tests/test_distributed.py::test_fsdp2_trainer_save_and_resume`（resume 后再走一步）、`scripts/test_distributed.py`（2 进程 resume step） |
| B41 | balance loss 跳过 hash 路由层 | `train/losses.py:72` | `tests/test_trainer.py::test_balance_loss_skips_hash_routers` |
| B42 | batch-token warmup 用共享 schedule，跨迭代器重启/epoch 继续 | `data/pretrain.py:283`（`BatchTokenSchedule`）、`data/pretrain.py:323`、`data/sft.py:189`、`train/run.py:62`（`_batch_factory`） | `tests/test_data.py::test_batch_token_warmup_persists_across_loader_restarts` |
| B43 | `MetricsLogger.summary` 对 key 取并集，val 记录不再清空训练指标 | `train/metrics.py:46` | `tests/test_trainer.py::test_factory_loader_cycles_and_validation_is_stable`（追加 summary 断言） |
| B44 | GRPO 顶层 string `tools` 先解析成 list | `data/grpo.py:126`（`_record_tools`）、`data/grpo.py:172/191` | `tests/test_data.py::test_grpo_top_level_string_tools_are_parsed` |
| B45 | consistency CLI 在 `n_mtp_layers=0` 时忽略 `mtp_layers.` | `consistency/check.py:82` | `tests/test_consistency.py::test_cli_ignores_mtp_heads_for_zero_mtp_model` |
| B46 | GRPO/OPD 在 ddp 模式下 all-reduce 梯度 | `rl/grpo.py:283`、`rl/opd.py:231` | `tests/test_grpo.py::test_grpo_ddp_all_reduces_gradients`、`tests/test_opd.py::test_opd_ddp_all_reduces_gradients` |
| B47 | `data.sources` 返回 `MixedDataset` 并逐个校验子数据集 | `train/run.py:30`、`train/run.py:184` | `tests/test_trainer.py::test_mixed_sources_validate_each_child` |
| B48 | `tokens` 指标只统计主头（LM）token | `train/trainer.py:402`、`train/trainer.py:538`、`model/wrapper.py`（注释） | 更新 `test_gradient_accumulation`、`test_summary_tokens_count_all_steps` |
| B49 | `tokenize_prompts` 与 rollout 一致（tools + open_thinking） | `data/grpo.py:265` | `tests/test_data.py::test_tokenize_prompts_matches_rollout_format` |
| B50 | 去掉 judge URL 的重复查找 | `rl/judge.py:114` | 现有 `test_http_judge_requires_env` |
| B51 | verl `extra_info` 带 prompt，`compute_score` 还原 | `rl/verl.py:54`、`rl/verl.py:97` | `tests/test_integrations.py::test_verl_world_reward_uses_prompt` |
| B52 | Compressor 投影按参考实现强制 fp32，pooling 后转回 x.dtype | `model/compressor.py:80-111` | 现有模型/一致性测试 + `scripts/verify_reference.py` |
| B55 | AR 预取的 batch 不计入 resume 位置 | `train/trainer.py:183`（`_pull_batch`） | `tests/test_trainer.py::test_anticipatory_routing_trainer`（追加计数断言） |
| B53 | MTP 无效 token 计入路由统计 | 未修（见文末） | — |

## B40 — FSDP2 优化器状态恢复

- 确认：1 进程 FSDP2 resume 后 `train_step()` 报
  `aten.lerp_.Scalar got mixed torch.Tensor and DTensor`；2 进程时 Muon 先报
  `size of tensor a (2) must match ... (4)`；rollback 路径同样。修复前已用脚本
  复现（见 BUGS_02）。
- 改动：
  - `train/distributed.py` 新增 `_state_to_local`：把 full state 张量先按参数
    分片，再用 `DTensor.from_local(..., shape=value.shape, stride=value.stride())`
    包回参数同 layout；`_load_single_optimizer` 改用它。非 DTensor 参数路径不变。
  - `train/optimizer.py::Muon.step`：若 `momentum_buffer` 是 DTensor（恢复后可能
    是分片状态），先 `full_tensor()` 还原成全量并写回 state；正常训练时 buffer
    本来就是普通全量张量，该分支不触发。
- 验证：
  - 1 进程 FSDP2：resume → step OK；2 进程 FSDP2（hybrid Muon+AdamW）resume →
    两个 rank step OK；rollback → restore → step OK。
  - resume 等价性：同一 seed 下“连续 2 步”与“1 步 → 存盘 → resume → 再 1 步”的
    最终权重最大差 **6.1e-6**（float32 gather 顺序噪声），模型/优化器状态逐位一致。
  - `tests/test_distributed.py` 的 resume 测试现在会实际走一步；
    `scripts/test_distributed.py` 在 2 进程 fsdp2/ddp 下都会 resume 后走一步。
- 剩余风险：RL 的 resume 只加载模型权重，不受影响；旧 checkpoint 的优化器状态格式
  不变。

## B41 — balance loss 跳过 hash 层

- 确认：修复前 debug 模型 hash 层的 `f·p` 项直接贡献 loss，并给 hash 层 gate
  权重约 0.6e-4 的梯度（与 learned 层同量级）；hash 路由只查固定表，这项无法
  改变路由，只会改变专家加权。
- 改动：`sequence_balance_loss` 在 `layer.is_hash` 时 `continue`，并在 docstring
  说明；`layer_count`/归一化因此只覆盖 learned 层。
- 验证：`test_balance_loss_skips_hash_routers` 断言 hash 层梯度 < learned 层
  梯度的 1%（实测约 0.2%，来自经 learned router 输入回传的间接路径）；修复前
  该比值约 0.9，测试会失败。
- 剩余风险：间接路径（经网络输入回传的二阶项）无法完全消除；量级 ~1e-7，
  可忽略。

## B42 — batch-token warmup 跨重启保持

- 确认：同一 factory 连续创建两个迭代器，batch 序列完全相同（warmup 归零）；
  `full_6b` 在 8 卡下约 95% 处数据回卷，最后约 5% 训练会掉回 base batch。
- 改动：
  - `data/pretrain.py` 新增 `BatchTokenSchedule`（带 `index` 状态与 `next()`）；
    `iter_batches` 增加可选 `schedule` 参数，传入时由它推进预算，否则保持原
    行为（局部 `batch_index`）。
  - `data/sft.py::iter_batches` 同样支持共享 `schedule`；同时把 limit 计算从
    “每个样本算一次”改为“每个 batch 算一次”（数值等价，避免 schedule 多推进）。
  - `train/run.py::_batch_factory` 在训练 loader 上创建一个 schedule 并传给每次
    factory 调用；`load_dataset` 增加 `persistent_warmup` 参数。验证集仍传
    `persistent_warmup=False`，保持每次验证都用 base 预算、验证子集稳定。
- 验证：`test_batch_token_warmup_persists_across_loader_restarts` 覆盖 pretrain
  与 SFT 两种 loader；端到端 probe：epoch1 `[8,16,32,8]` → epoch2 `[32,32]`
  （修复前 epoch2 会从 8 重新开始）。
- 剩余风险：curriculum 的每个 stage 会新建 loader（新 schedule，索引从 0 开始），
  该 stage 的 warmup 会重新计时；这是 stage 边界行为，比例很小（0.01/0），
  未改变，已在 BUGS_02 记录。

## B43 — MetricsLogger.summary

- 确认：一条训练 + 一条 val 记录后 `summary()` 返回 `{'steps': 3}`。
- 改动：key 改为所有记录的并集；每个指标只用包含它的记录计算；`steps` 改为
  不同 step 的数量（不再把 val 记录算成训练步）。
- 验证：`test_factory_loader_cycles_and_validation_is_stable` 追加断言
  `steps == 4`、`loss_mean` 与 `val_loss_mean` 同时存在。
- 剩余风险：无。

## B44 — GRPO 顶层 string tools

- 确认：string tools 经 `list(...)` 变成 124 个字符，渲染出逐字符 `<tools>` 段。
- 改动：新增 `_record_tools`（string → `json.loads`，非 list 返回 `[]`），
  `convert_record` 与 `load_records` 都用它。
- 验证：`test_grpo_top_level_string_tools_are_parsed` 覆盖两条入口，并检查渲染
  后的 prompt 含工具名。
- 剩余风险：非法 JSON 静默返回空 tools（与 system message 解析策略一致）。

## B45 — consistency 门禁加载 MTP checkpoint

- 确认：README 的旗舰命令在 `checkpoints/smoke/final`（n_mtp=1）上报
  `Unexpected key(s) ... mtp_layers.0...`。
- 改动：`consistency/check.py::main` 在 `--n-mtp-layers 0` 时给
  `load_checkpoint` 传 `ignore_prefixes=("mtp_layers.",)`；显式传 >0 时保持 strict。
- 验证：`test_cli_ignores_mtp_heads_for_zero_mtp_model` 保存一个 n_mtp=1 的
  checkpoint，再跑 CLI（`--demo --max-new-tokens 1`），断言返回码 0；手工命令
  同样通过。
- 剩余风险：若用户显式传 `--n-mtp-layers 0` 但希望严格检查 MTP 键，现在的行为
  与 eval/RL 一致（忽略），符合 D38。

## B46 — RL 的 DDP 梯度归约

- 确认：2 进程 `mode=ddp` 跑 GRPO 一步后两个 rank 权重 digest 不同。
- 改动：GRPO/OPD 在 `loss.backward()` 之后、裁剪之前，若
  `dist.enabled and dist.mode == "ddp"` 则调用 `all_reduce_gradients`。
- 验证：`test_grpo_ddp_all_reduces_gradients` / `test_opd_ddp_all_reduces_gradients`
  用 spy 断言调用；2 进程 probe 现在两个 rank digest 一致（修复前不同）；
  `scripts/test_distributed.py --mode ddp` 仍通过。
- 剩余风险：ddp 模式下各 rank 局部 loss 等权平均（与 pretrain DDP 相同），
  非 token 加权；属于既有设计。

## B47 — 混料数据校验

- 确认：`_validate_dataset` 对 dict 取 `getattr(..., "meta")` 得到 None 后直接
  return。
- 改动：`train/run.py` 新增 `MixedDataset(meta, datasets)`，sources 分支返回它；
  `_validate_dataset` 先递归校验每个子数据集，再校验自身 meta；报告行仍用
  `getattr(dataset, "meta", {})`。
- 验证：`test_mixed_sources_validate_each_child` 断言错误 hash 的子数据集会抛错；
  手工 probe 确认 `load_dataset` 返回 `MixedDataset`、两份子数据、factory 正常。
- 剩余风险：无。

## B48 — tokens 指标口径

- 确认：`tokens` 计入 MTP 目标，depth 1 时约为 LM token 的 2 倍。
- 改动：Trainer 的 step/val 统计改用 `main_token_count`；`LossOutput.token_count`
  保留但补注释说明它是“所有预测槽位”。测试按新口径更新。
- 验证：`test_gradient_accumulation`（2×5=10）、
  `test_summary_tokens_count_all_steps`（4×3=12）通过。
- 剩余风险：历史日志/报告里的 tokens 是旧口径，不可直接与新结果比较。

## B49 — tokenize_prompts

- 改动：渲染时传 `tools=record.resolved_tools()`、`open_thinking=True`，与
  `rl/rollout.encode_prompt`、`eval._generate_responses` 一致。
- 验证：新增测试断言解码文本含工具名与 `<think>`、不含 `</think>`。
- 剩余风险：仓库内无调用方；若有外部调用，prompt 形态会与旧行为不同（这是修复
  目标）。

## B50 — judge URL 重复查找

- 改动：`make_judge` 只保留一次 `MINIDEEPSEEK_JUDGE_URL` 读取；行为与原实现
  等价（原来第二个是同一个变量，属于复制粘贴）。保留现有 requires-env 测试。
- 剩余风险：无（未新增 fallback 变量，避免未文档化的行为）。

## B51 — verl prompt 透传

- 改动：`to_verl_record` 在 `extra_info` 里带 `prompt`；`compute_score` 读取并
  还原 `GRPORecord.prompt`。
- 验证：`test_verl_world_reward_uses_prompt` 断言带 prompt 的 world reward 严格
  高于不带 prompt 的（relevance 从 0 变为非 0）；原 math 测试仍通过。
- 剩余风险：verl 本体未 GPU 实跑；`extra_info` 增加一个键对 verl 无副作用。

## B52 — Compressor fp32 投影

- 改动：`Compressor.forward` 用 `torch.autocast(device_type="cuda",
  dtype=torch.float32, enabled=x.device.type == "cuda")` 包住 `wkv`/`wgate`；pooling
  后 `kv.to(x.dtype)` 再 norm，与参考实现一致。CPU fp32 下是 no-op。
- 验证：CPU 全部测试通过；`scripts/verify_reference.py --seq-len 160 --mtp`
  worst block rel 1.5e-4、logits mean 1.6e-4（与修复前同一量级）。
- 剩余风险：CUDA bf16 上的实际数值收益未在 GPU 验证（本机无 CUDA）。

## B55 — AR 预取与 resume 位置

- 改动：Trainer 拆出 `_pull_batch()`（只取数据）与 `_next_batch()`（取数据并推进
  `_batches_consumed`）；`_prefetch_step` 改用 `_pull_batch`。
- 验证：`test_anticipatory_routing_trainer` 追加断言：跑 2 步后
  `_batches_consumed == 1`（只有真正用于训练的 batch 计入）；其它 resume 测试不变。
- 剩余风险：rollback 时队列被清空，已预取但未消费的 batch 会被丢弃；这是原有
  行为，只是现在不会被误记为“已消费”。

## 未修复项

### B53 — MTP 无效 token 计入路由统计

- 状态：保留（需设计决策）
- 原因：参考实现给 MTP block 传 `padding_mask=~valid`，让 router 计数与 aux loss
  排除跨界填充位置；本实现只把 `prev_hc_hidden` 置零。修复需要把 padding mask
  从 `MTPBlock` 传到 `MoE.forward` → router 计数 → balance loss 的 mask，改动面
  覆盖模型、路由记录与损失三处，收益是每篇文档末尾 1 个 token 的统计偏差。
  按“最小改动、不引入新风险”的原则本轮不做，已在 BUGS_02 标注为设计缺口。

## 验证记录（本机）

```text
.venv/bin/ruff check src tests scripts                 All checks passed
.venv/bin/python -m pytest -q                          165 passed (CPU debug model, ~29s)
bash scripts/debug_smoke.sh                            exit 0（smoke + consistency + 165 tests）
scripts/test_distributed.py --world-size 2 --mode fsdp2 --steps 1   PASS（含 resume step）
scripts/test_distributed.py --world-size 2 --mode ddp --steps 1     PASS（含 resume step）
scripts/test_rl_distributed.py --method grpo           PASS world=2 step=1
scripts/test_rl_distributed.py --method opd            PASS world=2 step=1
scripts/verify_reference.py --seq-len 160 --mtp        worst block rel 1.5e-4
debug SFT → GRPO → OPD 实跑                            三个 checkpoint 均落盘
```

手工复现脚本（/tmp，不入库）额外验证：

```text
1/2 进程 FSDP2 resume → step                  OK（修复前分别报 DTensor/形状错误）
FSDP2 rollback → restore → step               OK（修复前同上）
连续 2 步 vs 1 步+resume+1 步                  max |Δ| = 6.1e-6
2 进程 ddp GRPO 1 步                           两 rank digest 相同（修复前不同）
epoch1/epoch2 batch token                     [8,16,32,8] → [32,32]（修复前 epoch2 从 8 开始）
balance loss 的 hash 层梯度                     ~2e-7 vs learned ~1.4e-4（修复前 ~0.9:1）
metrics summary                               训练指标与 val_loss 同时保留
```

## 文档同步

- `README.md` / `doc/testing.md` 的测试计数 156 → 165。
- `doc/BUGS_02.md` 的对应条目状态 待修 → 已修。
- 本文件与 `doc/PROCESS.md` 新增一条进度记录。
