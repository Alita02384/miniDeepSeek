# PROCESS — 开发进度日志

> 每次**实质 commit** 后追加一条。格式与规则见 `AGENTS.md` §3.10。
>
> 只记"做了什么、实现了什么、怎么验证的"；不写流水账，不贴大段日志。
> 时间用本地日期（Asia/Hong_Kong）。

---

## 2026-11-04 项目初始化

- **commit**: `e2a4435` 初始化代码根仓库并清空 PROGRESS 占位文件
- **做了什么**: 确定 60M 路线与 6 专家 / top-2 / 4K-8K 上下文；建立 `AGENTS.md`（含 Git 提交与进度记录规则）与 `miniDeepSeek_technical_roadmap.md`；划定代码根 `miniDeepSeek/` 与文档目录 `doc/`；建立本日志。
- **实现了什么**: 暂无代码实现；工作约束、技术计划、进度记录机制就位。
- **验证**: 文档审阅（无代码可跑）。
- **未完成 / 风险**: `mini_60m` flavor、debug 模型跑通、数据管线。

## 2026-11-04 全链路代码实现（只用 V4 debug 模型验证）

- **commit**: `0296b3f` feat: 实现 miniDeepSeek 全链路代码并用 V4 debug 模型验证
- **做了什么**:
  - 模型：复制 torchtitan `deepseek_v4/flavors.py` 到 `src/minideepseek/model/flavors.py`，新增 `mini_60m/110m/200m` 与 dense 对照 flavor、`swiglu_limit`/tied embedding 覆盖；新增 `PackedBatch`、`MiniDeepSeekModel`、路由 record/replay、Anticipatory Routing、checkpoint（torch / safetensors / DCP）。
  - 数据：MiniMind tokenizer 封装与 chat response-only mask；pretrain packing、SFT 打包、GRPO/OPD schema、MinHash 去重、13-gram 污染检查、fertility、质量过滤、manifest 校验。
  - 训练：Muon+AdamW 混合优化器、warmup/cosine、MTP loss、sequence-wise balance loss、expert bias 更新、SwiGLU clamp 监控、单进程 trainer（预训练/SFT）。
  - 后训练：torch rollout、math/code/world/medical reward、GRPO、多教师 OPD full-vocabulary reverse KL。
  - 验收：logprob / routing / checkpoint 一致性门禁、math/code/world 评估、YAML 配置与脚本。
- **实现了什么**:
  - debug model（10.60M；MTP 12.98M）forward/backward、MTP、路由、checkpoint 全部跑通；
  - mini 配置参数：`mini_60m` 65.26M、`mini_110m` 120.65M、`mini_200m` 218.21M、`mini_60m_dense` 62.52M（mini 仅构建配置与统计，未训练）；
  - 一致性实测：rollout↔actor logprob max Δ ≈ 9.5e-7、routing replay 100%、checkpoint logits Δ=0；
  - 数据管线用 minimind 样本实际生成 pretrain/SFT/GRPO/OPD/eval 小数据集；
  - 预训练/SFT/GRPO/OPD/eval 的 debug 配置均实跑通过。
- **验证**:
  - `.venv/bin/ruff check src tests` → All checks passed
  - `.venv/bin/python -m pytest -q` → **84 passed**（CPU，V4 debug model，约 16s）
  - `bash scripts/debug_smoke.sh` → 训练 smoke + 一致性 + 测试全部通过
  - `bash scripts/prepare_debug_data.sh` → 生成 debug 数据并打印各阶段 meta
  - `python -m minideepseek.data.prepare manifest --manifest data/manifests/pretrain_6b.yaml` → 无 errors（外部数据路径 warning）
- **未完成 / 风险**:
  - 未真实训练 mini-60M / 未产出 6B token 数据；数据只用 MiniMind 小样本；
  - 未接 verl/vLLM，rollout 为 full-recompute torch，无 KV cache；
  - 未提供 HF transformers 架构类；safetensors 只是权重格式；
  - 多 GPU/FSDP/EP 未验证；world judge 为启发式占位；未接 lm-eval-harness；
  - Anticipatory Routing 为两段式近似、默认关闭；DSpark 未实现。

## 2026-11-04 稳定性与稀疏调度补齐

- **commit**: `8869aa9` feat: 补齐 RL 初始化、loss spike 处理与稀疏 index 调度
- **做了什么**:
  - GRPO/OPD 从 `train.resume` 加载 SFT specialist / base student checkpoint；校验 tokenizer hash。
  - 新增 `train/spike.py`：median/MAD loss spike 检测、内存 rollback 快照、spike 后自动开启 AR 的开关。
  - 新增 `IndexTopKSchedule`：4K dense-over-compressed 默认 `index_topk=seq_len//4`，最后 5% steps 收紧到 64；`set_index_topk` 可运行时切换。
  - code domain 预训练增加 SPDX license 过滤（copyleft/未知 header），并统计 `dropped_license`。
  - eval/consistency/RL 构建模型支持 `index_topk` 覆盖；更新配置与文档（D19–D22）。
- **实现了什么**:
  - spike 检测与自动 AR 可测；rollback 快照可恢复模型/优化器；
  - index top-k 调度在 trainer 中逐步生效（测试从 16 → 中间值）；
  - GRPO/OPD resume 后 logits 与来源 checkpoint 一致；
  - code license 过滤在预训练 meta 中计数。
- **验证**:
  - `ruff check src tests` → All checks passed
  - `pytest -q` → **93 passed**（CPU，V4 debug model）
  - `bash scripts/debug_smoke.sh` → 训练 smoke + 一致性 + 测试通过
  - `prepare_debug_data.sh` + debug pretrain/SFT/GRPO/OPD/eval/consistency 全部实跑通过
- **未完成 / 风险**:
  - rollback 是内存快照，不恢复 dataloader 位置；默认关闭；
  - index top-k 调度与序列长度课程尚未联动（当前只按 step 比例收紧，8K 数据/课程后置）；
  - 其余风险同 `0296b3f` 条目（未真实训练、未接 verl/vLLM、未做多卡）。

## 2026-11-04 smoke index_topk 一致性修复

- **commit**: `15574d1` fix: smoke 主模型与重载模型使用同一 index_topk
- **做了什么**: `train.smoke` 的 `ModelSettings` 现在传入 `--index-topk`；主模型与 checkpoint 重载模型使用同一 CSA top-k。
- **实现了什么**: 之前带 `--index-topk` 时重载模型会用 flavor 默认值，导致 checkpoint 对比误报；现在 `checkpoint_max_abs_diff=0.0`。
- **验证**: `ruff check src tests` 通过；`pytest -q` → 93 passed；`python -m minideepseek.train.smoke --flavor debugmodel --steps 1 --index-topk 4` → `checkpoint_verified: true`。
- **未完成 / 风险**: 无新增；该参数只影响 CSA index 选择，不改变模型参数。

## 2026-11-04 自包含模型重构（移除 torchtitan 运行时依赖）

- **commit**: `8390daa` refactor: 将 DeepSeek-V4 模型重构进项目，移除 torchtitan 运行时依赖
- **做了什么**:
  - 将 V4 模型数学（config / nn / rope / mHC / MoE / compressor / attention / decoder / MTP）
    重构进 `src/minideepseek/model/`，去掉 torchtitan 的 config/SPMD/sharding/remat 框架；
    state dict key 与形状对齐参考实现。
  - 删除 `src/minideepseek/_compat.py` 与复制的 `model/flavors.py`；`routing` / `losses`
    / `metrics` / `wrapper` 全部改为项目内实现。
  - 补齐 CSA indexer 蒸馏 loss（`SparseIndexerLoss`）；SwiGLU clamp 改为可监控的
    `LimitedSwiGLU`；routed/shared expert 的 w2 使用 depth-scaled 初始化。
  - 新增 `scripts/verify_reference.py` 逐层数值对齐脚本；新增参数校准测试。
- **实现了什么**:
  - 项目在 `../torchtitan` 不存在时也能 import/构建/训练 debug 模型；
  - debug 模型与参考实现逐层相对误差 ≤ 1.6e-4（fp32，差异来自上游 expert bf16 cast）；
  - 参数校准锁定：mini_60m 65,262,922 / mini_110m 120,652,773 / mini_200m 218,205,261 /
    mini_60m_dense 62,526,593 / debugmodel+MTP 12,984,104。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **94 passed**（CPU，约 19s）
  - `bash scripts/debug_smoke.sh` → 通过
  - `python scripts/verify_reference.py --seq-len 320 --mtp` → block rel ≤ 1.6e-4，
    logits mean abs ≤ 2.7e-4
- **未完成 / 风险**:
  - 多卡 FSDP2 / EP、HF 格式、vLLM / verl 接入尚未实现（下一步）；
  - 没有真实训练 mini-60M；数据仍只用 MiniMind 小样本。

## 2026-11-04 多卡 FSDP2 / HF / vLLM + verl 可选接入

- **commit**: `3f04493` feat: 多卡 FSDP2/DDP、HF 导出、vLLM/verl 可选接入
- **做了什么**:
  - 新增 `train/distributed.py`：`torchrun` 启动、FSDP2 `fully_shard`、
    manual DDP、FSDP2-aware global grad-norm clip、full state gather/load；
    `train/trainer.py` 与 `train/run.py` 接入，数据按 rank 分片。
  - Muon 支持 DTensor（FSDP2）：`full_tensor()` → NS-5 → 按原 placement 写回。
  - 新增 `model/hf/`：`MiniDeepSeekConfig`、`MiniDeepSeekForCausalLM`、
    training↔HF key 映射、`scripts/export_hf.py`；导出目录含 tokenizer 与
    remote-code 文件，支持 `trust_remote_code=True`。
  - 新增 `rl/backends/vllm.py`（可选、CUDA-only）与 `rl/verl.py` +
    `scripts/export_verl.py`（数据 schema / reward / Hydra config）。
  - 新增 `test_distributed.py`、`test_hf.py`、`test_integrations.py`；
    `scripts/test_distributed.py` 做多进程验证。
- **实现了什么**:
  - CPU/Gloo 2 进程：FSDP2 与 DDP 各自在 world=1 / world=2 下同 seed、同 batch
    的一步更新后 `model.pt` sha256 完全一致；
  - FSDP2 与 DDP 两种模式之间最大权重差 1.5e-8（float32 计算顺序）；
  - HF export → `from_pretrained` 的 logits 与训练模型逐位一致（max |Δ| = 0）；
  - vLLM/verl 代码路径存在且带明确守卫，未在 GPU 上实跑。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **106 passed**（CPU，约 25s）
  - `bash scripts/debug_smoke.sh` → 通过
  - `python scripts/test_distributed.py --world-size 2 --mode fsdp2|ddp` → PASS，
    两种模式 world=1/2 sha256 一致
  - `python scripts/export_verl.py ...` → 生成 JSONL/config/run.sh
- **未完成 / 风险**:
  - 无 CUDA 实机：NCCL、bf16、8 卡扩展、EP/TP/CP 未验证；
  - EP 未实现（expert 参数按 FSDP 分片，不是 all-to-all EP）；
  - vLLM 需要目标版本支持 auto_map；权重同步是重建 engine；
  - verl HF rollout + FSDP actor 未实跑；OPD 的 full-vocab reverse KL canonical
    实现仍是本项目 `rl/opd.py`，verl 模板用 forward_kl_topk。

## 2026-11-04 FSDP2 一致性修复与可选评测/裁判

- **commit**: `ad9647a` feat: FSDP2 一致性修复、HF tied 导出、lm-eval/HTTP judge 可选路径
- **做了什么**:
  - `clip_grad_norm` 改为先对各 DTensor shard 平方和 all-reduce，再统一计算
    global norm；修复 DP 尺寸改变 clip 系数导致多卡与单卡不一致。
  - FSDP2/DDP 包装后 rank0 broadcast 所有 buffer（hash 表/expert bias/RoPE cache）；
    indexer aux loss 除以 world size 抵消 FSDP 梯度平均。
  - HF 导出强制 untied（展开 `lm_head.weight`），修复 safetensors shared-storage
    报错；新增 tied embedding 导出测试。
  - 新增 `eval/lm_eval.py` + `scripts/run_lm_eval.py`（可选 lm-eval）与
    `rl/judge.py` HTTP world judge（可选），均带缺依赖/缺 env 守卫。
  - `loss_spike_rollback` 在 distributed 下显式拒绝（内存快照是单进程实现）。
- **实现了什么**:
  - CPU/Gloo world=1 与 world=2：FSDP2/DDP 一步更新后的 model.pt sha256
    完全一致（seed/batch 相同）；
  - HF tied 模型 export → load logits 仍逐位一致；
  - 新增 3 个集成测试；pytest 110 passed。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **110 passed**
  - `bash scripts/debug_smoke.sh` → 通过
  - `python scripts/test_distributed.py --world-size 1|2 --mode fsdp2|ddp` → PASS
- **未完成 / 风险**:
  - 无 CUDA 实机；EP/TP/CP、bf16、NCCL 未验证；
  - lm-eval 与 HTTP judge 未真实运行；
  - vLLM 自定义模型支持与 verl GPU smoke 待做。

## 2026-11-04 独立性门禁与训练细节修复

- **commit**: `3b4c25d` test: 增加独立性门禁与 HF 1D mask 兼容
- **commit**: `3b791c2` fix: NCCL 兼容的 grad-norm dtype；mHC 参数走 AdamW；记录 indexer loss
- **做了什么**:
  - 新增 `tests/test_independence.py`：源码禁止 import torchtitan/minimind 等参考
    checkout、禁止 sys.path 注入，并在无 workspace checkout 的子进程中构建模型；
  - HF forward 兼容 1D `attention_mask`/`position_ids`；
  - `clip_grad_norm` 在 CUDA/NCCL 上用 float32 累加（NCCL 不支持 float64）；
  - `hc_*` 参数归入 AdamW（mHC 静态参数），其余矩阵走 Muon；
  - `StepMetrics` 记录 `indexer_loss`。
- **实现了什么**:
  - 独立性从“约定”变成 pytest 门禁；
  - FSDP2 2 进程仍与 world=1 sha256 一致；
  - mini-60M tied embedding 在 FSDP2 下 forward/backward/optimizer 正常。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **113 passed**
  - `bash scripts/debug_smoke.sh` → 通过
  - `OMP_NUM_THREADS=1 python scripts/test_distributed.py --world-size 2 --mode fsdp2` → PASS
- **未完成 / 风险**:
  - EP 仍未实现；vLLM/verl 仍无 GPU 实跑；
  - 6B 数据与真实训练不在当前阶段。

## 2026-11-04 收尾：评测/裁判接线与路径清理

- **commit**: `ad0f3fd` fix: eval 接 HTTP world judge，容忍 fenced JSON 输出
- **commit**: `2e5a9f4` chore: HF 导出使用 tokenizer 路径常量
- **做了什么**:
  - `eval/evaluate.py` 在 `reward.world_judge=http` 时创建 HTTP judge；
    judge 响应支持 ```json fenced 输出；
  - HF 导出从 `paths.TOKENIZER_DIR` 取 tokenizer（支持 env 重定位）；
  - scheduler 文档去掉“单进程”限定。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **113 passed**
  - `bash scripts/debug_smoke.sh` → 通过
- **未完成 / 风险**:
  - HTTP judge / lm-eval / vLLM / verl 均未在 CI 真实调用；
  - EP、真实 6B 数据、真实 60M 训练仍不在当前阶段。

## 2026-11-04 batch warmup 与验证集接线

- **commit**: `ffaa4b4` feat: batch-token 线性 warmup 与 val_path 接线
- **做了什么**:
  - `DataSettings` 增加 `batch_tokens_final` / `batch_warmup_ratio`；
  - `data.pretrain.iter_batches` 与 `data.sft.iter_batches` 通过
    `token_schedule` 按 micro-batch 线性升温；
  - `train/run.py` 读取 `train.val_path`，构建独立验证 loader 传给 trainer；
  - full_6b/pilot/dense_control 从 16K 升到 64K tokens/micro-batch。
- **实现了什么**:
  - roadmap 阶段 2 的“前 10% 步线性升 batch”有代码与单测；
  - 验证 loss 现在可以按 `val_every` 真正跑起来。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **114 passed**
  - `bash scripts/debug_smoke.sh` → exit 0
- **未完成 / 风险**:
  - sequence-length curriculum 仍是数据分阶段 recipe，没有单次训练自动调度；
  - weighted data mixer / paragraph-level dedup / 更细 quality filter 仍未实现。

## 2026-11-04 对照官方 V4 报告修正优化器与 LR

- **commit**: `7fbd0c1` fix: 按 V4 报告对齐 Muon/AdamW/LR schedule
- **做了什么**:
  - Muon 改为官方 Algorithm 1：8 步快速 NS + 2 步稳定 NS，
    更新缩放 `sqrt(max(n,m)) * 0.18`；
  - AdamW eps 改为官方 1e-20；
  - mHC 参数分组修正：`hc_base`/`hc_scale` → AdamW，`hc_fn` → Muon；
  - LR schedule 增加 `decay_start_ratio`，实现 warmup→保持→末期 cosine；
  - `doc/roadmap_audit.md` 补官方报告逐条对照。
- **实现了什么**:
  - Muon/AdamW/学习率三个此前与官方报告不一致的点已修正；
  - 新增 flat→cosine 单测；pytest 115 passed。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **115 passed**
  - `bash scripts/debug_smoke.sh` → exit 0
  - `scripts/test_distributed.py --world-size 1|2 --mode fsdp2` → sha256 一致
- **未完成 / 风险**:
  - 官方报告的 EP/CP/tensor-level activation checkpointing、FP4 QAT、
    batch-invariant kernels、GRM、teacher hidden-state cache、preemptible
    rollout 等仍未实现；见审计文档的分类。

## 2026-11-04 补齐 AR/rollback/长度课程/数据与门禁

- **commit**: `53eb7f6`, `bfdf6f9` feat: 补齐 AR 预取、分布式 rollback、长度课程与数据/指标/同步门禁
- **做了什么**:
  - AR：trainer 预取下一 optimizer step 的 batch，用当前 gate 计算并缓存
    routing；下一步 replay；queue 空回退 stale-gate probe；probe 包含 MTP router；
  - rollback：FSDP2 下全 rank gather 快照、rank0 broadcast spike 决策；显式
    DTensor shard 拷贝修复 resume/rollback（含空 shard 边界）；
  - sequence curriculum：`train/curriculum.py`、`scripts/train_curriculum.sh`、
    `data.prepare curriculum`、2K/4K/8K 配置；全局 `schedule_steps` 保持
    LR/MTP/index 调度连续；
  - data：`WeightedBatchMixer`（data.sources）、`ParagraphDeduper`、
    语言/script、vendored、binary、math formula/answer 过滤；
  - metrics：`domain_routing_metrics` 并接入 GRPO 可选日志；
  - vLLM：weight version + 可选 actor/rollout logprob 门禁。
- **实现了什么**:
  - 上述缺口从审计中的 ❌/⚠️ 变为有代码 + CPU 测试；
  - FSDP2/DDP world=1 与 world=2 一步更新 sha256 一致；
  - 2 进程 resume/rollback 通过；
  - pytest 125 passed。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **125 passed**
  - `bash scripts/debug_smoke.sh` → exit 0
  - `scripts/test_distributed.py --world-size 1|2 --mode fsdp2|ddp` → PASS
- **未完成 / 风险**:
  - AR probe 仍是同步执行，未做官方 EP 通信 overlap；
  - EP/CP/激活重计算按用户要求不做；GRM/world 专家按用户要求暂缓；
  - FIM/token-splitting、长文档/多语言语料、teacher hidden-state cache、
    vLLM/verl GPU 实跑仍未完成。

## 2026-11-04 收尾修复：curriculum 全局调度 / AR MTP / FSDP state

- **commit**: `0acc508` fix: 补齐 curriculum 全局调度、AR MTP 路由、FSDP optimizer state 与 rollback 全 rank 快照
- **做了什么**:
  - curriculum 每个 stage 共用最终 `schedule_steps`，LR/MTP/index top-k 调度连续；
  - AR probe 包含 MTP router（否则 replay 缺 MTP 层），预取按 optimizer step 粒度；
  - FSDP `load_model_state_dict` 处理空 shard/uneven shard；
  - FSDP optimizer state 按参数名映射并 shard 到 local DTensor，resume 真正恢复；
  - rollback 快照在所有 rank gather，非 rank0 restore 不再缺 key。
- **验证**:
  - `ruff check src tests scripts` → All checks passed
  - `pytest -q` → **125 passed**
  - `bash scripts/debug_smoke.sh` → exit 0
  - `scripts/test_distributed.py --world-size 1|2 --mode fsdp2` → sha256 一致
  - 2 进程 resume/rollback 临时脚本 → PASS
- **未完成 / 风险**:
  - AR probe 同步执行、无 EP overlap；
  - EP/CP/激活重计算、GRM/world 专家按用户要求不做；
  - vLLM/verl GPU、FIM/长文档、teacher cache 仍未做。

## 2026-11-04 bf16 统一与 FSDP optimizer state

- **commit**: `d363fef` feat: rollout/eval/teacher 统一 bf16 autocast；修复 FSDP optimizer state 恢复
- **做了什么**:
  - `MiniDeepSeekModel` 增加 `autocast_dtype`，CUDA `forward` 统一 bf16；
    build_model/build_trainer/GRPO/OPD/eval/smoke 传递 dtype，teacher 增加 dtype；
  - FSDP `load_optimizer_state_dict` 按参数名映射并 shard 到 local DTensor；
  - `doc/roadmap_audit` 更新 dtype/autocast 与已补齐项。
- **验证**:
  - ruff 通过；pytest **125 passed**；debug_smoke exit 0；
  - FSDP2 world=1/2 sha256 一致；
  - 2 进程 resume/rollback 临时脚本通过。
- **未完成 / 风险**:
  - teacher 权重仍 fp32（只 autocast），未做 hidden-state cache/bf16 offload；
  - AR probe 同步；EP/CP/激活重计算、GRM/world 专家按用户要求不做；
  - vLLM/verl GPU、FIM/长文档仍未做。

## 2026-11-04 GRPO/OPD FSDP2 多卡

- **commit**: `bcf4c2b` feat: GRPO/OPD 支持 FSDP2 多卡路径
- **做了什么**:
  - `RLConfig.distributed`；builder 初始化/包装 FSDP2，record/prompt 按 rank 分片；
  - GRPO/OPD 使用 FSDP-aware clip、gather checkpoint、手工 model/optimizer resume；
  - vLLM backend 与 FSDP2 RL 显式拒绝；
  - 新增 `scripts/test_rl_distributed.py` 与 `tests/test_rl_distributed.py`；
  - train_grpo.sh/train_opd.sh 支持 torchrun。
- **验证**:
  - ruff 通过；pytest **127 passed**；debug_smoke exit 0；
  - `scripts/test_rl_distributed.py --method grpo|opd`（CPU 2 进程）→ PASS，
    各跑 1 step 并保存。
- **未完成 / 风险**:
  - vLLM/verl GPU 路径未验证；teacher 每 rank 复制；局部 loss 等权平均；
  - teacher hidden-state cache、FIM/长文档仍未做。

## 2026-11-05 修复 hash 层 bias 与哈希表重建

- **commit**: `7d9e5ce` fix: hash 层不再保留负载均衡 bias，哈希表改为确定性生成
- **做了什么**:
  - hash 层不再注册 `expert_bias_E`；`update_expert_biases` 只更新 learned
    层，所有层计数器照常清零；
  - `tid2eid` 改为由 `(HASH_SEED, layer_id)` + 局部 Generator 确定性生成；
    `init_weights()` 不再重建表；
  - 新增/调整 4 个测试；`doc/decisions.md` 记 D34。
- **实现了什么**: hash 层不再有会漂移的死 bias；同一配置任意进程重建出
  位级相同的哈希表；load 后调 `init_weights()` 不再改变路由。
- **验证**: pytest **130 passed**；`ruff check src tests` 通过；
  `python -m minideepseek.train.smoke --flavor debugmodel --steps 2` →
  `checkpoint_verified=true, checkpoint_max_abs_diff=0.0`。
- **未完成 / 风险**: 旧 checkpoint 的 hash 层 bias 键会让 strict 加载报错，
  需重新生成；多卡 `broadcast_buffers` 仍是必要保险（确定性未在 CUDA 上验）。

## 2026-11-05 审计确认问题的分批修复

- **commit**: `837a3da` fix: 修复审计确认的训练/数据/RL 逻辑问题
- **做了什么**:
  - 数据/训练：`load_dataset` 改为可重建 batch factory；验证集每次重置；`batch_tokens`/`prompts_per_step` 按 world_size 全局平分；resume 的 data/optimizer/step 三开关；dataset tokenizer/seq_len 校验；flat dataset 改 `.npy` sidecar 真 mmap（旧 npz 回退）。
  - 配置链：SFT 从 pretrain base 初始化；GRPO/OPD 加载时忽略 MTP 头且只加载模型权重；curriculum 8K top-k=2048；运行期 top-k 写入 checkpoint meta，eval/export/teacher 优先读取。
  - 模型/RL：context 上限与生成截断；indexer aux loss 去掉额外 `/world`；MiniMind tool 数据解析与透传；AR snapshot 时序；Muon/AdamW 分组；tied safetensors；数学答案解析/format bonus；采样 logprob；GRPO vLLM 同步；verl 的 rollout n/val/reward import。
  - 工具链/配置：consistency 传递运行时配置、HF export meta 回退、reference parity 过滤旧 hash-bias key、grpo manifest 预算、SFT/GRPO/OPD/eval config。
  - 文档：新增 `doc/BUGS.md`（问题与证据）与 `doc/FIXES.md`（修复内容、验证、具体位置）；`doc/decisions.md` 增加 D35–D45。
- **实现了什么**:
  - 训练/验证 loader 不再在 generator 耗尽时崩溃；val loss 不再 NaN；多卡 global batch 与配置注释一致；从 base 初始化 SFT/RL 不再继承旧 step/LR。
  - 155 个 CPU 测试通过（新增 25 个回归测试）；debug smoke、debug SFT→GRPO→OPD、FSDP2 2-proc、reference parity 均实跑通过。
- **验证**:
  - `.venv/bin/ruff check src tests scripts` → All checks passed
  - `.venv/bin/python -m pytest -q` → **155 passed**
  - `bash scripts/debug_smoke.sh` → smoke + consistency + 155 passed
  - `.venv/bin/python scripts/test_distributed.py --world-size 2 --mode fsdp2 --steps 1` → PASS
  - `scripts/test_rl_distributed.py --method grpo/opd` → PASS
  - `scripts/verify_reference.py --seq-len 160 --mtp` → worst block 1.5e-4
  - debug SFT→GRPO→OPD→eval 实跑通过
- **未完成 / 风险**:
  - 未提交/未验证的剩余项见 `doc/BUGS.md` P2：GRPO/OPD expert bias 更新、部分配置空转、污染/manifest 未接训练入口、HF remote-code 非自包含、grad accumulation token 等权等；
  - vLLM/verl GPU 实机、bf16/NCCL、真实 mini-60M 数据与训练仍未验证；
  - `.npy` sidecar 改变了新建数据集的物理格式（逻辑路径仍为 `.npz`）；旧 `.npz` 仍可读。

## 2026-11-05 审计修复收尾（RL bias / 配置 / 泄漏检测）

- **commit**: `7e93371` fix: RL 阶段更新 expert bias 并让 broadcast_buffers 配置生效；`3b49339` fix: 收紧 GRPO reference 泄漏检测
- **做了什么**:
  - GRPO/OPD 在 optimizer.step 前 all-reduce usage counts 并执行 loss-free expert-bias 更新，计数器清零；
  - `DistributedSettings.broadcast_buffers=false` 现在真正跳过 buffer 广播；
  - GRPO reference 泄漏检查改为 answer marker/长串判断，避免 `1+2` 这类问题数字误杀；
  - 更新 BUGS/FIXES/README/testing。
- **验证**:
  - `pytest -q` → **156 passed**
  - `bash scripts/debug_smoke.sh` → 通过
  - FSDP2 2-proc → PASS
  - RL GRPO/OPD 2-proc → PASS
  - `verify_reference.py` → worst block 1.5e-4
- **剩余 / 设计决策**:
  - B31（code license 未知 header 策略）、B32（污染检查/manifest 强制接入）、B35（HF remote-code 自包含）、B37（grad accumulation token 加权）仍需产品/数据策略决定，不是当前可复现的静默逻辑错误；
  - vLLM/verl/NCCL/bf16 实机与真实 60M 训练仍未验证。

## 2026-11-06 审计 02 与修复（FSDP2 resume / balance / warmup / metrics / 数据 / RL）

- **commit**: `6da5bb0` fix: 修复审计 02 的训练/数据/RL 逻辑问题
- **做了什么**:
  - 第二轮独立审计（通读源码 + 参考实现对照 + /tmp 数值脚本），新增
    `doc/BUGS_02.md`（1 个 P0、5 个 P1、6 个 P2、若干设计项）与
    `doc/FIXS_02.md`（逐项修复记录/验证/剩余风险）；
  - FSDP2 optimizer-state 恢复：state 按参数 layout 包回 DTensor；Muon 收到分片
    momentum buffer 时先 `full_tensor()` 还原；
  - sequence balance loss 跳过 hash 路由层；batch-token warmup 改为共享
    `BatchTokenSchedule`，跨 epoch/resume 继续（val 仍用 base 预算）；
  - `MetricsLogger.summary` 取 key 并集；`tokens` 指标只统计主头 LM token；
  - GRPO 顶层 string tools 解析；consistency CLI 在 n_mtp=0 时忽略 `mtp_layers.`；
  - GRPO/OPD 在 ddp 模式下 all-reduce 梯度；`data.sources` 返回 `MixedDataset`
    并逐源校验；`tokenize_prompts` 对齐 rollout prompt；
  - verl `extra_info` 透传 prompt；Compressor 投影按参考实现 fp32；
    AR 预取不再计入 `extra.data_batches`。
- **实现了什么**:
  - 多卡（FSDP2）中断续训/curriculum 交接/rollback 恢复后可继续 step；
  - 多 epoch 训练不再把 batch-token warmup 重置；
  - 带验证集的 run 报告保留 lr/grad_norm/routing/swiglu 汇总；
  - ddp RL 的 rank 权重保持一致；混料训练恢复 tokenizer/seq_len 门禁。
- **验证**:
  - `.venv/bin/ruff check src tests scripts` → All checks passed
  - `.venv/bin/python -m pytest -q` → **165 passed**
  - `bash scripts/debug_smoke.sh` → exit 0
  - `scripts/test_distributed.py --world-size 2 --mode fsdp2|ddp --steps 1` →
    PASS，且脚本现在会在 2 进程下 resume 后继续走一步
  - `scripts/test_rl_distributed.py --method grpo|opd` → PASS
  - `scripts/verify_reference.py --seq-len 160 --mtp` → worst block 1.5e-4
  - debug SFT → GRPO → OPD 实跑，三个 checkpoint 落盘
  - 手工：1/2 进程 FSDP2 resume/rollback step、resume 等价性（max |Δ| 6.1e-6）、
    ddp GRPO 两 rank digest 一致、epoch1/2 batch token `[8,16,32,8]→[32,32]`
- **未完成 / 风险**:
  - B53（MTP 无效 token 计入路由统计）保留为设计缺口，需 padding-mask 贯通；
  - Compressor fp32 对齐的 CUDA bf16 数值收益未在 GPU 验证；
  - vLLM/verl GPU、NCCL、真实 mini-60M 数据与训练仍未验证。

## 2026-11-06 三个模型档位与 configs/model 架构文件

- **commit**: `363e3af` feat: 三个模型档位改用 configs/model 架构文件加载
- **做了什么**:
  - 采用 `doc/Parameter_Setting.md` 的设计，新增 `mini_220m_a80m`
    （默认）、`mini_145m_a60m`、`mini_60m_dense_d768` 三个完整
    `ModelSettings` YAML；删除旧 `mini_60m/110m/200m`；
  - `model/config.py` 改为从 YAML 构建：只保留校验、hybrid 层映射、MTP
    派生与 index top-k 派生；训练配置用 `model_file` 选择架构，`model:`
    覆盖运行参数，CLI 增加 `--model-config`；
  - `seq_len`（4K）与 `max_context`（8K）分离；HF 导出/加载携带
    `max_context`；
  - dense 档单 expert 权重固定为 1.0，`sequence_balance_loss` 对单 expert
    返回 0；修复 `parameter_report` 的 MTP expert 重复统计；
  - 更新所有 pretrain/SFT/GRPO/OPD/eval 配置、README、decisions
    D46-D52、testing 与 sources。
- **实现了什么**:
  - 三个档位参数与激活（MTP depth 1、tied embeddings）：
    `mini_220m_a80m` 216,652,538 / 82.17M；
    `mini_145m_a60m` 145,841,946 / 60.91M；
    `mini_60m_dense_d768` 60,876,570 / 60.88M；
  - 新模型只需新增/替换 `configs/model/*.yaml`，无需改 Python；
  - `mini_220m_a80m` 在 CPU 完成 1-step forward/backward + checkpoint
    回读，三档均进入单测。
- **验证**:
  - `.venv/bin/ruff check src tests scripts` → All checks passed
  - `.venv/bin/python -m pytest -q` → **175 passed**
  - `bash scripts/debug_smoke.sh` → exit 0（175 passed）
  - `scripts/verify_reference.py --seq-len 64 --mtp` → worst block 1.631e-4，
    logits mean abs 1.58e-4
  - `python -m minideepseek.train.smoke --flavor mini_220m_a80m --seq-len 64
    --n-mtp-layers 1 --steps 1 --device cpu` → `checkpoint_max_abs_diff=0.0`
  - `load_config(TrainConfig, configs/pretrain/full_6b.yaml)` → 构建
    216.65M / max_context 8192，跑 1 step 通过
- **未完成 / 风险**:
  - head_dim=256 的 torch gather attention 在 4K/8K 真实 batch 下的显存
    未验证（decisions D51）；full_6b 的 64K tokens/micro-batch 大概率需要
    fused kernel / activation checkpointing / 分块重算；
  - 旧的“dense attention 对照”（compress_ratios 全 0）未保留；
  - vLLM/verl GPU、NCCL、真实数据训练仍未验证。

## 2026-11-06 训练监控、可视化与实验基建（P0+P1）

- **commit**: `9f44ce0` feat: 增加训练监控、可视化与实验基建；`9a019d6` docs: 增加训练执行计划与监控/基准工具说明
- **后续小修**: `6a534d2` 测试计数；`2257361` plot_training 单 run；`de8d449` full_6b/pilot 启用 keep_last 与 report_every
- **做了什么**:
  - 扩展 `StepMetrics`/`metrics.jsonl`：`tokens_seen`、`tokens_per_sec`、
    `step_time_s`、`eta_seconds`、`loss_ema`、`mem_allocated/reserved/peak_mb`、
    `mtp_loss_d{depth}`、`val_ppl`/`val_main_loss`/`val_mtp_loss` 等；
    控制台输出从 JSON 改为人类可读单行。
  - 新增 `src/minideepseek/viz/`：无依赖 SVG 折线图 + 静态 HTML 报告
    （loss/EMA/val、学习曲线、ppl、lr、grad norm、吞吐、显存、MoE 负载、
    路由健康、SwiGLU clamp、辅助 loss），`train.report_every` 训练中自动刷新。
  - NaN/Inf：写 `nan_step_*/`（模型、优化器、`nan_report.json`）后退出；
    val 最优写 `best/`；`keep_last_ckpt` 清理旧 `step_*`。
  - 新增 `scripts/benchmark.py`（单卡 / torchrun FSDP2，warmup + CUDA event +
    峰值显存）、`scripts/scaling_law.py`（幂律拟合 α/R² + Chinchilla 线）、
    `scripts/plot_training.py`。
  - 模型/优化器 docstring 补 V4 论文章节号（§2.1/2.2/2.3/2.4/4.2.3）。
- **实现了什么**:
  - `pytest -q`：**197 passed**（新增 22 个：metrics 5、viz 7、scaling 5、
    benchmark 1、trainer 4）。
  - debug 模型 4 步训练产出 `metrics.jsonl` + `report.html`（11 个面板），
    打印 tok/s、峰值显存（CPU 无）、ETA；`best/` 与 keep-last 清理生效。
  - 单进程 benchmark：debugmodel/seq16/batch2 → 246 ms/step、122 tok/s；
    2 进程 CPU torchrun + FSDP2 路径跑通（1 step，world_size=2）。
  - NaN 注入测试：`nan_step_000000/nan_report.json` 落盘后按预期抛
    `NonFiniteLossError`。
- **验证**:
  - `.venv/bin/ruff check src tests scripts` → All checks passed
  - `.venv/bin/python -m pytest -q` → 197 passed, 2 warnings（~40s）
  - `.venv/bin/torchrun --nproc_per_node=2 scripts/benchmark.py ... --distributed fsdp2`
    → exit 0（CPU Gloo，1 step）
  - `python scripts/plot_training.py` / `scaling_law.py` 生成 HTML 并人工检查渲染
- **未完成 / 风险**:
  - 多卡 CUDA（NCCL）benchmark 与 `report_every` 长时间训练未实测；
  - 分布式 NaN 快照需要所有 rank 同时进入 `save()`，单 rank 失败时仍可能
    等待集合通信（与现有 DDP/FSDP 失败语义一致，未额外处理）；
  - 可视化刻意不引入 matplotlib，复杂交互（缩放/悬停数值）暂不支持。

## 2026-11-07 UTF-8 文本 I/O 修复

- **commit**: `3cd3ebd` fix: 文本读写显式使用 UTF-8，避免依赖进程 locale
- **做了什么**: 22 处 `read_text`/`write_text` 补 `encoding="utf-8"`
  （checkpoint/dataset meta、config YAML、eval 报告、HF 导出、verl 数据、
  `nan_report.json` 等）；新增 `tests/test_encoding.py`，在 C/ASCII locale 下
  验证 checkpoint、dataset、config、eval 报告的中文 round-trip。
- **实现了什么**: 不再依赖 `LANG`/Windows 默认编码；在 `LC_CTYPE=C` 且关闭
  UTF-8 mode 时，含中文的 meta/report/config 仍可读写。
- **验证**:
  - `.venv/bin/ruff check src tests scripts` → All checks passed
  - `.venv/bin/python -m pytest -q` → 204 passed（工作树；其中 3 个为并行
    未提交的 activation checkpointing 测试，本 commit 自身为 201）
- **未完成 / 风险**: 工作树同时存在 activation checkpointing / 数据打包优化
  的未提交改动，不属于本 commit；两者已在同一工作树上通过全量测试。

## 2026-11-07 24GB 卡上的 145M 2K→4K→8K 路径

- **commit**: `a8db9e2` feat: 24GB 卡上的 145M 2K→4K→8K 训练路径
- **做了什么**:
  - `DeepSeekV4Model.grad_checkpointing` + `_run_block()`：按 block
    `torch.utils.checkpoint(use_reentrant=False)`；`train.activation_checkpointing`
    与 `CurriculumStage.activation_checkpointing` 两个开关；
  - `pack_documents` 用 `array("i")`/`bytearray` + `np.frombuffer`，逐字节等价；
  - `run_curriculum` 新增 `latest_stage_checkpoint()`：阶段内续跑、重复执行幂等；
  - 新增 `configs/pretrain/pretrain_145m.yaml`、
    `configs/pretrain/curriculum_145m_4x4090.yaml`、
    `configs/pretrain/curriculum_debug_smoke.yaml`；
  - 语料复制到 `data/raw/`（2.3B token，8.3 GB），新增
    `data/manifests/pretrain_local.yaml`，6B manifest 的 minimind 路径改到 `data/raw/`；
  - `scripts/measure_activation_budget.py` 增加 `--activation-checkpointing`；
  - 新增 `doc/server_runbook.md`（上传、环境、数据、上卡检查、训练、续跑、降级）。
- **实现了什么**:
  - 145M@4K saved 激活 **12.27 GiB → 0.53 GiB**（实测，CPU bf16 autocast）；
    8K 不开重算约 25 GiB（24 GB 必 OOM），开重算预估 12–15 GiB；
  - debug 模型 2K→4K→8K 三阶段课程在 CPU 上 21 s 跑完，产出三个 `final/`；
  - 3000 文档在 2K/4K/8K 三种打包长度下都产出可用数据集（33 s）。
- **验证**:
  - `.venv/bin/python -m pytest -q` → 204 passed（含 3 个新的 checkpointing 测试）；
  - `.venv/bin/ruff check src tests scripts` → All checks passed；
  - `scripts/train_curriculum.py --config configs/pretrain/curriculum_debug_smoke.yaml`
    → 3 阶段 until_step 1/2/3 全部完成；
  - `measure_activation_budget.py --flavor mini_145m_a60m --seq-len 4096`
    → ac=0: 12.27 GiB, ac=1: 0.53 GiB；
  - `manifest --manifest data/manifests/pretrain_local.yaml` → ok: true。
- **未完成 / 风险**:
  - 8K + 重算的 **真实 CUDA 峰值未实测**（本机无 GPU），runbook §3 给了上卡
    先跑 `measure_activation_budget.py --device cuda` 的顺序；
  - 数据准备仍是“每个长度一次全量扫描”，三阶段需并行跑（约 4.5 h/阶段）；
    单遍多长度打包未实现；
  - 220M 档在 24 GB 上 4K 需重算、8K 需 chunked attention（未实现）。

## 2026-11-07

- **commit**: `0f8b9bb` docs: 新增 145M 数据方案（预训练/SFT/GRPO 数据源与筛选口径）
- **做了什么**: 新增 `doc/data_plan_145m.md`：按 145M/61M 激活、6400 词表、4K 上下文收敛
  路线图 §4 的数据方案；给出预训练 1.2B 六源混料、数学/代码 SFT 各 12M token 的来源与
  过滤规则、数学/代码 GRPO 的 prompt 池与难度带做法，以及与 `data/manifests`、
  `prepare` CLI、`rl/reward.py` 的对接格式。
- **实现了什么**:
  - 用仓库自带 tokenizer 实测 fertility：中文 0.233、英文 0.330、Python 0.38–0.44、
    JS 0.471、C++ 0.546 tok/byte；4 空格缩进 5 token/行，2 空格 4 token/行；
  - 逐个核对候选数据集的规模/许可/字段（HF API + dataset card），确认
    Orca-Math 200k(MIT)、GSM8K_zh 8,792(MIT)、Magicoder-OSS 75k(MIT)、
    self-oss-instruct 50k(ODC-By)、MBPP 374+120(CC-BY-4.0)、
    open-r1 verifiable 15,068（含 fn_name/input/output 测试）、TACO 25,443(Apache-2.0)。
- **验证**:
  - `score_text()` 对仓库 Python 源码与 GSM8K 样本均 `accepted=True`，现有质量过滤不会误杀；
  - 文档内每条规模/许可均来自本次 API 查询，未使用记忆值。
- **未完成 / 风险**:
  - 只出方案，**未写下载/过滤/转换脚本**，也未新增 manifest；
  - `rl/reward.py::code_reward` 目前是全通过=1、否则=0 的二值奖励，145M 上大概率恒 0，
    方案要求改成部分分（未实施）；
  - 代码 reward 直接 `subprocess` 跑模型生成代码、无沙箱，长期使用需加固。

## 2026-11-07（补充：GRPO 数据量与词表评估）

- **commit**: 见本条 commit
- **做了什么**: `doc/data_plan_145m.md` 新增 §1.4（GRPO 数据量口径）与 §1.5（词表要不要换）：
  按 200/500/1000/2000 步给出 in-band prompt 数与原始池需求；对比 5 个公开 tokenizer 的
  tok/byte 与 embedding/LM-head 算力，结论是不换公开词表。
- **实现了什么**:
  - 实测 5 个 tokenizer 在中文/英文/Python/GSM CoT 上的 tok/byte：
    minimind-6400 = 0.245/0.283/0.426/0.444；SmolLM-49k = 0.549/0.183/0.290/0.317；
    Qwen2.5-151k = 0.207/0.179/0.241/0.309；TinyLlama-32k = 0.410/0.193/0.339/0.335；
    GPT-2-50k = 0.709/0.194/0.455/0.259；
  - 把 LM head 算力（2×768×vocab）折进"每字节有效算力"后：英文/Python 上 49k 与 6.4k 基本打平
    （36.1 vs 37.3、57.3 vs 56.2），中文上 6.4k 便宜 3.4×；151k 词表 embedding 116.5M 不可行。
- **验证**:
  - tokenizer 数据由 `tokenizers` 库直接加载各家 `tokenizer.json` 实测，非估算；
  - `src/minideepseek/rl/opd.py` 文件头确认使用 full-vocabulary reverse KL，且注释写明依赖 6400 词表。
- **未完成 / 风险**:
  - 未实际训练 16k 自定义词表，文中 0.21/0.24/0.36 是按 6.4k→49k 插值的估计值；
  - 词表若不换，代码/英文的 token 成本要长期用"短样本 + 只做 Python"来对冲。

## 2026-11-07 general/tool specialist + flat data + one-click pipeline

- **commit**: `30baa59` feat: general/tool specialists, flat data, multi-turn tool RL
- **做了什么**：
  - 新增 `data/build.py`：从 `minimind/dataset` 生成平铺 `data/*.jsonl`
    （general/tool SFT、general/tool GRPO、OPD prompt pool）；
  - 新增 `data/tokenize.py`：2K/4K/8K pretrain + SFT general/tool tokenization；
  - `data/grpo.py` 增加 `multi_turn` / `gt`；`data/opd.py` 保留多轮历史；
  - `rl/tools.py`、`rl/agent_rollout.py`：MiniMind 多轮 tool rollout + 工具执行；
  - `rl/reward_model.py`：1.8B reward model 包装；
  - `rl/reward.py`：general/tool reward + MiniMind shaping + clip；
  - `rl/grpo.py` / `rl/opd.py`：token mask、多轮 engine、OPD observation mask；
  - `sft.py`：array/bytearray 缓冲，604M token SFT 从 ~50GB 降到 ~6GB；
  - 新增 general/tool SFT/GRPO/OPD/eval 配置、pipeline 配置、数据/服务器脚本；
  - 删除旧的 `data/raw`、`data/tokenized`、`data/processed` 与过时 manifest。
- **实现了什么**：
  - 平铺数据实际生成：`sft_tool` 84,832 tool + 20,204 general；`grpo_general`
    17,464 train / 1,940 eval；`grpo_tool` 34,734 train / 3,859 eval；
    `opd_prompts_mixed` 37,464（general 17,464 + tool 20,000）；
  - CPU 上 tool 多轮 rollout + GRPO 1 step 通过，observation 被 mask；
  - `pipeline.py` 支持 dry-run、阶段跳过、按 token meta 自动算 step。
- **验证**：
  - `.venv/bin/ruff check src tests scripts` → All checks passed
  - `.venv/bin/python -m pytest -q` → **217 passed**
  - `python -m minideepseek.data.build ...` → 上述 JSONL 数量
  - `python -m minideepseek.data.tokenize --smoke ...` → pretrain/SFT meta 正常
  - `tests/test_tool_agent.py` → 多轮 tool rollout / GRPO 1 step
- **未完成 / 风险**：
  - 完整 2.3B token 的 2K/4K/8K tokenization 未在本机跑（16GB 内存不足），
    需在 64GB+ 服务器执行；
  - 1.8B reward model 未下载/未在 CPU smoke（无 `transformers` 时按可选路径）；
  - code specialist、3-teacher OPD 未实现（数据/reward 缺口）；
  - tool rollout 的 CUDA/多卡与真实 145M 训练未验证。

## 2026-11-07 本地 reward model 与 debug 训练验证

- **commit**: `0e0da48` feat: add local reward-model smoke check
- **做了什么**：
  - 下载 `internlm/internlm2-1_8b-reward` 到代码根下
    `internlm2-1_8b-reward`（3.2 GB，未进 git）；HF 主站 SSL 不稳定，
    用 `hf-mirror.com` 的 `resolve/main` 直链下载；
  - `pyproject` 的 `hf` extra 补 `sentencepiece`、`protobuf`（InternLM2
    tokenizer 需要）；
  - 新增 `scripts/check_reward_model.py`，并在 `server_setup.sh` 中按需调用。
- **验证**：
  - reward tokenizer 加载：`InternLM2TokenizerFast`，vocab 92,544；
  - `LocalRewardModel` 加载 + 打分：score `-0.814453`（CPU，约 3.8 s）；
  - debug 模型训练：
    - pretrain smoke 2 step → `checkpoints/debug_smoke/final`；
    - SFT smoke 2 step → `checkpoints/debug_sft/final`；
    - GRPO debug 1 step → `checkpoints/debug_grpo/final`；
    - OPD debug 1 step → `checkpoints/debug_opd/final`；
    - general GRPO 1 step + 真实 1.8B RM（debug model，2 prompts）→
      `reward_model=-1.637`、`reward_mean=-2.137`、loss 有限；
  - `ruff check src tests scripts` → All checks passed；
  - `pytest -q` → 210 passed。
- **未完成 / 风险**：
  - 本地 `~/.cache` 受沙箱限制，运行 RM 需 `HF_HOME=$PWD/.cache/huggingface`；
    服务器无此限制；
  - 1.8B RM 在 CPU 上只做了单次打分/1 步 GRPO，未测 CUDA 批量吞吐；
  - 完整 2.3B token tokenization 与 145M GPU 训练仍未执行。

## 2026-11-07（三专家数据分工 / MoE 分化）

- **commit**: `2cf3aa5` docs: 三专家数据分工、MoE 分化方法与 145M 数据方案更新
- **做了什么**:
  - 新增 `doc/specialist_data_145m.md`：MiniMind 预训练/SFT 的领域构成、块状分布、
    三专家 SFT 甜点量、能否从现有语料提取；
  - 新增 `doc/moe_specialization.md`：领域同质 batch、平衡损失退火、路由指标验证、
    学习率计划的定位；
  - 重写 `doc/data_plan_145m.md`：专家改为 code/tool/general，预训练 3B 七源混料；
  - `doc/decisions.md` 追加 D63（1.2B→3B + code/tool 入混料）、D64（SFT 量按饱和点）。
- **实现了什么**（可验证的数据，不是"写完了"）:
  - `pretrain_t2t.jsonl` 抽样 3 万条：general 96.1% / code 2.1% / math 1.8% / **tool 0%**；
  - `sft_t2t_mini.jsonl` 905,718 条，tool 块位于第 771,075–905,718 行（134,644 条，14.9%）；
  - `sft_t2t.jsonl` 14.10 GB、约 857 万条，general 72.4% / code 18.0% / math 9.5%；
    按位置采样发现 code 在 83% 处占 72.6%、math 在 75% 处占 22.8%——数据是成块拼接的；
  - 由此确定：general 需按内容过滤后抽样，不能按行号截取。
- **验证**:
  - 全部数字由本地文件分窗口采样得到（12 窗口 × 4000 行）；
  - 未下载任何外部数据；外部数据集仅查 HF 元信息（规模/许可/字段）。
- **未完成 / 风险**:
  - 现存 `pretrain_t2t.jsonl` 仍是 2.2B 单源，3B 七源混料尚未落地（无 manifest、未打包）；
  - `code_reward` 仍是二值奖励，code GRPO 上线前必须先改部分分；
  - MoE 分化方案（领域 batch + 平衡损失退火）只有设计，未在 145M 上验证。

## 2026-11-07 code specialist 槽位 + RL 对齐 MiniMind

- **commit**: `31a7bb5` feat: complete code specialist slots and align RL with MiniMind
- **做了什么**：
  - `data/build.py` 增加 `--code-sft/--code-grpo/--code-eval`；生成
    `sft_code.jsonl`、`grpo_code_train/eval.jsonl`、`opd_prompts_code.jsonl`；
    缺 code 时写进 `specialists.yaml` 的 `missing`；
  - `configs/{sft,grpo,eval}/code.yaml` 改为 145M 档；code reward 权重
    `code_pass_reward/code_compile_bonus/code_syntax_penalty/code_format_bonus`
    显式化；
  - `configs/pipeline/minimind_3teacher.yaml`：sft_code → grpo_code →
    3-teacher OPD → eval_code；pipeline preflight 缺数据时提前报错；
  - `rl.rollout` 支持 `thinking_ratio`（general 0.9 / tool 0.1）；
  - `rl.loss_type` 支持 `cispo`；general/tool 配置用 CISPO + epsilon_high=5；
  - `pipeline` 支持 `sft_domains` 与 OPD teacher prompt 预检。
- **验证**：
  - `pytest -q` → **213 passed**；
  - `ruff check src tests scripts` → All checks passed；
  - `tests/test_build_data.py` 的 code 槽位测试通过；
  - `pipeline --config configs/pipeline/minimind_3teacher.yaml --to tokenize`
    → 明确列出 `sft_code.jsonl`、`grpo_code_train.jsonl`、
    `opd_prompts_code.jsonl`、`eval_code.jsonl` 缺失。
- **未完成 / 风险**：
  - code 数据尚未提供，code reward 权重只是首版；
  - 3-teacher OPD 未实际跑过；
  - MiniMind 的 KL(ref) / PPO 没有照搬，当前仍是 GRPO/CISPO + 可验证/奖励模型。

## 2026-11-08 梯度累积按 token 数加权

- **commit**: 见本条 commit
- **做了什么**:
  - `trainer._forward_backward` 由 `(total / accum).backward()` 改为
    `(total * n_i).backward()`（`n_i` = 该 micro-batch 的主头 target 数）；
  - `train_step` 在累积循环结束后统一除以 `Σn_i`，新增
    `train/distributed.py::scale_gradients`（DTensor 原地缩放，FSDP2 安全）；
  - 新增回归测试 `test_gradient_accumulation_weights_by_target_tokens`。
- **为什么**: `compute_loss` 返回的是**每个 micro-batch 内的 token 均值**，
  等权平均只有在各 micro-batch target 数相同时才等于真实的 token 加权均值。
  SFT 的 response-only mask 使各 micro-batch 的 assistant token 数差异很大
  （例：3 vs 5 → 等权把 5/8 的影响力给错了批），预训练因为 pack 等长所以原本无偏。
- **验证**:
  - `.venv/bin/python -m pytest -q` → **214 passed**（新增 1 条，对照
    `torch.autograd.grad` 的参考梯度，atol=1e-6）；
  - `.venv/bin/ruff check src tests scripts` → All checks passed；
  - 测试里断言两个 micro-batch 的 target 数确实为 3 和 5，排除"恰好相等"的假通过。
- **未完成 / 风险**:
  - 多卡下按 `1/Σn_i(本地)` 缩放，前提是各 rank token 数相等（packed batch 满足）；
    若将来做变长 shard，需要改成全局 `Σn`（一次 all-reduce）；
  - `target_tokens == 0` 的极端 batch 不会跳过 optimizer step（与改动前行为一致）。

## 2026-11-08 领域同质 batch 与 OPD teacher 锁定

- **commit**: 见本条 commit
- **做了什么**:
  - `data/mix.py`：`WeightedBatchMixer` 新增 `lockstep` / `batches_per_step`，
    配额可按 optimizer step 分配；新增 `current_index` / `current_source`
    与 `_MixerIterator`（迭代器同时暴露当前源）；
  - `train/run.py`：多源时按 `domain` 命名 source，并把 `grad_accum` 传成 `batches_per_step`；
  - `train/trainer.py`：每步记录 `domain_id` 到 `metrics.jsonl`；
  - `data/opd.py::sample_prompts` + `rl/opd.py`：`rl.teacher_lockstep` 开关（默认 false）；
  - `config.py`：`DataSettings.lockstep`（默认 true）、`RLSettings.teacher_lockstep`；
  - 新增 `doc/batching.md`，并把 `doc/data_plan_145m.md` 的预算/数据排布指向它。
- **实现了什么**:
  - `lockstep=True` 时，连续 `grad_accum` 个 batch 来自同一源（单测断言每 4 个一组同源）；
  - 多源训练时 `domain_id` 按 step 写入 metrics（单测断言两条记录为 0.0 / 1.0）。
- **验证**:
  - `.venv/bin/python -m pytest -q` → **218 passed**（新增 4 条）；
  - `.venv/bin/ruff check src tests scripts` → All checks passed。
- **未完成 / 风险**:
  - 按域打包的 `pretrain_{general,code,tool,mixed}_*.npz` 未生成（依赖 code 数据）；
  - 262k 配置与课程 `until_step` 未改（仍是 131k）；
  - OPD teacher 锁定默认关，收益未验证。

## 2026-11-08 外部语料下载（code + pretrain）

- **commit**: 见本条 commit
- **做了什么**:
  - 新增 `scripts/fetch_external_data.py`：按 shard 列表从 HF 拉取 17 个数据集，
    外层 8 次重试 + 断点续传，落 `download/<group>/<dataset>/`，结束后写
    `download/manifest.json`（bytes + sha256 + 上游 revision）；
  - `download/README.md`：逐数据集的实测行数、过滤后可用量、6400 词表 token 估算、
    复现命令、许可与评测边界；
  - `.gitignore` 忽略 `download/*`，只保留 README；
  - `doc/decisions.md` D66 记录“按 token 预算取 shard”的口径与 xLAM gated 的影响。
- **实现了什么**:
  - 50 个文件 / 21.66 GB，覆盖 pretrain 5 源、code SFT 6 源、code GRPO 2 源、eval 2 源；
  - 过滤后可用量（6400 词表抽样估算）：英文教育 641M、中文教育 1.23B、
    预训练 code ~750M（≤200 行口径）、合成教材 902M、math 419M、tool ~65M；
  - code SFT 可用 ~21.6 万条 / 131M token；GRPO 题池 MBPP 494 + LeetCode 2,641 +
    TACO EASY 3,940 + open-r1 15,068；
  - `Salesforce/xlam-function-calling-60k` 实测 gated（401），已从清单移除并记录替代方案。
- **验证**:
  - `python scripts/fetch_external_data.py --dry-run` → 50 files / 17 datasets；
  - 全部 parquet / Arrow stream / json / jsonl 可解析，行数取 parquet metadata 或
    Arrow batch 汇总；token 用 `minimind/model/tokenizer.json` 每文件抽样 2,000 行估算；
  - `ruff check scripts/fetch_external_data.py` → All checks passed；
  - HumanEval 164、HumanEval+ 164、MBPP test 500/257、TACO test 1,000 行数与上游一致。
- **未完成 / 风险**:
  - 尚未 tokenize，也没有写 `data/manifests/pretrain_3b.yaml` 的 `available_tokens`；
  - codeparrot 需 license 白名单过滤，当前 token 估算是“permissive + ≤200 行”口径；
  - python_code_instructions / open-r1 无 license 声明，正式训练前要确认；
  - `opencsg/chinese-fineweb-edu` v1 已 deprecated，暂按计划使用 MAP-CC slice。

## 2026-11-08 外部数据落盘 + 3B/262k 课程 + 流式打包

- **commit**: 见本条 commit
- **做了什么**:
  - 回滚 OPD teacher 锁定（`rl.teacher_lockstep`、`sample_prompts` 及其测试），
    保留 `_group_by_teacher` 的 KL 分组与 per-teacher 统计；
  - 新增 `scripts/build_external_data.py`：把 `download/` 的 17 个数据集转成
    `data/pretrain_{general,code,tool}.jsonl`、`data/sft_code.jsonl`、
    `data/grpo_code_train.jsonl`、`data/eval_code.jsonl`；
  - 新增 `scripts/pack_domain.py` + `scripts/build_pretrain_packs.sh`：按域、按长度
    流式打包 2K/4K/8K（内存 O(seq_len)）；
  - `curriculum.py`：`CurriculumStage` 支持 `sources`（每阶段可换域配比）；
  - `pretrain_145m.yaml` 改为 3B / 262k-per-step / 三源 lockstep；
    `curriculum_145m_4x4090.yaml` 改为 1717/8011/11444 步、三段数据不重叠；
  - 新增 `data/manifests/{pretrain_3b,sft_code,grpo_code}.yaml`。
- **实现了什么**（实测数字）:
  - `pretrain_general.jsonl` 1,470,192 文档 / 6.14B chars（≈2.06B token）；
    `pretrain_code.jsonl` 667,497 / 2.14B chars（≈923M token）；
    `pretrain_tool.jsonl` 124,260 / 266M chars（≈89M token）；
  - `sft_code.jsonl` 148,723 条 / 60.8M token；`grpo_code_train.jsonl` 3,135 条 /
    1.2M prompt token；`eval_code.jsonl` 328 条（HumanEval + HumanEval+）；
  - 分域 val：`val_{general,code,tool}_{2k,4k,8k}.npz` 各 1M token。
- **验证**:
  - `.venv/bin/ruff check src tests scripts` → All checks passed；
  - `pytest tests/test_curriculum.py tests/test_mix.py tests/test_trainer.py` → 全部通过
    （新增 2 条 curriculum sources 测试）；
  - 配置解析：三段 tokens = 0.450B / 1.650B / 0.900B，`lockstep=True`，`data.sources` 生效。
- **未完成 / 风险**:
  - `pretrain_{general,code,tool}_{2k,4k,8k}.npz` **正在后台打包**（`data/tokenization/pack.log`），
    本次提交时尚未全部完成；
  - 训练侧 per-domain val 还未接线（`train.val_path` 只支持单文件）；
  - `scripts/build_external_data.py` 未做跨源去重（依赖上游去重）。

## 2026-11-08（续）分域验证接线

- **做了什么**: `TrainSettings.val_sources` + `CurriculumStage.val_sources`；
  `trainer.evaluate` 按域复用，训练循环每 `val_every` 步额外写
  `val_loss_<domain>` / `val_main_loss_<domain>` / `val_tokens_<domain>`；
  `manifest._fingerprint` 支持只有 sidecar 的 packed 数据集。
- **实现了什么**: 三套配置（base 4K + 课程 2K/4K/8K）各自解析出 3 个 val 源。
- **验证**: `pytest -q` → 221 passed（新增 2 条：trainer 分域 val、curriculum val_sources）；
  `ruff check` 通过；配置解析打印每阶段 val 文件名正确。
- **未完成**: 打包仍在跑（`data/tokenization/pack.log`）。

## 2026-11-09 停止本机 tokenize，源数据精确去重并裁到 3B

- **做了什么**: 停掉本机打包（删除未完成的 npz）；新增 `scripts/dedup_and_split.py`；
  把 `pretrain_t2t.jsonl` 合并进 `pretrain_general.jsonl`；code/tool 按预算裁剪；
  更新 `pretrain_3b`/`pretrain_local` manifest（新增 `tool`/`mixed` 域）。
- **实现了什么**（实测）:
  - general：9,939,019 文档 → 精确去重删 10,645（0.1%）→ 4.56B token →
    哈希裁剪保留 54% → **5,362,052 文档 / 2.46B token**；
  - code：→ 310,244 文档 / **0.45B token**（去重 6,364，裁剪 350,889）；
  - tool：→ 114,737 文档 / **0.09B token**（去重 9,237）；
  - 三个文件合计 3.0B token，`data/pretrain_t2t.jsonl` 已合并删除。
- **验证**: `prepare manifest --manifest data/manifests/pretrain_3b.yaml` → errors 为空
  （仅有 packed 路径待服务器生成的 warning）；`wc -l` 与统计一致。
- **未完成**: 打包仍在服务器上做；本机 `data/tokenization/` 只剩 debug 数据。

## 2026-11-09 清理 MiniMind 身份数据并换成本项目身份

- **做了什么**: 新增 `scripts/scrub_identity.py`；对 9 个 SFT/GRPO/OPD/eval 文件
  按 `jingyaogong | gongjy | minimind | 小参数AI模型` 整条删除；生成
  `data/identity_sft.jsonl`（17 条）并追加到 `data/sft_general.jsonl`。
- **实现了什么**:
  - 删除：sft_general 28,136 / sft_tool 5,484 / grpo_general 4 / grpo_tool 774 /
    eval_general 1 / eval_tool 79 / opd_general 4 / opd_tool 464 / opd_mixed 468；
  - 预训练三个文件复查为 0 命中，未改动；
  - 新身份："我是 mini-deepseek，基于 DeepSeek-V4 架构的小型 MoE，由 chengjionghao 训练"。
- **验证**:
  - `grep -c -i -E 'jingyaogong|gongjy|minimind' data/*.jsonl` → 全部 0；
  - `encode_sft_record` 对身份样本可编码（66 token，50 个 target，think 空块正常）。
- **风险**: 身份数据只有 17 条，145M 模型可能记得不牢；需要时可在 OPD/SFT 里加权。

## 2026-11-09（续）身份数据扩到 69 条

- **做了什么**: `IDENTITY_PAIRS` 从 17 条扩到 69 条（6 组问法、中英各半），
  重跑 `scripts/scrub_identity.py`（幂等：先删旧的 `source=identity` 再追加）。
- **实现了什么**: `data/identity_sft.jsonl` 69 条；`sft_general.jsonl` 末尾恰好 69 条
  identity 记录（旧的 17 条已替换）。
- **验证**: 旧标记 `jingyaogong|gongjy|minimind` 全库 0 命中；
  抽样编码正常（66/50、54/37、48/36 token/target）。
- **未完成**: 69 条仍是"小模型够用"的量级，若身份不稳可在 SFT 混料里加权。

## 2026-11-09 8K 阶段真正跑 sparse top-k

- **commit**: 见本条 commit
- **做了什么**:
  - `IndexTopKSchedule` 增加显式窗口 `start_step`/`end_step`，优先于旧的
    end-of-run `warmup_ratio`；
  - `TrainSettings` 增加 `index_topk_start_step`/`index_topk_end_step`；
  - `CurriculumStage` 增加 `index_topk_final` / `index_topk_warmup_ratio`，
    `_stage_config` 按“本阶段首步 + 阶段比例”生成绝对窗口；
  - `curriculum_145m_4x4090.yaml` 8K 阶段：2048 → 64 在前 10%（343 步）
    线性收缩，其后保持 sparse。
- **为什么**: 原 schedule 以全程 11,444 步为分母，只在 10,872–11,444 收缩，
  8K 阶段 95% 仍是 dense（2048 = 8K 全部 CSA 项），稀疏路径只跑了 5%。
- **验证**:
  - `pytest -q` → **225 passed**（新增 4 条：显式窗口数学、窗口校验、
    curriculum 窗口接线、trainer 应用窗口）；
  - 配置解析：`stage2 seq=8192 model_topk=2048 final=64 window=8011..8354`；
  - `ruff check src tests scripts` → All checks passed。
- **未完成 / 风险**:
  - 稀疏段占比随 8K 阶段比例变化（当前 ~27% 全程 token），若把 8K 降到 25%
    会降到 ~22%；
  - 4K 推理默认 top-k=1024（dense），与 8K 的 64 不同；如需完全一致，4K 评测
    可显式设 64 做对照。

## 2026-11-09 download/ 不进 git

- **commit**: 见本条 commit
- **做了什么**: `.gitignore` 从“忽略 `download/*`、保留 README”改为忽略整个 `download/`；
  说明文档 `download/README.md` 移到 `doc/external_data.md`，脚本 docstring 同步。
- **为什么**: `download/` 是 21.66 GB 原始语料，只应存在于本地/服务器，不应出现在仓库
  任何位置。
- **验证**: `git ls-files download/` 为空；`git check-ignore download/manifest.json`
  命中；`ruff check` 通过。
- **未完成 / 风险**: 21.66 GB 数据从未被 git 跟踪，但 `download/README.md` 曾出现在
  本地历史 commit `c6e1279`（纯文档、无数据）；当前分支尚未 push。

## 2026-11-09 断点续训

- **commit**: `ccf1f64` feat: 断点续训——信号优雅停、auto_resume、final 只表示阶段完成
- **做了什么**:
  - `Trainer` 安装 SIGTERM/SIGINT 处理器：第一次信号跑完当前 step、写
    `step_NNNNNN` 后退出；第二次立即 `KeyboardInterrupt`；
  - `final/` 语义收紧为“阶段完成”，被打断只写 `step_*`；
  - `train.auto_resume`（`pretrain_145m.yaml` 打开）：重跑同一条命令从
    `out_dir` 里最新的 `final`/`step_*` 续训；
  - `run_curriculum` 在 `summary.interrupted` 时停止推进下一阶段；
  - `doc/server_runbook.md` §7 重写为断点续训说明；
  - 新增 `tests/test_resume.py`（8 个用例）。
- **实现了什么**:
  - debug 课程实测：SIGTERM 打断在第 8 步 → 目录里只有 `step_000008`（无
    `final`）→ 重跑从第 8 步继续到 30 步，再进 4K 阶段跑到 40 步完成；
  - 恢复后 step / LR / index_topk / MTP 权重连续（都按全局 step 计算）。
- **验证**:
  - `.venv/bin/python -m pytest -q` → 233 passed；
  - `.venv/bin/ruff check src tests scripts` → All checks passed；
  - `scripts/train_curriculum.py --config <demo>` + `kill -TERM` +
    重跑同命令 → 两个 stage 均 `interrupted: false` 且写出 `final`。
- **未完成 / 风险**:
  - 硬杀（OOM kill / 掉电）仍只回退到最近 `ckpt_every`（200 步）；
  - 多卡要求所有 rank 都收到信号（torchrun/Slurm/k8s 正常；单卡 Xid 掉线会
    卡在集合通信，需要先杀进程）；
  - GRPO/OPD 有各自 trainer，未接信号处理，中断从配置 `resume` 重跑该 stage。

## 2026-11-09 8K 阶段 token/step 修正与卡数自适应配置

- **commit**: 见本条 commit
- **做了什么**:
  - `train/curriculum.py::CurriculumStage` 增加 `grad_accum` 覆盖，`_stage_config` 应用，
    `validate_stages` 校验为正；
  - 新增 `scripts/autoconfig_145m.py`：按卡数生成 `configs/pretrain/generated/` 下的
    base + curriculum 配置，保持 262,144 token/optimizer-step；
  - 新增 `doc/run_145m_gpuhome.md`：Mac 打包 → rsync → 无卡准备 → 切 GPU 开训的完整手册；
  - 测试补 2 条（grad_accum 覆盖 / 非法值）。
- **实现了什么**:
  - 定位并修复 8K 阶段每步 524,288 token 的问题（loader 每 rank 至少一个 8K pack）；
  - 1/2/4/8 卡生成的 tokens/step 均为 262,144：
    4 卡时 2K/4K=16384×16、8K=32768×8；
  - 卡数非 1/2/4/8/16 时脚本报错而不是悄悄改变 token 预算。
- **验证**:
  - `.venv/bin/python scripts/autoconfig_145m.py --gpus {1,2,4,8} --dry-run` 输出符合手算表；
  - `.venv/bin/python -m pytest -q` → **235 passed**；
  - `ruff check` 通过。
- **未完成 / 风险**:
  - 生成的配置未在真实多卡上跑过，需在 GPU 实例上先 `benchmark.py` 冒烟；
  - `--gpus 16` 未实测（8K micro 131,072，显存不受影响但需确认 FSDP2 分片）；
  - 文档里的 gpuhome 流程未在服务器实测，只核对了脚本与路径。

## 2026-11-09 开训前审计：验证集缺失 / venv 无 torchrun / stage 内续跑回放

- **commit**: 见本条 commit
- **做了什么**:
  - runbook 补 9 个验证 pack 的打包（§1）、单独压缩与上传（§2/§5）、解压校验（§6）；
  - `scripts/train_{curriculum,pretrain,sft,grpo,opd}.sh` 改用
    `.venv/bin/python -m torch.distributed.run`，不再依赖 `.venv/bin/torchrun`；
  - `train/curriculum.py`：`_stage_config` 增加 `resume_data` 参数，
    `run_curriculum` 只在“续本 stage 自己的 checkpoint”时置 True；
  - runbook 更正 `code_8k` 说明（按 pack 数不会重放）、压缩体积 3.4 GB、
    加 8K+重算峰值预检与 8K OOM 回退；`*.tar`/`*.zst` 加入 `.gitignore`。
- **实现了什么**:
  - 修掉第一个阻塞：`val_*` 从未打包/上传，`run()` 在 stage 0 启动时加载
    `val_sources` 直接 `FileNotFoundError`（已复现）。本机产出 9 个
    `val_{general,code,tool}_{2k,4k,8k}`（各 1.0M token），打成
    `val_packs.tar.zst`（7.6 MB）；
  - 修掉第二个阻塞：`--system-site-packages` 建的 venv 没有 `torchrun`
    console script（脚本在系统 bin），§8.3/§8.4 多卡启动会 “No such file”；
  - 修掉续跑回放：原先 stage 内中断后 `resume_data=False`，数据从 stage 开头
    重放（4K 阶段中途续跑最多重放 ~1.1B token，且尾部数据永远看不到）；
    现在同 stage 续跑按 `data_batches` 精确跳过。
- **验证**:
  - `.venv/bin/python -m pytest -q` → **237 passed**（新增 2 条 curriculum 用例）；
    `ruff check` 通过；
  - `autoconfig_145m.py --gpus 4 --dry-run` → 16384×16 / 16384×16 / 32768×8，
    三阶段 tokens/step 均为 262,144；`--gpus 3` 按预期报错；
  - 生成配置解析后 9 个训练源 + 9 个验证源全部加载，并通过 tokenizer hash /
    seq_len 校验；
  - debug 课程实跑两轮（stage1 `until_step` 4→8）：第二轮从 `step=4` 续跑到
    `step=8`，`data_batches` 2→6（只新增 4 个 micro-batch，确实跳过 2 个）；
  - `python -m torch.distributed.run --help` 正常。
- **未完成 / 风险**:
  - macOS 上 torchrun 起 2 进程 gloo 会崩在 gloo UV transport
    （`pair.cc:253 op.nread == op.preamble.nbytes`）；旧 `.venv/bin/torchrun`
    同样崩，属本机环境问题，**多卡 NCCL 路径未在本机验证**；
  - 8K + 激活重算的真实显存仍无 CUDA 实测（硬件预算文里 27.9 GiB 是未开重算的
    估算，PROCESS 2026-11-07 估 12–15 GiB）；runbook 已加 `measure_activation_budget`
    预检与 `index_topk=1024` 回退；
  - 换卡数续跑时 `data_batches` 的跳过量只是近似。

## 2026-11-10

- commit: 931be68 perf: micro-batch 8192 + attention 边界 bf16 + MoE 排序 dispatch
- 做了什么: 按「micro-batch → attention dtype → MoE dispatch」改了三处，并改了
  `autoconfig_145m.py` 的默认 per-rank token 数。
- 实现了什么:
  - `benchmark.py` 4K×1：6,220 → **7,357 tok/s（+18%）**；4K×2：9,304 → **11,910
    tok/s（+28%）**，峰值 11.9 GiB；对照原始 4K×1 的 6,220 是 **1.92x**；
  - 16384 token（4K×4 / 2K×8）在 20.4 GiB 处 OOM，确认 8192 是 24 GB 卡上限；
  - 三阶段 `batch_tokens=8192`、`grad_accum=32`，每步 token 仍 262,144；
  - 新增 fp32 输入下 fused-vs-reference 测试、MoE 排序 dispatch 的 fp32/bf16
    等价测试（含空专家分支）。
- 验证: 本地 `pytest -q` **242 passed, 2 skipped**；4090 上
  `tests/test_moe_dispatch.py tests/test_gather_attention.py` **6 passed**；
  benchmark 数字见上。
- 未完成 / 风险:
  - trainer 每个 micro-batch 仍有 8 次 `float()` 同步（下一步，改完需重新 A/B）；
  - mHC 的 20 次 Sinkhorn 仍是纯 eager，未融合；
  - 2K 阶段的 bf16 shape 首次仍要十几分钟 Triton 编译（缓存后不再付）；
  - 8K 阶段（8192/rank + activation checkpointing）未在 24 GB 上实测。

## 2026-11-10

- commit: c30ae40 feat: 训练报告补分领域验证/调度/稳定性面板，并打开 report_every
- 做了什么: 训练可视化补全。`viz/report.py` 从 10 个面板加到最多 17 个，卡片从 10 个
  加到 12 个；`trainer._write_report` 改为从磁盘 metrics.jsonl 取数；base 配置打开
  `report_every: 200`。
- 实现了什么:
  - 新面板：分领域验证损失（`val_loss_<domain>` 各一条，聚合 `val_loss` 作参考）、
    Schedules（`mtp_weight` + `lr_factor`）、CSA index top-k（`index_topk`）、
    Timing（`step_time_s` + `eta_seconds`，对数轴）、SwiGLU activation magnitude
    （`swiglu_*_abs_max`）、Stability events（`loss_spike`/`ar_active`/`rollback`，
    只在事件发生时出现，同一 step 的多个 flag 只算一步）；
  - 新卡片：ETA（`45s`/`3.4h`/`2.1d`）、CSA top-k；
  - 续跑不再截断报告：原实现用 `logger.history` 渲染，`auto_resume` 后内存为空，
    report.html 会丢掉中断前的所有步；
  - `build_report` 的返回值新增 `val_domains`，`plot_training.py` 会把它打出来。
- 验证:
  - 本地 `pytest tests/ -q` → **248 passed, 2 skipped**；`ruff check` 通过
    （`ruff format --diff` 对该文件本来就有既有差异，没有顺手重排）；
  - 服务器上用真实 metrics.jsonl（step 0–30，4 条记录）生成 **14 面板**；
  - 1203 条合成数据（3 领域 + 2 次尖峰）生成 **17 面板**，
    `val_domains=[code, general, tool]`，`Stability events (2 steps)`；
  - 服务器上 `pytest tests/test_viz.py tests/test_trainer.py` → **34 passed**；
  - 跨阶段合并（`cat stage_0*/metrics.jsonl` → `plot_training.py`）实测可用。
- 未完成 / 风险:
  - `report_every` 的真实训练路径只在 CPU debug 模型上验证过，**没在 4090 上跑**；
  - `val_main_loss_<domain>` 已经记录了但没画（比 `val_loss_<domain>` 更接近 ppl 口径）；
  - 三个 stage 各一份 report.html，跨阶段视图只能手工 cat；
  - 服务器上 `configs/pretrain/generated/*.yaml` 已按 `autoconfig_145m.py --gpus 1`
    重新生成（micro 8192 / grad_accum 32，与 931be68 的默认值一致），
    上一次运行用的是旧的 4096/64，续跑时 `data_batches` 跳过量只是近似。

## 2026-11-10（下午，续）

- commit: f770267 perf: 训练循环去同步 + mHC Sinkhorn 走 torch.compile
- 做了什么: 把每 micro-batch 的 host 同步从 ~230 次降到 ~4 次（SwiGLU 监控、
  routing_metrics、trainer 标量），mHC 的 20 次 Sinkhorn 在 CUDA 上走
  torch.compile（带回退）。
- 实现了什么:
  - mHC 单次调用 1.21 ms → **0.557 ms（2.17x）**，数值误差 1.8e-7；
  - 4K×2 benchmark 11,910 → **11,987 tok/s**（+0.6%，同步收益 benchmark 不覆盖）；
  - 新增 `tests/test_mhc.py`（eager/compiled 一致、双随机矩阵结构）。
- 验证: 本地 `pytest -q` **251 passed, 2 skipped**；4090 上 mhc/moe/attention
  **9 passed**；benchmark 见上。
- 未完成 / 风险: 训练端真实收益待 step 10 后的 tok/s 确认；2K 阶段 bf16 的
  Triton 首次编译仍需 10–20 分钟；`torch._grouped_mm`（torchtitan 的 MoE 做法）
  实测 1.46x 但整步仅 ~0.7%，暂不接。

## 2026-11-10

- commit: b2b84af feat: 路由可观测性——hash/learned 拆分、逐层×逐专家快照、bias 轨迹
- 做了什么: 把「采了但没留下」的路由数据落盘，并修了一个路由指标的正确性问题。
- 实现了什么:
  - `routing_metrics` 按 `is_hash` 拆成 `router_learned_*` / `router_hash_*` 两组
    （原来 2 个哈希层和 6 个学习层被平均进同一个 max/mean）。实测 145M 配置：
    学习层 1.53、哈希层 1.11、混合 1.43，**学习层真实不均衡被稀释 0.10**；
  - 新增 `min_relative_load`：6 专家 top-2 下 `experts_used_fraction` 恒为 1.0，
    没有信息量；最小份额才是有信息的那一端；
  - 新增 `train.router_every`（预训练配置设 50）：把 `{router: {hash, counts[E],
    bias[E]}}` 跨 micro-batch 求和后写进 `metrics.jsonl`。`record_routing` 本来
    每步都在跑，但 `routing_metrics` 压成 5 个标量就丢了；
  - 顺带记录 `expert_bias_E`——aux-loss-free 均衡器的全部状态，此前一个数都没有；
  - 顺带记录 `domain_name`（mixer 的 `current_source`），使快照天然可按领域聚合；
  - report.html 新增 Layer × Expert 热力图、Domain × Expert 富集、Router bias 轨迹；
    卡片 "experts used" → "min load share"；旧 metrics.jsonl 仍可渲染（key 回落）。
- 验证:
  - 本地 `pytest tests/ -q` → **260 passed, 2 skipped**；
    `ruff check src/ tests/ scripts/` 全绿（顺便修了 HEAD 上就存在的 B905 与 F401）；
  - 服务器 `pytest tests/test_routing.py tests/test_viz.py` → **27 passed**；
  - 服务器用旧 metrics.jsonl（无 `router_layers`）仍生成 14 面板；
  - 用 145M 真实模型（CPU）跑 12 个 step 生成真实 routing 报告：8 个 router
    （L0/L1 为 hash）、L6/E0=1.40 vs E4=0.68 的初始化不均衡、bias spread
    0.002→0.018 单调上升。
- 未完成 / 风险:
  - **没在 4090 上跑过带 `router_every` 的真实训练**，CUDA 数值路径未验证；
  - 未做 update-to-weight ratio / expert 梯度范数 / expert 输出贡献 E[g·||h||]；
  - 哈希层的"专业化"是 `tid2eid` 表的性质，不能当成学出来的分工来解读。

## 2026-11-10（夜间：编译问题定位）

- 结论: **`MINIDEEPSEEK_ATTN_AUTOTUNE=0` 会强制选中一个会被 prune 掉的 tile
  config，该 config 的 Triton 编译在本机是病理性的（>40 分钟不产出）**。关闭限制、
  走完整 autotune 后，隔离脚本在冷缓存下 **86 秒**编完，稳态 **12,637 tok/s**
  （4K/2doc、fp32 fused）。但训练进程仍会在某个变体上卡住（24+ 分钟），
  尚未定位到具体是哪个 shape/方向的组合。
- 另一个实测: `MINIDEEPSEEK_ATTN_BACKEND=torch` 在训练真实 batch 下是
  **203–235 s/step（~870 tok/s）**，比 fused 的 27.8 s 慢 **8.4x**，不能当 fallback。
- 已做的代码改动: `MINIDEEPSEEK_ROUTING_GPU`（routing 记录不落 CPU）、
  bf16 cast 开关、mHC compile 开关；见 `doc/decisions.md` D72。
- 当前训练进程仍在跑（tmux `train`），处于编译阶段；日志
  `logs/curriculum_145m.log`。morning checklist 见 `doc/optimization_2026-11-10.md` §7。

## 2026-11-10（下午：冷编译收口）

- commit: 103a096 fix: 冷编译改为一次性 prewarm + 仓库内持久缓存
- 做了什么: 把"第一次启动要编 20-40 分钟、看起来像挂死"这件事从训练路径里
  移出去，并撤掉夜里那个反而更慢的开关。
  - 删除 `MINIDEEPSEEK_ATTN_AUTOTUNE=0` 与 `_limit_attn_gym_autotune`；
  - `paths.configure_compile_cache()`：三个 kernel 缓存指向 `PROJECT_ROOT/.cache`，
    显式环境变量优先，`train_curriculum.sh` / `train_pretrain.sh` 同步导出，
    `_load_attn_gym()` 在 import attn_gym 前兜底；
  - 新增 `scripts/prewarm_attn.py`（逐变体打印编译耗时，强制 attn_gym 后端，
    `--dry-run` 可在无卡机跑）；
  - `train.benchmark` 拆出 `synthetic_batch_factory`，benchmark / prewarm 共用；
  - `curriculum.stage_train_configs()`：不碰数据集也能拿到每个 stage 的真实
    `seq_len` / `index_topk` / `batch_tokens`。
- 实现了什么:
  - 训练启动不再含编译：prewarm 一次写缓存，之后每个 stage、每次重启都是 0；
  - prewarm 的 batch 形状与训练一致：2K→2 doc、4K→2 doc、8K→1 doc，
    `index_topk` 512/1024/2048 三个 specialization 都会被覆盖；
  - 超时变体会被单独标出（夜里那个 >40 分钟的病态 tile 不会再无声消失）。
- 验证:
  - 本地 `pytest tests/ -q` → **267 passed, 2 skipped**；`ruff check` 全绿；
  - 服务器（无卡）`pytest tests/test_prewarm.py tests/test_paths.py tests/test_curriculum.py`
    → **18 passed**；
  - 服务器用真实 `configs/pretrain/generated/curriculum_145m_g1.yaml` 跑
    `prewarm_attn.py --dry-run` → 三阶段 plan 正确（含 8K 的 docs=1、topk=2048）；
  - 服务器工作区已 reset 到该 commit（仅 `.venv` 与 `mini_145m_a60m_topk64.yaml`
    是 untracked）。
- 未完成 / 风险:
  - **CUDA 上的真实编译与缓存命中没验证**：本机和服务器当前都是无卡模式，
    `nvidia-smi` 无输出、`/dev/nvidia*` 不存在；
  - prewarm 的 8K 阶段按 `train.activation_checkpointing` 建模型以对齐显存，
    但 24 GB 上 8K×1 的真实峰值仍未实测；
  - bf16 fused 仍默认关闭，需要 `MINIDEEPSEEK_ATTN_BF16=1` + 单独 prewarm 一次。
- 补充（同上一条 commit）: prewarm 改成**逐 stage 容错**——某个 stage 失败（例如 8K
  在 24 GB 上 OOM）不会连带丢掉前面已预热的 2K/4K，失败原因进 summary 并以非零退出；
  另加 `--max-seq-len` 用来先跳过 8K、`--fail-fast` 用于 CI。已暖的 kernel 留在缓存里，
  重跑会从中断处继续。

## 2026-11-10（晚间：reward model 上服务器）

- commit: fix: 奖励模型依赖补 einops + 强制 venv 内 protobuf，runbook 记录下载与校验（本条即该 commit）
- 做了什么:
  - 把 `internlm2-1_8b-reward`（3.2 GB，12 个文件）下到
    `/root/rivermind-data/miniDeepSeek/internlm2-1_8b-reward`（代码根，配置里的相对
    路径就是这个），`.gitignore` 已排除；
  - 服务器连不上 HF/github，hf-mirror 大文件 307 到 `api/resolve-cache` 后卡死，
    改走 ModelScope 的 repo 文件 API + aria2c 8 连接（~15 MB/s，2–3 分钟）；
  - 修两个装不上的坑：`[hf]` 缺 `einops`（InternLM2 remote code 直接 import）；
    `--system-site-packages` 让 pip 复用镜像 protobuf 7.35.1，新 sentencepiece
    的 gencode 7.36.1 报 `VersionError`。
- 实现了什么:
  - `pip install -e '.[dev,hf]'` 现在一次到位（einops 进 extra，脚本补
    `pip install --ignore-installed 'protobuf>=5'` 并 import 自检）；
  - runbook §5 记录了下载命令、两个权重的 sha256、依赖坑和打分自检。
- 验证:
  - 服务器 /tmp 隔离 venv 跑改后的 `setup_env_gpu.sh`：transformers 4.57.6、
    sentencepiece 0.2.3、protobuf 7.36.2、einops 0.8.2，torch 2.12.1+cu130 可见 4090；
  - `scripts/check_reward_model.py --path internlm2-1_8b-reward --dtype float32`
    （CPU，`CUDA_VISIBLE_DEVICES=`）→ `reward=-0.812751`，与本机 fp16 的
    `-0.814453` 一致（精度差）；11 个文件 sha256 与本地副本逐一相同；
  - 本地 `pytest -q` → **271 passed, 2 skipped**。
- 未完成 / 风险:
  - 服务器 `/opt/mds-venv` 还没重装（当时有 `prewarm_attn.py` 在跑），所以 GRPO
    general 在服务器上仍跑不了；需要时重跑一次 `bash scripts/setup_env_gpu.sh`；
  - RM 的 CUDA 批量打分吞吐仍未测（只在 CPU 上验证过单条打分）。

## 2026-11-10（晚：4090 上跑通训练）

- commit: f811b50 fix: HCA 的 top-k 改为静态上界，不再随 batch 布局重编
- 做了什么: 在真 4090 上把"能训练"这件事收口。
  - `index_topk=64`、`hca_topk_max=128`、schedule 量化到 2 的幂（D74）；
  - `_hca_topk` 的宽度从 `int(limit.max().item())`（数据相关 + 每层一次 D2H
    同步）改为静态 `ceil(seq_len / compress_ratio)`，短文档由 `cmp < limit` 掩掉。
- 实现了什么:
  - 编译成本：`K=64` 冷编 89.2s、`K=128` 459s、`K=256` >1h（夜里的 28min 是
    `K=512`）；现在 2K stage 两个变体 **0s 编译**（全命中缓存）；
  - 重编消失：改之前训练进程 2 分钟内新建 17 个 cache 目录、GPU 0% 空转；
    改之后 prewarm 0s、训练 0 新建目录；
  - 8K 的 `K=256` 病态编译被 `hca_topk_max=128` 消掉（8K 只保留最近 128 个压缩
    条目），最坏情况退回一次 ~5 分钟的 `K=128` 编译。
- 验证（4090，24 GB）:
  - `prewarm_attn.py --curriculum ... --max-seq-len 2048` → 2 variants, **0s 编译**，
    稳态 **7,221 tok/s**，11.4 GiB；
  - 训练 `step 0` → `step 10`：loss 12.0116 → **11.9252**（main 9.2386→9.1565），
    |g| 3.33，**5.85k tok/s**，12.0 GiB 峰值，ETA 4.9d；
  - 同机 benchmark（4K×2，load ~100 时）：8,545 tok/s / 11.4 GiB。
- 未完成 / 风险:
  - 吞吐 5.85k vs 历史 8.05k：这台机器 load 150–250（128 核超卖），进程 pcpu
    973%，host 侧被抢；等负载降下来再复测，不能拿现在的数字当结论；
  - 8K stage 仍未端到端跑过：`docs=1` 时 `num_documents==1` 会命中 Triton 的
    "==1" specialization，可能还要一次 `K=128` 编译（分钟级，不是小时级）。

## 2026-11-10（夜：吞吐 A/B + HCA 语义收口）

- commit: 5d4a928 perf: 路由/SwiGLU 诊断只在累加组最后一个 micro-batch 采
- commit: 531408a fix: 去掉 HCA 的 128 封顶，8K 恢复覆盖全部压缩段（D75）
- 做了什么:
  - `routing_metrics` / `swiglu.as_dict()` / `layer_expert_counts` 只在
    `micro == accum-1` 时执行（一个 step 32 个 micro-batch → 1 次）；
  - HCA 去掉 `hca_topk_max`，宽度回到静态 `ceil(seq_len/ratio)` = 64/128/256。
- 实现了什么（4090 A/B，同一台机器，前后各跑 10+ step）:
  - `step_time_s` **33.0 → 21.8**；tok/s **5.82k → 9.43k（1.62x）**；
    ETA **4.7d → 3.0d**；
  - GPU 占用率 1Hz 采样：**38.7% → 72%**；
  - `step 10` 的 loss/main/mtp/|g| 与改前**逐位相同**（11.9252 / 9.1565 /
    9.2229 / 3.329）——只动了观测，没动训练。
- 未完成 / 风险:
  - 8K 的 `K=256` 变体首次编译 >1 h，需找 GPU 空闲窗口预热（训练占着 12 GiB，
    8K 预检也要 ~12 GiB）；8K stage 在 step 8011，约 2.5 天后才到。

## 2026-11-10（新实例 hn01-30167）

- commit: 6c3cbf2 feat: 每次启动一个 run 目录，日志与 metrics 按启动时间/模型命名
- commit: 6700317 fix: grad_accum>1 时路由/SwiGLU 指标被整个丢掉
- 做了什么: 直接回应“好几次训练都记到同一个 metrics.jsonl”。根因不是日志设计本身，
  而是那几次启动都没存过 checkpoint（ckpt_every=200，只跑到 30–80 步），
  auto_resume 找不到可续点，每次从 step 0 重来，全部追加进同一文件 → 文件里有
  3 段（0→30、0→30、0→70）和两次 step 回退，report 的 loss 在无关曲线间来回跳。
- 实现了什么:
  - `minideepseek/runs.py`：run id = `启动时刻_模型_参数量_卡数`，
    每次启动分配 `<out_dir>/runs/<run_id>/`；同秒冲突加 `-2` 后缀；
  - `MetricsLogger` 写 run 目录，每条记录带 `run_id`；
  - trainer 写 `meta.json`（启动时间、argv、hostname、device/卡数、tokenizer hash、
    参数量、git commit/dirty、planned_steps）与 `config.yaml`，resume 时补
    `resumed_from` / `resumed_at_step`；
  - report.html 按 lineage 渲染（当前 run + 续跑来的祖先，废弃的 step-0 分支不混入）；
  - `scripts/launch_curriculum.sh`：一次启动一个时间戳，同时决定
    `logs/<run_id>_<config>.log` 与 `MINIDEEPSEEK_RUN_ID`；
  - `plot_training.py` 支持 metrics 文件 / run 目录 / stage 目录，以及
    `--run lineage|latest|all|<id>`；
  - **顺带修了 5d4a928 引入的回归**：`_merge_dicts` 用 key 集合交集，而诊断只在
    最后一个 micro-batch 采 → 交集为空 → `grad_accum>1` 时 metrics 里一个
    router/swiglu 字段都没有。实测当前这次训练 steps 0–80 全是 0 个路由字段。
- 验证: 本地 `pytest tests/` → **280 passed, 2 skipped**；`ruff check src/ tests/ scripts/`
  通过；服务器上旧 stage（扁平文件、3 段混合）→ 22 条记录、16 面板仍可渲染。
- 未完成 / 风险:
  - **已跑的这次训练用的是旧代码**，要重启才会写进 run 目录；它在 step 84 收到
    SIGTERM 并存了 `step_000085`，所以下次启动会续跑（step 85）而不是从 0 重来；
  - 服务器上当时的 pytest 与训练抢 CPU（load average 130+），跑测试前应设
    `OMP_NUM_THREADS=1` 或先停训练；
  - 工作区里 `model/mhc.py`、`model/moe.py` 有别人未提交的改动，本次未触碰；
  - 旧的扁平 `metrics.jsonl` 保留未动，供对比。

## 2026-11-10（夜：实测分解 + mHC/MoE 内存重写，续训 12.0k tok/s）

- commit: perf: mHC 分支混合改 bmm、MoE dispatch 去掉 bf16 原子累加
- 做了什么:
  - 在**真 4090 上独占 GPU**（SIGTERM → 存 `step_000085` → 定位 → 重启续跑）做了
    一次干净的分解测量：`data / fwd+bwd / clip / optimizer` 四段分别计时，
    `torch.profiler` 出 CUDA kernel 口径的 profile，再用 CUPTI kernel 时间做 A/B
    （宿主 CPU 被邻居抢到 load 77–185，墙钟口径噪声 ±30%，不可用）；
  - `HcPost` 的 `sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)` 改成
    `torch.bmm(comb.transpose(1, 2), residual)`；`HcPre`/`HcHead` 的同型收缩
    抽成 `_contract()` 走 bmm；
  - `RoutedExperts` 的 dispatch 尾部从 `index_add_`（bf16 原子累加）改成
    `inverse[order]` 反置换 + `view(T, k, dim).sum(1)`；**不能省 transpose**：
    `sum(..., dim=1)` 求和的是 comb 的第一维，等价于 `combᵀ @ residual`，
    而 Sinkhorn 之后的 comb 并不对称（写错过一次，max|Δ|=5.5 才发现）。
- 实现了什么:
  - **分解结果（8192 token/micro-batch，2K stage）**：
    `data` 0.5ms/batch（可忽略）、`fwd+bwd` 705ms、`clip` 3ms、`optimizer` 356ms；
    32 个 micro-batch 推算 22.9s/step，与线上 21.2–24.4s 吻合；
  - GPU kernel 时间占比：**attention 40%**（fp32，5 层共 205ms/次）、`aten::mm` 27%、
    MoE dispatch 15%、mHC 的 mul+sum 7.5%；
  - 这两个改动**数值等价**：同一 seed、同一真实 batch，`main loss` 6 位小数完全相同
    （9.177011）；单测 23 passed / 1 skipped（mhc、moe、router、expert 相关）；
  - 线上效果（同一进程，改前 step 80 vs 改后 step 100）：**22.4s → 17.3–19.0s**，
    **8.4–9.4k → 10.5–12.0k tok/s**，ETA **3.0d → 2.4d**；峰值显存 12.8 GiB。
  - 其中一部分来自 `MINIDEEPSEEK_ROUTING_GPU=1`（本次重启才真正打开，见 D72）。
- 验证:
  - CUPTI kernel 时间 A/B：base 1113ms/iter → patched 1104ms/iter（-0.8%），
    `mhc_mulsum` 84→83ms、`moe_dispatch` 168→164ms —— 省的主要是显存搬运
    （`[T,4,4,768]` fp32 单次 402 MiB 不再物化），墙钟上被 attention 盖住；
  - 线上 `grep ^step logs/curriculum_145m.log`：step 100 `17.34s / 12,017 tok/s`，
    loss 8.27→7.28 继续下降，|g| 1.95。
- 未完成 / 风险:
  - **attention 才是大头（kernel 时间 40%）**：q/kv 仍是 fp32，走 attn_gym 的非
    tensor-core 路径；`MINIDEEPSEEK_ATTN_BF16=1` 要先 prewarm，本次因为工作区
    被并行会话改到 import 失败而没跑成，需要单独一个窗口；
  - micro-batch 8192 仍受 autotune 的 12.8 GiB 峰值限制，16384 未验证；
  - 宿主 load 77–185 时同一配置能差 30%，所有吞吐数字都要标注负载。

## 2026-11-11（迁移到 2×4090D：跨卡数续训的两个 bug）

- **commit**: `47eed99` fix: 单卡存档的 optimizer 状态在多卡续训时找不到
- **commit**: `2f02131` fix: 续训按全局 token 偏移回放数据，换卡数不再多跳数据

- 做了什么:
  - `_load_single_optimizer` 接受 `torch.optim` 的原始整数键格式：按 `param_groups`
    展平顺序映射回参数名，再走原来的 DTensor 分片，不再假定键是参数全名；
  - trainer 记录并回放 `data_tokens`（全局 token），不再按 micro-batch 数回放；旧存档
    回退到 `data_batches × 存档 config 的全局 batch_tokens`。
- 实现了什么:
  - 1 卡 → 2 卡续训能真正接上：Muon 动量不再丢，也不再抛
    `parameter group that doesn't match the size of optimizer's group`；
  - 数据偏移不再随卡数翻倍：旧逻辑在 2 卡上按 32480 个 micro-batch 回放会跳过
    5.06 亿 token（2K 阶段总量才 4.5 亿），新逻辑落在同一全局位置（1.62 万 ×
    16384），最多多走一个 micro-batch。
- 验证: `pytest tests/` **287 passed, 2 skipped**；`ruff check src/ tests/ scripts/` 通过；
  新增 3 个用例：整数键 optimizer 恢复、新格式按 token 回放、旧格式回退。
- 背景: 145M run 在单卡 4090 上跑到 step 1015（2K 阶段 1717 步），迁到 2×4090D
  续训时暴露这两个问题；换机步骤见 `doc/server_runbook.md` §7。
- 未完成 / 风险:
  - 2 卡（FSDP2 分片）的实测吞吐还没记录；跨 NUMA、无 NVLink，`NCCL_P2P_DISABLE=1`
    必须开；
  - 回放假定 `data.batch_warmup_ratio == 0`（现有配置都是 0），warmup 打开时最多
    多走一个 micro-batch。

## 2026-11-11（2×4090D 迁移：并行、口径、日志）

- **commit**: `bd22c8c` perf: 145M 多卡并行改 ddp（实测 2.16×）
- **commit**: `52e0684` feat: 多卡日志按全局口径统计，续训沿用同一 run 目录与日志文件
- **commit**: `e30ef7c` chore: 日志只留训练主线，框架噪声分流到 .err

- 做了什么:
  - `pretrain_145m.yaml` 的 `distributed.mode` 由 `fsdp2` 改 `ddp`（依据 D81）；
  - metrics 的 `tokens` / `tokens_seen` 求和、loss 与 router/swiglu 按 token 加权、
    `mem_*` 取各 rank 最大（`Trainer._globalize`，新增 `dist.all_reduce_max`）；
  - `runs/current.json` 记录在跑的 run，只有"同 flavor + 盘上有 checkpoint"才续写同一个
    `runs/<id>/`，否则新建；`loss_ema` 与 `tokens_seen` 写进 checkpoint extra 并在
    resume 时恢复；
  - 日志改名 `logs/<config>_<campaign stamp>.log`（stamp 存 `logs/.<config>.stamp`），
    stderr 分流到 `<log>.err`，`init_process_group` 传 `device_id`，非 rank0 静音。
- 实现了什么:
  - 2×4090D：19.12 → **8.86 s/step**，13,708 → **29,580 tok/s**（2.16× 单卡），
    峰值显存 14.5 → **12.8 GiB**/卡，3B tokens ETA 2.5 d → **1.2 d**；
  - 一份训练从 step 1015（单卡）续到 1200（双卡）再续，metrics 与日志始终是同一份；
    日志只剩 `== start` / `[trainer]` / `step` 三类行。
- 验证: `pytest tests/` **293 passed / 2 skipped**，ruff 通过；线上 step 1200 续训
  loss 3.6331、29.2k tok/s，val 在 1199 跑出 2.4560 / ppl 6.3。
- 未完成 / 风险:
  - 迁移途中一度同时跑了两个训练进程（旧进程未退），已清掉并按 step_001200 续训；
    `step_001200` 的 meta 校验过（step/data_tokens 都对）;
  - 老机 hn01 已关机（用户操作），代码以 GitHub 为准；单卡那份
    `checkpoints/curriculum_145m_3b/stage_00_seq2048` 已随新机的续训接上；
  - `mem_allocated_mb` 是"取数时刻"的分配量（优化器步之后），会偏低，看 `mem_peak_mb`。

## 2026-11-11（后训练：vLLM 核查、KV-cache rollout、CUDA graph、SFT 启动）

- **commit**: `e17bcce` feat: KV-cache 增量 rollout 引擎（prefill/decode/fork/批量）
- **commit**: `2d2453c` feat: KV-cache rollout 上 CUDA graph，并接好后训练训练侧
- **做了什么**:
  - 核查 vLLM 0.33 的 DeepSeek-V4 实现：仅 FlashMLA（SM90，`head_dim_v=512`）
    与 FlashInfer（SM100/120）；4090 是 SM89、模型 `head_dim=64`，无法服务
    （D82）。保留 `rollout_backend` seam，vLLM 仍是可选 extra；
  - 自研 slot 寻址 KV cache：`model/cache.py`；compressor/indexer 增量状态机
    按 official `DeepSeek-V4-Flash/inference/model.py` 的 `kv_state` 语义移植；
    attention/model/wrapper 支持 prefill 写 cache、单 token decode；
  - `rl/cached_rollout.py`：批量 prefill、group fork、活跃 slot 压缩解码；
  - `rl/graph_decode.py`：分桶 CUDA graph，padding 行走 scratch slot；
  - decode 热路径去 host sync（MoE 静态 dense 分支、静态 top-k、vectorized
    comp/indexer 写入），`einsum`→`bmm`；
  - GRPO old logprob 改由 actor 重算（D83）；`MiniDeepSeekModel.logits_at`
    只算 response 位置的 LM head；
  - 145M 的 SFT/GRPO/OPD 配置切到当前 base（step_001015）、`rollout_backend=cached`、
    `distributed.mode=ddp`；新增 `doc/posttraining_plan.md`。
- **实现了什么**:
  - fp32 CPU：cache prefill/decode 与全量前向逐 token 对齐（max|Δ|≈2e-6），
    覆盖 SWA/CSA/HCA、不同长度、多 slot、group fork；
  - 4090 bf16：decode **eager 49 ms/step → CUDA graph 5.4 ms/step**（batch 64，
    ≈12k tok/s）；rollout 16 prompt×group4×32 token ≈1.4 s（含 prefill）；
  - 2 进程 CPU DDP + cached 的 GRPO/OPD smoke 通过；
  - SFT 数据：general 50k 条 / 33.1M token（28.4M target），code 30k / 17.6M，
    tool 8k / 5.1M；SFT general 145M 正在跑（6–7k tok/s，12.7 GiB peak）。
- **验证**:
  - `pytest -q` → **298 passed / 2 skipped**；`ruff check` 变更文件通过；
  - `scripts/test_rl_distributed.py --dist-mode ddp --rollout-backend cached`
    GRPO/OPD 均 PASS；
  - 服务器 `bench_decode.py`：prefill 16×~561 token 4.1 s，graph decode
    5.39 ms/step；`bench_graph.py` 1.4 s / 2048 response token（含 prefill）。
- **未完成 / 风险**:
  - prefill 仍是 torch reference（~81 ms/561 token），未走 chunked prefill；
  - tool 多轮 rollout 仍回退全量重算，未接 cached continuation；
  - FSDP2 + cached 未接（只支持 DDP）；
  - general GRPO 的 1.8B reward model 尚未在 4090 上实测加载/显存；
  - SFT/GRPO/OPD 的完整三领域实跑尚未完成（SFT general 进行中）。

## 2026-11-11（README / 视觉资产）

- **commit**: `5d41f5f` docs: 重做 README，加入专属 logo、banner 与架构图

- 做了什么:
  - 重写 README：hero banner、徽章、模型矩阵、V4 架构图、Mermaid 流水线、
    当前进展/实测、目录、边界、文档地图；
  - 新增 `assets/logo.svg|png`、`assets/banner.svg|png`、`assets/architecture.svg|png`；
  - 快速开始拆成 CPU 两分钟验证、数据准备、145M 训练、监控/评估/导出四步，
    所有相对链接指向仓库内实际存在的文档/脚本。

- 实现了什么:
  - README 从“能力清单式”改为按使用路径组织：先跑通 CPU 门禁，再按
    数据 → 预训练 → SFT/GRPO/OPD 的顺序复现；
  - 视觉资产全部自包含（无外部图片依赖），SVG 在 GitHub 上直接渲染；
  - 更新了过期状态：305 tests、145M 2K stage step 1200/1717、SFT/GRPO general
    结果、vLLM/EP 等边界。

- 验证:
  - `pytest -q` → **305 passed, 2 skipped**；
  - README 的 28 个相对路径全部存在；锚点与外部链接另行检查；
  - Mermaid 流水线用 mermaid 11 在本地渲染通过；
  - 用 GitHub Markdown API 渲染预览，检查了 hero/表格/代码块/图片路径。

- 未完成 / 风险:
  - `scripts/run_pipeline.sh` 与 3B 分域 pack 布局仍未完全对齐，README 已注明
    当前以 `train_curriculum.sh` + 单阶段脚本为准；
  - 训练状态数据来自 `doc/PROCESS.md` / `doc/posttraining_plan.md`（2026-11-11），
    后续 run 推进后需要同步更新；
  - logo/banner 的 PNG 是本地 Chrome 渲染产物，不参与运行时。

## 2026-11-11（后训练实跑：三领域 SFT、general GRPO 50 步、三教师 OPD smoke）

- **commit**: `e25cffe` fix: GRPO 全配置 OOM（indexer 分块 + actor mini-batch + checkpointing）
- **commit**: `9696da4` feat: tool 多轮 rollout 接 KV-cache（observation 逐 token 追加）
- **做了什么**:
  - 完成 SFT general/code/tool 三个 specialist（120/90/80 step，单卡 4090）；
  - GRPO general 全量 50 step（16 prompt × group 6 × 512 token，1.8B GRM）；
  - GRPO code/tool 各 2 step smoke；tool 走新的 `CachedToolRolloutEngine`；
  - 三教师 OPD 2 step smoke：general=GRPO final、code/tool=SFT final；
  - 清理磁盘：删除 hn01 上 26 GB 的 pretrain tokenization、旧中间
    checkpoint、/tmp 下的 pytest 临时目录（11 GB）与 smoke checkpoint。
- **实现了什么**:
  - SFT finals 落盘：`checkpoints/sft_{general,code,tool}_145m/final`；
  - GRPO general final：`checkpoints/grpo_general_145m/final`，reward −1.8~−2.2，
    grad ~0.6，49k response token/step；
  - OPD 三 teacher KL 分量非零：general 0.063 / code 0.197 / tool 0.173，
    total 0.144；student/teacher 不同权重时 KL 路径正确；
  - disk：数据盘 98%→29%（35 GB free），系统盘 72%→10%（27 GB free）。
- **验证**:
  - `pytest -q` → **300 passed / 2 skipped**；
  - 1-step 全配置 GRPO OOM 前 → 修复后 18.6 s/step、9.0 GB 峰值；
  - `_feed` 追加路径 fp32 与全量前向对齐 <1e-4（tests/test_cached_tool.py）。
- **未完成 / 风险**:
  - 三教师 OPD 只跑了 2 step smoke，未做 50 step 收敛；
  - code/tool GRPO 只跑 smoke（reward 方差/正确率还很低，需更长 SFT 或调 reward）；
  - tool 多轮 continuation 只在 CPU 单测/2-step smoke 验证，未做长轨迹 A/B；
  - prefill 仍是 torch reference，未做 chunked prefill；
  - FSDP2 + cached 未接，多卡只支持 DDP。

## 2026-11-11（TitanRL review + DAPO 数值保护）

- **做了什么**: 阅读 `torchtitan/rl/`（TitanRL）的 vLLM model wrapper、batch invariance、
  DAPO/GRPO loss、advantage estimator；确认其 V4 支持仍是 "Support coming soon"，
  不能解决 mini-145M 的 vLLM 后端问题。
- **实现了什么**:
  - 从 `trainer/losses/dapo.py` 借鉴两条数值保护并落到 `rl/grpo.py`：
    `log_ratio` clamp ±10、可选 asymmetric clip（`rl.clip_low/clip_high`，默认仍
    对称 0.2）；没有复制整文件；
  - `AdvantageEstimator` 的 std-normalized GRPO 与本项目 `group_advantages` 一致，
    无需改动。
- **验证**: `pytest tests/test_grpo.py` 17 passed；ruff 通过。
- **未完成 / 风险**: 其异步 controller/Monarch/TorchStore 与 vLLM generator 未采用；
  V4 的 vLLM adapter 仍需上游支持或自写。

## 2026-11-11（OPD 20-step 实跑 + tool/OPD 合并 prefill）

- **commit**: `92cee73` perf: tool/OPD 混合 batch 的 rollout 合并 prefill
- **做了什么**:
  - OPD 20 step（9135 条三域 prompt pool：6000 general/tool + 3135 code；
    student=sft_general final，teacher=GRPO general / SFT code / SFT tool）；
  - 发现 OPD 混合 batch 里 `CachedToolRolloutEngine` 对每条 plain record
    单独 `super().rollout`，每条都重置 cache + 重捕 CUDA graph，prefill
    占 ~40 s；改成 plain/tool 各一次批量首轮，只有 observation continuation
    按 slot 处理，step time 降到 ~17 s。
- **实现了什么**:
  - OPD 20 step final：KL total mean 0.065（first 0.066 / last 0.070，
    min 0.055 / max 0.085），general 0.041 / code 0.052 / tool 0.100；
    grad mean 1.16 / max 3.8；8192 response token/step；
  - `checkpoints/opd_three_teacher_145m/final` 落盘；中间 step_10/20 已清理；
  - 磁盘：数据盘 32 GB free、系统盘 27 GB free。
- **验证**: `tests/test_cached_tool.py` + `tests/test_opd.py` 8 passed；
  OPD 20 step 完整跑完并保存 final。
- **未完成 / 风险**:
  - 20 step 的 KL 没有单调下降，不能说明收敛；lr/步数需要扫；
  - code/tool teacher 还是 SFT（GRPO 只有 smoke），上限受 teacher 限制；
  - prefill 仍是 reference；下一步 chunked prefill。

## 2026-11-11（三领域 GRPO 10 step + 全 GRPO teacher OPD 20 step）

- **做了什么**:
  - 修正 `configs/grpo/{code,tool}.yaml` 的 `train.resume`（旧路径
    `checkpoints/sft_code/final` → `sft_code_145m/final`）；
  - code/tool GRPO 各 10 step（cached rollout；tool 走 CachedToolRolloutEngine）；
  - 用三个 GRPO final 作 teacher，再跑 20 step OPD；
  - 清理 `/tmp` smoke checkpoint 与 OPD 中间 step。
- **实现了什么**:
  - code GRPO：reward mean −0.085（std 0.040），grad 0.27，24.6k resp tok/step；
  - tool GRPO：reward mean −2.235（std 0.827），grad 0.46，49.2k resp tok/step；
  - OPD（全 GRPO teacher）：KL total mean 0.065（first 0.066 / last 0.067），
    general 0.040 / code 0.054 / tool 0.098，grad mean 0.87；
    `checkpoints/opd_three_grpo_145m/final`；
  - 磁盘：数据盘 31 GB free、系统盘 25 GB free。
- **验证**: 三个 GRPO/OPD 进程均正常退出并保存 final；OPD 20 step 完整。
  服务器 `pytest -q` → **308 passed**（首轮有 2 个 GPU host 顺序相关
  flaky，单独跑均通过，第二轮全通过）。
- **未完成 / 风险**:
  - 10 step GRPO 不足以提升 reward，code/tool specialist 仍弱；
  - OPD 20 step KL 未收敛，需 200+ step 与 lr 扫描；
  - 多卡仍只有 CPU 2-proc DDP smoke，没有真实 2-GPU 实跑。

## 2026-11-11（145M 训练 plateau 诊断：混料/衰减/验证集三处改动）

- 背景: 2×4090D 上 145M curriculum 跑到 step 2200+，用户报"loss 很久不降"。
  诊断结论（见 D85–D87 + 下方事故记录）：
  - 日志 headline `loss = main + 0.3·mtp`，比 LM loss 高 ~0.85 nats，且被
    域级整步切换放大成 1.47↔3.85 的锯齿；
  - general 的 train main loss 从 step ~1300 起压在 2.55，约 235M token
    无进展；val main 1.996 @999 → 1.690 @1599 之后回退到 1.874 @2199；
  - router 在 general 上的 min-share 14.2% → 10.4% → 9.3%（step
    2000/2200/2393），code/tool 上有 0.0% 的专家；没有全局死专家。
- 做了什么:
  - `data.lockstep: true → false`（D85），新增 `train_main_loss_<domain>` /
    `train_loss_<domain>` 分域 EMA，并把它们接进 `_globalize` 的
    token 加权归约（`_STAT_PREFIXES` 加 `train_`）；
  - `schedule.decay_start_ratio: 0.8 → 0.2`（D86），并修 `resume()` 让
    配置 LR 覆盖 checkpoint 的 `base_lr`；
  - 控制台以 `lm`（main head）为主线、附分域列与 `VAL lm`（D85/D87）；
  - 新 val 集 general 40M / code 16M / tool 6M，`val_steps` 10 → 64（D87）。
- 事故（必须记录）: `make_val_split.py --trim-source` 第一版用
  `save_flat_dataset(source, ...)` 原地重写，`np.save` 的 `open("wb")`
  在拷贝任何 token 之前就把正在被 memmap 读的
  `pretrain_general_8k.input_ids.npy` 截断 → **738M token（2.95 GB）的
  general 8K 区间丢失**，源 JSONL 早已按磁盘预算删除，无法重建。
  当时的临时处置：`pretrain_general_8k` 改由 `pretrain_general_4k` 的 token 流
  （corpus [370M,1723M)）在 8192 打包替代（hardlink + 重写 meta）。
  **已于同日恢复**（见下节"8K 数据恢复"），现在是真实区间。已在 `pretrain_general_8k.json` 的
  `substituted_from` / `note` 字段里写明。修复后的 `trim_source` 改为
  “暂存目录 + `os.replace`”，并加了 `np.array(..., copy=True)`；
  `tests/test_make_val_split.py` 覆盖了"源文件不被改写"和"保留区逐字节不变"。
- 验证:
  - 本地 `pytest -q` → **313 passed, 2 skipped**；`ruff check` 通过；
  - 服务器 `pytest -q` → 282 passed / 15 failed，15 个全是**既有**问题
    （服务器代码比本地落后 23 个 commit：`test_trainer` 的 stub 不接受
    `entropy=` 形参、`test_independence` 被 `._*` AppleDouble 文件
    干扰、其余是 GRPO/consistency/generation 的新修复）；清掉 7 个
    `._*` 后 `test_independence` 恢复；
  - 18 个数据集逐个可加载（`PackedPretrainDataset` + 首 pack 校验）；
  - 重启日志：`step 2400 | lm 2.2625 (ema 2.2685) | general 2.4153
    code 1.3959 tool 0.9745 | total 3.0088 (mtp 2.4858 x0.3) |
    lr 6.00e-04 x1.000 | |g| 0.331 | 28.89k tok/s`；
    `step 2399 | VAL lm 1.5746 ppl 4.83 | general 2.1476 code 1.3509
    tool 1.1813`。
- 未完成 / 风险:
  - ~~general 8K 数据是替代品~~ → 已从本地工作区的 tokenization 备份完整恢复
    （见下节），8K 阶段用的是真实区间 [1723M, 2461M)，不再重复 4K 语料；
  - LR 峰值是否仍偏高未做 A/B；看点：step 2400→2900 的 general 分域
    loss 斜率与 router min-share；
  - `balance_loss_coeff` 仍是 1e-4，code/tool 上的极端路由大概率还会存在。

### 2026-11-11 补充：口径统一的跨 checkpoint 复测（更正 plateau 结论）

- 同一评测代码（CPU fp32、同一批 3 个 micro-batch）在 step 2200 / 2393 上复测：
  general train 2.3034→2.2844、val 2.1660→2.1315、code train 1.4350→1.3428、
  code val 1.5289→1.4351。**三个域都在下降**，不是停滞。详见 D88。
- 训练侧数字比 fp32 实测高 ~0.24 nats（bf16 autocast + `full_packs` 重切 pack），
  所以控制台 train loss 不能与 val loss 直接比；这本身是既有的口径问题，
  已在 D88 记录，暂不改（改 `full_packs` 会动数据管线）。
- 报告侧已改为以 LM(main) loss 为主线并新增分域训练/验证面板（commit de65ccb），
  服务器上用 `scripts/plot_training.py` 重渲染确认三面板各 3 条曲线。

### 2026-11-11 复测：改造后 400 步，新验证集三域单调下降

| val step | 总体 LM | general | code | tool |
|---|---:|---:|---:|---:|
| 2399（改造后第一个点） | 1.5746 | 2.1476 | 1.3509 | 1.1813 |
| 2599 | 1.4803 | 2.0729 | 1.2837 | 1.0468 |
| 2799 | 1.4457 | 2.0019 | 1.2666 | 1.0320 |
| 2999 | **1.4158** | **1.9662** | **1.2475** | **0.9994** |

对比改造前的 200 步：1999→2199 是 1.8672→1.8738（general +0.004、tool +0.085）。
改造后 4 个验证点（2399/2599/2799/2999）三域全部单调下降，general
每 200 步稳定 −0.04~−0.07（累计 2.1476→1.9662，已跌破 2.0），是改造前
（−0.019/193 步）的约 3–4 倍。控制台分域 EMA：general 2.52→2.45、
code 1.40→1.27、tool 1.03→0.93–1.00。**结论：混料是这轮的主要增益，
峰值 LR 不需要动**（grad_norm 0.3、从未触发 grad_clip）。

`best/` 选择也正常：step 2600 的 `best/meta.json` 记 `best_val_loss=1.98897`。

### 2026-11-11 风险：8K 阶段的冷编译尚未预热

`logs/prewarm.log` 显示预热跑到 `compiling float32 H=12 D=64 T=8192 S=256
K=256 win=128` 就断了 —— 4K 阶段（494s）是热的，**8K 那一个变体没编完**。
按 `doc/decisions.md` D74 的说明，它是冷编译的离群点（约 1 小时）。
step 8011 进 8K 阶段时会原地停顿约 1 小时（不报错，只是没输出）。
现在不能补烤：单个 4090D 显存只剩 ~10 GB，而预热本身要 11.4 GiB 峰值。
计划：下次因为别的原因停训练时，先跑
`scripts/prewarm_attn.py --curriculum configs/pretrain/generated/curriculum_145m_g2.yaml`
再启动；否则就接受 step 8011 处的一次性停顿。

### 2026-11-11 8K 数据恢复（完全修复）

本地工作区 `miniDeepSeek/data/tokenization/` 里存着**同一份**打包产物
（Nov 9 18:24–18:28 构建，服务器是 10:02–10:28；两侧所有共享边车的字节数与
md5 一致，见下），因此不需要重新下载语料就能整体还原。

**先证明两侧是同一份数据**（字节级）：
- `pretrain_general_2k.offsets.npy` / `pretrain_general_4k.offsets.npy` /
  `pretrain_code_2k.offsets.npy` / `pretrain_general_2k.target_mask.npy`
  的 md5 两侧逐项相同（offsets 对文档边界极敏感，撞不了）；
- general_2k / general_4k 的 token 流前 1 MiB md5 相同
  （`22228f76…` / `804aba6d…`）；
- `pretrain_general_8k.json` 的 packs/tokens/target_tokens/documents_kept/
  skip_tokens/max_tokens 六项与损坏前服务器上读到的一致。

**传输与替换**：`gzip -1` 压缩后走 SSH（2.62× 压缩率、链路 6.9 MB/s），
5 个文件 6.64 GB 原始体积、约 3 分钟传完，逐个过 md5。替换时**先 `rm` 掉
hardlink 再 `mv`**，绝不用 `cat >` 写穿 —— 否则会连带毁掉 general_4k
（当时两者 inode 相同、links=2）。替换后 general_4k 的 inode/links(2→1)/
前缀 md5 均未变。

**完整性判据**（关键）：`val_general_8k_v2` 是从**被毁前**的原文件切出来的。
用恢复后的文件重新切同样的 40M 尾部，四个数组 md5 全部对上：
`input_ids 576fc062…`、`positions 812b6707…`、`target_mask bfc6961f…`、
`offsets 8dc163ab…`。这证明恢复是逐字节精确的，不是"看起来像"。

**收尾**：再用修好的 `make_val_split.py --trim-only` 把 40M 验证尾部从训练
文件里裁掉 → `pretrain_general_8k` = 698,000,234 token / 87,932 packs /
`skip_tokens 1723000000`（真实区间），与 `curriculum_145m_4x4090.yaml`
里写的 8K 阶段 698M general 预算一致。18 个数据集全部通过
`PackedPretrainDataset` 加载校验，磁盘剩 14 GB，`.incoming/` 与暂存目录均已清空。

## 2026-11-11（实时看板扩容 + 分域路由快照）

- 用户要求：指标太少要扩容（含 matrix），专家路由从"总体"改成 general/code/tool/总体 四种。
- trainer：新增 `router_domain_layers`，在 `train.router_every` 的快照步里按
  域累积**整步 16 个 micro-batch** 的每层每专家计数，并额外给一个 `overall`
  桶。原来 `router_layers` 只取最后 1 个 micro-batch，在 `lockstep=false`
  之后那只是"某一步里恰好排在最后的那个域"，做不了分域对比。
- 看板 `scripts/live_dashboard.py`：
  - `KEEP` 从 ~40 个字段扩到 ~60（分域 train/val、lr_factor、balance/indexer/MTP、
    router 的 min/entropy/std、swiglu、mem_allocated 等）；
  - 新增 `best_val`（历史最低 `val_main_loss` 及其 step）；
  - `router_domain_layers` 不进 `rows`（每个约 1 KB），单独保留最新 2 份；
  - `_slim` 只透传标量形式的 `router_domain_*`，避免把嵌套快照塞进每一行。
- 看板 `scripts/live_dashboard.html`：
  - loss 面板改为 LM(main) 为主线 + total 虚线 + val LM 点；
  - 新增 "training loss by domain"（`train_main_loss_*`）、"metrics matrix"
    （30 个指标的当前值 + 窗口内变化，绿=改善）、"auxiliary losses"
    （balance/indexer/MTP，对数轴）；
  - lr 面板加 lr_factor（右轴），throughput 面板加 step time（右轴），
    router 面板加 min-share 与归一化熵；
  - 专家负载改成四视图（overall / general / code / tool）带标签页切换，
    每层每专家一根柱 + 1/E 公平线 + 低于半个公平值标红。
- 验证：
  - `pytest -q` **316 passed / 2 skipped**；`ruff` 通过；`node --check` 校验页面 JS；
  - 端到端起本地 bridge（`--local-port 8799`，用 `MINIDEEPSEEK_SSH_PASSWORD`）
    拉 `/data`：546 行、latest step 5320、`router_domain_layers` 在 step 5300
    带 code/general/overall/tool、`best_val` 1.2654@5199、`error` 为空；
  - 重启训练后 step 5300 的 L2 计数：general `[41482, 22416, 27205, 49833,
    39486, 48954]`（均衡），code `[1202, 6889, 5428, 26, 2839, 0]`、
    tool `[2589, 4815, 4738, 7, 4235, 0]`（e3/e5 基本为 0）——分域视图直接
    把"总体看起来均衡、按域其实极端"这件事暴露出来了。
- 未完成 / 风险：`balance_loss_coeff` 仍是 1e-4，code/tool 的 0% 专家还在；
  改它需要单独一次实验（见 D88 末尾的判据）。

### 2026-11-11 补：看板"双击即开" + 分域快照跨窗口累积

- 用户问"能不能直接打开 HTML"。**不能**：浏览器不能开 SSH，必须有本地进程
  把远端 `metrics.jsonl` 转成 HTTP。所以把那一步压到最小：
  - `live_dashboard.py` 的每个参数都读环境变量
    （`MINIDEEPSEEK_SSH_HOST/PORT/REMOTE_ROOT/STAGE/LOCAL_PORT`），
    `--stage` 默认 curriculum 根目录（可跨 stage 跟着跑）；
  - 新增 `--open/--no-open`，在 tty 上默认自动开浏览器；
  - 新增 `scripts/live.command`（macOS 双击）：先试密钥，失败则**只问一次**
    密码并缓存到 `~/.config/minideepseek/ssh_pass`（700 目录 + 600 文件，
    仓库外，不会被提交），之后零交互；随后 exec 看板并自动开页；
  - 页面在 bridge 不通时直接显示"双击 scripts/live.command"，
    而不是只说 offline。
- 分域快照改成**跨整个 `train.router_every` 窗口累积**（原来只累一个 step）。
  原因：`tool` 只占 3%，一个 step 的 16 个 micro-batch 平均只有 ~0.5 个，
  约 60% 的快照里根本没有 tool，四视图会时有时无。
- 验证：`pytest -q` **317 passed / 2 skipped**，ruff 通过，`node --check`
  校验页面 JS；重启后 step 5350/5400 的快照都同时含 code/general/overall/tool；
  本地用 `scripts/live.command`（只设 `MINIDEEPSEEK_SSH_PASSWORD`）拉起，
  `/data` 返回 550 行、step 5360、`error` 为空。
- 顺带记录一个未采用的优化：`scripts/launch_curriculum.sh`（多卡启动器）
  没有导出 `MINIDEEPSEEK_ROUTING_GPU=1`，而 `train_curriculum.sh`（单卡）有。
  按 D72 的实测这会省掉每 micro-batch 21 次 D2H 同步（约 15–20% 吞吐）。
  改成多卡路径需要一次重启做 A/B，本次没动。

### 2026-11-11 补：按用户要求撤掉双击启动器

- 用户明确要求"改回之前的 Python 文件版本"，已撤掉 `scripts/live.command`、
  `live_dashboard.py` 的环境变量默认值与 `--open/--no-open`，恢复成
  `--host` 必填的纯命令行脚本。
- **保留**的两个改动（与本条无关，且是修 bug）：
  - 分域路由快照跨 `router_every` 窗口累积（否则 tool 视图 60% 的快照为空）；
  - `--stage` 默认值指向 curriculum 根目录（旧默认 `stage_00_seq2048` 已经跑完，
    照抄会显示一个死掉的 stage）。
- 顺带把"为什么必须有本地进程"测实了：从外网探 `hb01-ssh.gpuhome.cc` 的
  8080/8888/3000/8765，**TCP 都能连上但 HTTP 一律空响应**（NAT 网关假接受），
  也就是说训练机没有任何可达的 HTTP 口，页面不可能自己去取数据。
- 验证：`pytest -q` 317 passed / 2 skipped；`node --check` 页面 JS；
  纯命令行方式起 bridge（`--local-port 8801`）拉 `/data`：559 行、step 5440、
  四视图 code/general/overall/tool 齐全、`error` 为空。

### 2026-11-11 补：更正——code/tool 的 0% 专家是领域分化（D89）

- 用"富集比"复测：general 每层最强特化只有 1.02–1.23（无专属专家，它 82% 的流量
  本身就是 pooled 分布）；code 2.20–2.83；tool 1.65–2.06。正向特化真实存在。
- pooled 分布 8 层 min/mean 0.86–0.97 → 没有全局死专家，"0%" 只是 general 侧专家的反面。
- JS 散度 2000→3000 步上升后平台化（code 0.118→0.135→0.141→0.135，
  tool 0.069→0.092→0.088→0.082），不是持续塌缩。
- **撤回"提高 balance_loss_coeff"的建议**：那会压制这里想要的分化，与项目目标相反。
- 分化轴是 "general vs code+tool" 而非三个独立簇（L2 的 code 专家 e1/e2 同时是
  tool 的专家）。tool 仅在 L5 的 e0 上显出独立偏好（1.56）。

## 2026-11-11（后训练代码审计：SFT / GRPO / OPD / rollout）

- **commit**: `e6b3c13` docs: 后训练代码审计（SFT/GRPO/OPD/rollout/eval）

- **做了什么**:
  - 审计 SFT、GRPO/CISPO、多教师 OPD、cached/tool rollout、reward、eval、pipeline；
    对照 verl/minimind 参考实现，并在 CPU debug 模型上写临时脚本复现关键怀疑。
  - 只读检查 hb01 的 pretrain job；未改预训练配置/代码。
  - 同步：本地 `main` 推到 GitHub `bf6a670`；hn01 `c946f5b → bf6a670`
    fast-forward；hb01 未动（等当前预训练结束）。

- **实现了什么**:
  - `doc/posttraining_audit_2026-11-11.md`：P0/P1/P2/P3 清单、行号、证据、
    最小修复建议；P0 是多卡 SFT 的梯度缩放（reduce 后按本 rank target 数缩放，
    rank 间权重一步后分叉；2 进程复现 79/106 参数差 2.1e-5）。
  - P1 覆盖：tool continuation 的 train-mode counter 污染、context 越界、
    open_thinking 推断错误、max_turns 少一轮；reward clip 饱和把好组 advantage
    清零；tool reward unfinished 仍给 gt、alignment 用 len(gt)；eval 旧路径、
    reward dispatch、eval_code tests 丢失；SFT 数据前缀切片与 0.41/0.48/1.39
    epoch；域混料按 input token；pipeline 断点；HCA 长上下文 decode/forward 不一致。
  - 同步了本地与 GitHub、hn01 的代码；hb01 的未提交 pretrain 改动经去注释
    对比与本地 `main` 语义一致。

- **验证**:
  - 本地与 hn01：`pytest tests/test_{grpo,opd,cache,cached_tool,tool_agent}.py -q`
    → **34 passed**；DDP GRPO/OPD smoke（`scripts/test_rl_distributed.py`）PASS。
  - 临时复现：`/tmp/ddp_scale_probe.py`（2 进程 18 vs 4 target → 79/106 参数分叉；
    18 vs 18 → 0）、`/tmp/tool_probe2.py`（counter 0→369、context IndexError）、
    reward clip 饱和、eval_code tests 丢失。

- **未完成 / 风险**:
  - 本次只审计，没有改 `src/`/`configs/`/`tests/`；修复顺序见审计文档 §7。
  - §3 的 P0 修复在共享 `trainer.py`，需等预训练 checkpoint 落盘并确认后再改。
  - 没有真实多卡 GPU 后训练验证；HCA 长上下文、CUDA-graph counter 只在 CPU 复现。
  - “RPO”未在仓库中找到实现（只有 GRPO/CISPO + OPD）；需确认是否指 DPO/RPO。

### 2026-11-11 补：路由按 10 步与其它指标同频记录 + 面板时间滑块

- 用户要求路由历史不要单独一步一记，而是和普通指标同频（每 10 步）。
  `train.router_every: 50 -> 10`，与 `log_every` 对齐；`router_domain_layers`
  每个窗口约 3 KB，跑满 11,444 步约 3.4 MB。
- **合并了两个累加器**：`_accumulate_domain_routing` 现在对每个 micro-batch
  都跑（原来只在 `collect_metrics` 的那一个），同时喂
  - 每域每层每专家计数（`router_domain_layers`，面板四视图 + 趋势）
  - 每域浓度标量（`router_domain_*_share_max/min/js_vs_pooled`）
  两路输出同源，不会再出现"标量和明细对不上"。`overall` 桶从 pooled 计算里
  排除（它是各域之和，喂回去会重复计数）。
- 面板新增**时间滑块**：拖动滑块选中历史快照，下面按域的四视图柱状图随之
  变化；`▶` 可按 160 ms/帧回放整段历史，「最新」跳回末端。滑块被拖动后
  新的轮询不再把它拽回末端。
- 实测窗口统计（10 步 = 159 个 micro-batch）：general 126–139、code 17–28、
  tool 1–9。也就是 **tool 在 10 步粒度下单窗口只有 1–9 个 micro-batch，
  噪声比 general/code 大得多**；约 2% 的窗口完全没有 tool micro-batch，
  面板对这种情况显式提示而不是画空条。这是提高时间分辨率的直接代价；
  需要更稳的 tool 曲线就用更大的 `router_every`。
- 验证：本地 `pytest -q` 318 passed / 2 skipped、ruff 通过、`node --check`
  页面 JS；服务器 13 个失败全为既有（RL/consistency，服务器代码落后 23 个
  commit）；重启后快照 cadence 为 10 步（5660→5760），四域齐全；
  bridge 的 `/data` 里 `router_series` 19 个快照、8 层、四域、无长度错位、
  `error` 为空。
