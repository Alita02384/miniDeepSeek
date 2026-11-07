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
