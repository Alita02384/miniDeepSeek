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
