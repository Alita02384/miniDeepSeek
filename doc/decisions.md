# 配置与实现偏离记录

> 规则见 `AGENTS.md` §4：与原计划不一致的地方必须写清原方案、实际方案、依据、影响。
> 本文件只记录**已实施**的偏离；待定项在 roadmap §12。

## 2026-11-04 模型与训练

### D1. 不整体复制 `deepseek_v4/`，改为运行时依赖 torchtitan

- **原方案**：roadmap 阶段 1 要求把 `torchtitan/models/deepseek_v4/` 复制到
  `src/minideepseek/model/`。
- **实际**：只复制 **flavor 构建器**（`flavors.py`），模型实现类（model / attention /
  compressor / mHC / MoE / MTP / sharding）继续从只读 checkout 导入。
- **依据**：最小闭包原则（`AGENTS.md` §3.3）；该目录依赖 `torchtitan.models.common`
  ~13.6k 行与 `torchtitan.config/distributed`，整体复制会引入大量需要同步维护的代码，
  而我们没有修改这些实现的需求。只复制“需要新增 mini flavor”的构建层。
- **影响**：仓库必须能访问同级 `torchtitan/`（或设置 `MINIDEEPSEEK_TORCHITAN_PATH`）；
  upstream 升级时需重跑模型测试。`doc/sources.md` 记录了复制范围。

### D2. mini 配置的实测参数与目标有偏差

- **原方案**：mini-60M ≈ 63M；mini-110M ≈ 114M；mini-200M ≈ 192M。
- **实际**（`sum(p.numel())`，含 MTP depth 1、tied embeddings）：
  - `mini_60m`: **65,262,922 (65.26M)**，+3.6%
  - `mini_110m`: **120,652,xxx (120.65M)**，+5.8%
  - `mini_200m`: **218,205,xxx (218.21M)**，+13.65%
  - `mini_60m_dense`（去掉 compressor/indexer）：62.52M
  - debugmodel: 10.60M（无 MTP）/ 12.98M（MTP depth 1）
- **依据**：roadmap 阶段 1 的验收门槛是“与设计值偏差 < 15%”；60M 档优先保持
  `d=640 / L=12 / heads=10×64 / moe_inter=256` 这组“同构但小”的结构，不为凑数牺牲
  可对照性。
- **影响**：60M/110M 达标；200M 档偏高 13.65%，仍在门槛内，若后续严格卡 192M 可下调
  `q_lora_rank/o_lora_rank` 或 `moe_inter_dim`，需同步记录。

### D3. HCA 压缩率取 32（官方 128），index top-k 取 64

- **原方案**：官方 `m'=128`；roadmap 推荐 32。
- **实际**：mini flavor `compress_ratios = (0,0)+(4,32)*k`，`index_topk=64`。
- **依据**：官方 128 是为 1M 上下文调的；4K 下只剩 32 个全局表项，过粗。
- **影响**：与官方不同构于“压缩率”一个数值；机制（compressor + indexer + top-k）一致。
  严格对齐官方可另建 `compress_ratios=(0,0)+(4,128)*k` 消融。

### D4. tied embeddings 默认开启

- **原方案**：roadmap §3.4 建议共享 embedding / 输出头，留消融。
- **实际**：mini 全部 `enable_weight_tying=True`；debug 保持上游 False。
- **影响**：60M 档省约 4.1M 参数；若做消融，可在 ModelSettings 里覆盖
  `tie_word_embeddings: false`。

### D5. debug 模型与 6,400 词表不匹配：增加 debug-only 的 vocab remap

- **问题**：V4 debug model 的 `vocab_size=2048`，MiniMind tokenizer 是 6,400；
  直接用真实 tokenizer 会触发 embedding 越界。
- **实际**：`MiniDeepSeekTokenizer(vocab_limit=2048)` / 环境变量
  `MINIDEEPSEEK_VOCAB_LIMIT` 把 token id 取模映射到 2048；debug 配置显式设置
  `vocab_limit: 2048`。**真实训练（mini flavor）绝不能用**。
- **影响**：debug 数据只用于管线验证，token 语义有损；测试中 GRPO/OPD 使用
  2048 词表的 stub tokenizer，数据/词表测试仍用真实 6,400 tokenizer。

### D6. MTP 训练标签在本地 loss 中实现

- **问题**：torchtitan 的 V4 `preprocess_inputs` 没有像 V3 那样为 MTP 生成
  `mtp_labels_{depth}` / `mtp_input_valid_masks`；直接用 `MTPCrossEntropyLoss`
  会缺标签。
- **实际**：`MiniDeepSeekModel.compute_loss` 用 `shift_targets(batch, depth)` 生成
  “同文档内、位置偏移 depth”的标签，主 loss + 0.3/深度加权 MTP loss；MTP 输入由
  torchtitan `mtp_forward` 内部 roll。
- **影响**：MTP 路径可用且测试通过；与官方实现的差别仅在标签掩码的文档边界处理，
  已用 positions 校验（`positions[t+d] == positions[t] + d`）。

### D7. Muon 自实现，不用 torchtitan `DistMuon`

- **原方案**：复用 torchtitan 的 Muon。
- **实际**：`train/optimizer.py` 实现 Muon（NS-5、momentum、decoupled weight decay）
  与 AdamW 混合容器。
- **依据**：`DistMuon` 依赖 FlexShard / FSDP mesh / 分布式 bucket，单进程 CPU
  不可用；自实现 ~80 行，接口可替换。
- **影响**：大规模训练若要换回 `DistMuon`，只需替换 `build_optimizer`，trainer API 不变。

### D8. SwiGLU clamp 在 debug 配置中显式开启

- **实际**：上游 `_debugmodel` 的 `swiglu_limit=None`；本项目 `build_model_config`
  增加 `swiglu_limit` 覆盖，debug 测试传 10.0，mini 配置默认 10.0。
- **影响**：debug 测试能覆盖 `LimitedSwiGLU` 与 clamp 监控；与官方 V4 一致。

### D9. Anticipatory Routing 采用两段式近似，默认关闭

- **原方案**：第 t 步用 θ_t 算特征、θ_{t-Δt} 算 routing，并预取索引。
- **实际**：先用 stale gate 做一次 no-grad forward 记录索引，再用 replay 做训练
  forward；开销约多一次 forward。
- **依据**：roadmap §3.6 的“预取并缓存索引”实现要点；60M spike 风险低，默认关闭。
- **影响**：与论文的严格并行实现有差异（probe 的后续层输入受 stale routing 影响），
  但满足“第 t 步使用的索引 == stale gate 重算结果”的验收，且开关可测。

### D10. sequence-wise balance loss 自实现

- **实际**：`train/losses.py::sequence_balance_loss` 从 router gate 输出与 hard routing
  map 按文档计算 `E/k * Σ f_i P_i`，权重 1e-4；同时保留 loss-free expert bias。
- **依据**：torchtitan V4 配置没有启用 router 的 `AuxLoss`；但 roadmap §3.1 要求保留
  sequence-wise balance loss。
- **影响**：为一个额外 loss 项（默认 1e-4）；若只想用 loss-free bias，设 0。

## 2026-11-04 后训练 / RL

### D11. GRPO / OPD 用原生 torch，不接 verl

- **原方案**：roadmap 阶段 4/5 使用 verl + vLLM。
- **实际**：`rl/rollout.py`（full-recompute torch rollout）、`rl/grpo.py`、
  `rl/opd.py`、`rl/reward.py`；接口按 verl 的语义设计，便于后续替换。
- **依据**：当前验证环境是 macOS CPU，verl/vLLM 依赖 CUDA 栈，无法安装运行；
  本阶段目标是“代码逻辑与一致性”，原生实现可直接测。
- **影响**：没有 vLLM 吞吐优化、没有 verl 的分布式 rollout/权重同步；rollout 无
  KV cache，仅用于正确性验证和小规模 RL。生产路径仍需接 verl（roadmap 阶段 6）。

### D12. OPD 使用 hard teacher key，权重先等权

- **原方案**：`teacher_key` 硬路由，权重待定。
- **实际**：`OPDTrainer` 按样本 `teacher_key` 分组，`teacher_weights` 默认等权；
  full-vocabulary reverse KL（6,400 词表直接算全词表）。
- **影响**：不做 soft 多教师混合（留消融）；KL 分量按 teacher 记录。

### D13. GRPO 配置与 roadmap 对齐，但 MTP 在 RL 中关闭

- **实际**：GRPO/OPD 的 model 配置 `n_mtp_layers: 0`，因为 ratio / KL 定义在主头上；
  group size 8、prompts/step 32–64、temperature 1.0、clip 0.2、KL 0。
- **影响**：RL 不使用 MTP 辅助头；预训练/SFT 仍保留 MTP。

### D14. DSpark / 推测解码不实现

- **原方案**：第一版不用（roadmap §0.2）。
- **实际**：未实现；rollout 只有 torch full-recompute。
- **影响**：无；MTP head 可作为未来的 draft。

### D15. checkpoint 链路

- **实际**：训练 checkpoint 用 `torch.save(state_dict)`；另有 safetensors 导出与
  `torch.distributed.checkpoint`（DCP）保存/读取；三者都已验证 logits 位级一致。
- **偏差**：没有 HF `transformers` 架构类，无法让 verl 直接 `AutoModelForCausalLM`
  加载；DCP 是给未来 torchtitan/verl 的交接格式。
- **影响**：需要时在 `model/checkpoint.py` 增加 HF adapter（state dict 键映射）。

## 2026-11-04 数据与评估

### D16. 当前只接 MiniMind 样本，不外扩数据

- **依据**：roadmap §0.2 / §3.4 明确“先用 minimind tokenizer 与基础数据，扩充后置”。
- **实际**：`scripts/prepare_debug_data.sh` 只取 pretrain 200 条 / SFT 50 条 /
  math GRPO 20 条；`data/manifests/*.yaml` 写好了 6B 配比但外部数据集路径未下载。
- **影响**：所有数据统计与 manifest 是“管线可用”的证据，不是 6B 语料的实际产出。

### D17. world 评估默认 judge 是启发式占位

- **实际**：`reward.py::default_world_judge` 用 token overlap / 长度 / 句子数
  给六维分；真实使用应替换为 API judge 或本地 GRM。
- **影响**：world GRPO/评估的数字不可当作质量结论；只验证接口与权重聚合。

### D18. 评估未接 lm-evaluation-harness

- **实际**：`eval/` 自写 math accuracy / code pass rate / world rubric 聚合。
- **依据**：当前阶段先验证自写评测链路；harness 接入属于阶段 7。
- **影响**：没有 MMLU/ARC/HellaSwag 等通用分数。

## 2026-11-04 补充：训练稳定性、稀疏调度与 checkpoint 初始化

### D19. 预训练 / SFT trainer 自实现，不包 torchtitan Trainer

- **原方案**：roadmap 阶段 2/3 使用 torchtitan Trainer（FSDP2 + EP）与
  `ChatProcessor` 数据路径。
- **实际**：`train/trainer.py` 是单进程、显式 forward/backward/step 的循环；
  `data/sft.py` 自己实现 packing 与 response-only mask。
- **依据**：当前验证目标是在 CPU 上逐项验证机制（MTP、balance loss、路由、
  clamp、checkpoint）；torchtitan Trainer 需要分布式 launcher、它自己的 config
  注册表与 grain dataloader，调试成本高。生产训练可换回 torchtitan Trainer，
  模型与 PackedBatch 契约已对齐。
- **影响**：没有 FSDP2/EP/TP/CP、没有 torchtitan 的 dataloader 与 checkpoint
  管理；多卡路径未验证。

### D20. 4K 主训使用 dense-over-compressed，8K 前收紧到 top-k 64

- **实际**：mini flavor 默认 `index_topk = max(seq_len//4, 64)`（4K 时 1024，
  即全部压缩项）；`TrainSettings.index_topk_final=64` +
  `index_topk_warmup_ratio=0.05` 在最后 5% steps 内线性收紧。`_dense` flavor
  则完全去掉 compressor/indexer。
- **影响**：与 roadmap “先 dense 后 sparse”一致；`configs/model/mini_60m.yaml`
  的 1024 是 4K 配置，8K 时需要另建 8K 数据与 `seq_len=8192` 配置。

### D21. loss spike 检测、rollback 与 AR 自动开启

- **实际**：`train/spike.py` 用 median/MAD 检测；`TrainSettings` 的
  `loss_spike_auto_ar` 在 spike 后强制开启 AR `loss_spike_ar_steps` 步；
  `loss_spike_rollback` + `rollback_every` 可回滚到最近一次内存快照。
- **依据**：roadmap §3.6 的触发机制；60M 默认关闭。
- **影响**：rollback 是内存快照（不是磁盘 checkpoint），不恢复 dataloader 位置；
  默认关闭，只用于长训练/验证机制。

### D22. GRPO/OPD 的初始化 checkpoint

- **实际**：`RLConfig.train.resume` 在 `build_grpo_trainer` / `build_opd_trainer`
  中加载模型权重（不加载 RL 优化器状态）；GRPO 配置指向对应 SFT specialist，
  OPD 配置指向 base/pilot checkpoint。
- **影响**：符合“GRPO 从 SFT 出发、student 从 base 出发”的路线；RL 阶段仍重新
  初始化优化器。
