# 配置与实现偏离记录

> 规则见 `AGENTS.md` §4：与原计划不一致的地方必须写清原方案、实际方案、依据、影响。
> 本文件只记录**已实施**的偏离；待定项在 roadmap §12。

## 2026-11-04 模型与训练

### D1. 模型实现整体重构进本项目，不再运行时依赖 torchtitan

- **原方案**：roadmap 阶段 1 要求把 `torchtitan/models/deepseek_v4/` 复制到
  `src/minideepseek/model/`；早期实现只复制了 flavor 构建器，其余运行时 import。
- **实际**：把 V4 的**模型数学与参数布局**重构为普通 `nn.Module`
  （`config/nn/rope/mhc/moe/compressor/attention/model`），去掉 torchtitan 的
  config / SPMD / sharding / remat 框架；state dict key 与形状对齐上游。
  运行时不再 import torchtitan，`_compat.py` 与复制的 `flavors.py` 已删除。
- **依据**：项目要独立开源（`AGENTS.md` §3.0）；上游目录依赖约 1.4 万行框架代码，
  整目录复制会引入大量无关并行/配置逻辑，而我们需要的是 V4 数学本身。
- **影响**：
  - debug 模型与上游逐层对齐，fp32 相对误差 ~1e-4（来源是上游把 routed expert
    输入 cast 到 bf16；见 `scripts/verify_reference.py`）；
  - 并行/分片改由本项目 `train/distributed.py` 的 FSDP2 路径负责，不再依赖上游
    sharding；TP/CP/EP 的上游实现未复制，属于后续工作；
  - upstream 升级时按 `doc/sources.md` 的映射重新核对。

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
- **实际**：mini flavor `compress_ratios = (0,0)+(4,32)*k`；
  `index_topk` 默认取 `min(2048, max(seq_len // 4, 64))`（4K→1024、8K→2048，
  即 dense-over-compressed），训练末期按 `IndexTopKSchedule` 收紧到 64。
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

### D11. 默认 torch rollout；vLLM/verl 作为可选集成

- **原方案**：roadmap 阶段 4/5 使用 verl + vLLM。
- **实际**：
  - `rl/rollout.py` 的 torch full-recompute 仍是**默认与一致性基准**；
  - `rl/backends/vllm.py` 提供可选 `VLLMGenerator`（CUDA-only）；
    actor log-prob 始终由训练模型计算，vLLM 只负责采样；
  - `rl/verl.py` 提供 verl 数据 schema、`compute_score` reward entrypoint、
    GRPO/OPD Hydra config 与 `scripts/export_verl.py`。
- **依据**：CPU 环境装不了 vLLM/verl；而 vLLM 对自定义稀疏注意力没有现成
  paged-attention/kernel 支持，不能假称“已验证”。先把接口、数据、奖励、HF
  权重格式打通，GPU 上再替换执行引擎。
- **影响**：没有 vLLM 吞吐优化；vLLM 权重同步是“重建 engine”的保守实现；
  verl 只在接口层验证，GPU smoke 待做。torch rollout 无 KV cache，仅用于
  正确性验证和小规模 RL。

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

### D15. checkpoint 链路与 HF 导出

- **实际**：训练 checkpoint 用 `torch.save(state_dict)`；另有 safetensors 导出与
  `torch.distributed.checkpoint`（DCP）保存/读取；三者均已验证 logits 位级一致。
  新增 `model/hf/`：
  - `MiniDeepSeekConfig` / `MiniDeepSeekForCausalLM`（`transformers` 可选依赖）；
  - `scripts/export_hf.py` 导出 `config.json` + `model.safetensors` + tokenizer +
    remote-code 文件；
  - export → `from_pretrained` 的 logits 与训练模型**逐位一致**。
- **键映射**：HF 模型内部是 `self.model`，safetensors key 带 `model.` 前缀；
  `model/hf/convert.py` 提供训练 key ↔ HF key 映射，转换脚本会去掉 MTP 头。
- **影响**：verl / vLLM 可以直接按 HF 目录加载；DCP 仍是分布式 checkpoint 的
  交接格式。HF 模型不支持 KV cache（`use_cache=False`），生成走 full recompute。

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

### D19. 预训练 / SFT trainer 自实现，内置 FSDP2/DDP

- **原方案**：roadmap 阶段 2/3 使用 torchtitan Trainer（FSDP2 + EP）。
- **实际**：`train/trainer.py` 是显式 forward/backward/step 的循环；
  `train/distributed.py` 提供 `torchrun` 启动、FSDP2（`fully_shard`）与
  manual DDP、global grad-norm clip、full state gather；
  `data/sft.py` 自己实现 packing 与 response-only mask。
- **依据**：torchtitan Trainer 依赖它自己的 config 注册表、grain dataloader 与
  sharding 框架；本项目要独立开源，模型和 PackedBatch 契约已经稳定，自己实现
  ~200 行分布式胶水更可控。
- **影响**：没有 EP/TP/CP、没有 torchtitan 的 dataloader 与 checkpoint 管理；
  FSDP2 的 expert 参数按普通参数分片（专家复制，不做 all-to-all）。

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

## 2026-11-04 多卡、HF、vLLM/verl

### D23. FSDP2 + manual DDP，Muon 在 shard 上 all-gather 后更新

- **实际**：
  - `DistributedSettings.mode`：`fsdp2`（默认）或 `ddp`；`WORLD_SIZE>1` 自动启用；
  - FSDP2 对每个 transformer block / MTP block 调 `fully_shard`，再包 root；
  - Muon 遇到 DTensor 参数时 `full_tensor()` all-gather，做 NS-5，再用
    `DTensor.from_local` 写回 local shard；AdamW 直接支持 DTensor；
  - `clip_grad_norm` 对 shard 的平方和做 all-reduce，避免“各 rank 本地 norm”
    导致不同 DP 尺寸下 clip 系数不同；
  - expert bias 的 token 计数在更新前 all-reduce。
- **依据**：roadmap 目标环境是多卡 CUDA；单进程 CPU 只能验证逻辑，但 FSDP2 的
  DTensor 路径必须现在写对，否则上 GPU 才发现 clip/Muon 不一致。
- **验证**：CPU/Gloo 两进程，world=1 与 world=2 在相同 seed/batch 下保存的
  `model.pt` sha256 完全一致（FSDP2 与 DDP 各自内部一致；两种模式之间最大差
  1.5e-8，来自 float32 norm 计算顺序）。
- **影响**：Muon 每步 all-gather 全部矩阵，60M 可接受，更大模型需要换成
  distributed NS 或把 Muon 参数设为 replicated；EP 仍未实现。

### D24. vLLM 只做可选采样后端，权重同步用重建

- **实际**：`rl/backends/vllm.py` 的 `VLLMGenerator` 通过 HF 目录构建
  `vllm.LLM`，请求 `logprobs=0` 取采样 token 的 logprob；actor logprob 仍来自
  训练模型；`update_weights` 重新导出 HF，若 vLLM 没有 `reload_weights` 就重建
  engine。
- **依据**：vLLM 没有 miniDeepSeek 的 paged-attention kernel；不能在无 GPU 的
  环境验证自定义模型注册。先保证“采样接口 + HF 权重格式 + 一致性门禁”正确，
  不伪造吞吐结论。
- **影响**：现在不能声称 vLLM 可用；需要在 CUDA 上确认目标 vLLM 版本能否通过
  `auto_map` 加载。不能加载时用 `rollout_backend: torch` 或 verl HF rollout。

### D25. verl 适配走 HF rollout + FSDP actor

- **实际**：`rl/verl.py` 生成 verl `main_ppo` 的 Hydra overrides，数据转成
  `prompt/data_source/reward_model/extra_info`，`compute_score` 复用项目 reward；
  `scripts/export_verl.py` 生成 `train.jsonl`/parquet + `run.sh`。
  默认 `rollout.name=hf`（verl `HFRollout` 调 HF `generate`），避免依赖未实现的
  vLLM kernel；`rollout.name=vllm` 只在目标 vLLM 支持时使用。
- **OPD**：verl 的 `loss_mode` 是 k1/k3/forward_kl_topk，没有 V4 的 full-vocab
  reverse KL；`build_verl_opd_config` 只作为模板，**canonical OPD 仍是
  `rl/opd.py`**，它直接在 6,400 词表上算全词表 reverse KL。
- **影响**：verl 未安装、未实跑；接口按 verl 当前源码编写，GPU smoke 待做。

### D26. HF 导出不含 MTP 头，且把 tied embedding 展开

- **实际**：`export_hf` 丢弃 `mtp_layers.*`，HF 模型 `n_mtp_layers=0`；
  导出 config 强制 `tie_word_embeddings=False`，把共享的 `lm_head.weight`
  写成独立张量。
- **依据**：MTP 是训练辅助头；推理/rollout 只用主头。safetensors 拒绝共享
  storage，vLLM/verl 的加载器也偏好显式 `lm_head.weight`。
- **影响**：HF logits 与训练模型主头逐位一致；HF 目录比训练 checkpoint 多一份
  embedding 参数（mini-60M +4.1M）；MTP 实验仍用训练 checkpoint。

### D27. DP 上 buffer 与 indexer aux loss 的同步

- **实际**：`apply_parallelism` 在 FSDP2/DDP 包装后从 rank0 broadcast 所有
  buffer（hash 表 `tid2eid`、`expert_bias_E`、RoPE cache 等）；indexer aux
  loss 按 local token 数归一化后再除以 world size，抵消 FSDP 梯度平均。
- **依据**：buffer 不参与梯度归约；如果不 broadcast，rank 间 hash 路由可能
  不一致。aux loss 若只除 local tokens，FSDP 平均后会被放大 world 倍。
- **影响**：单进程不变；多卡下与全局 per-token loss 一致（已验证 world=1/2
  权重 sha256 一致）。

### D28. lm-eval 与 HTTP world judge 作为可选路径

- **实际**：
  - `eval/lm_eval.py` 通过 `lm_eval.simple_evaluate` + `HFLM` 加载 HF 目录，
    覆盖 MMLU/ARC/HellaSwag 等通用基准；`lm-eval` 是可选 extra。
  - `rl/judge.py` 提供 OpenAI-compatible HTTP judge（`RewardSettings.world_judge=http`），
    默认仍是 `default_world_judge` 启发式。
- **依据**：CPU 环境不能跑 lm-eval 全量、也没有 judge API；代码与守卫先就位，
  评测结论等 GPU/API 环境。
- **影响**：默认 math/code/world 自写评估可用；lm-eval 与 HTTP judge 未在 CI
  实跑，不能把占位分数当质量结论。
