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

### D29. batch-token 线性 warmup 与 val_path

- **原方案**：roadmap 阶段 2 要求前 10% steps 将 batch 从较小值线性升到最大。
- **实际**：`DataSettings.batch_tokens_final` + `batch_warmup_ratio`；
  `data.pretrain.iter_batches` / `data.sft.iter_batches` 用
  `token_schedule` 按 micro-batch 序号线性升温；`TrainSettings.val_path`
  现在会构建独立验证集并传给 `Trainer`。
- **依据**：不需要改模型/优化器，属于数据迭代器；预热步数按
  `steps * grad_accum` 计。
- **影响**：`full_6b`/`pilot_0.6b`/`dense_control_1b` 从 16K 升到 64K
  tokens/micro-batch（最终 global batch 仍 0.5M）；debug 配置不受影响。
  sequence-length curriculum（2K→4K→8K）仍是数据分阶段 recipe，尚未做成
  单次训练的自动调度。

### D30. Muon / AdamW / mHC 参数分组按官方 V4 报告修正

- **官方依据**：V4 report §2.4 Algorithm 1：
  - 8 步 hybrid NS（系数 3.4445/-4.7750/2.0315）+ 2 步稳定
    （系数 2/-1.5/0.5）；
  - 更新缩放 `O = O' * sqrt(max(n, m)) * γ`，γ=0.18；
  - AdamW 用于 embedding、prediction head、RMSNorm、mHC 的 static biases/gating；
  - AdamW eps=1e-20、betas=(0.9,0.95)、wd=0.1。
- **修正**：
  - `train/optimizer.py` 改为 10 步 hybrid NS，默认 γ=0.18；
  - `OptimizerSettings` 暴露 `ns_steps=10`、`ns_late_steps=2`、
    `muon_update_scale=0.18`、`eps=1e-20`；
  - `hc_base`/`hc_scale` 走 AdamW，`hc_fn` 走 Muon（之前误把 `hc_fn`
    也归入 AdamW）。
- **影响**：Muon 更新量与官方一致；旧的“NS-5 + sqrt(rows/cols)”配置不再使用。

### D31. 补齐 roadmap 的 AR/rollback/curriculum/数据/指标缺口

- **Anticipatory Routing**：`train/trainer.py` 增加一步预取队列；在 step t 用当前
  gate 预计算 t+1 的 routing 索引并缓存，t+1 直接 replay；首步/Δ>1 仍回退
  probe。probe 仍同步执行，没有做官方 §3.1 的通信 overlap。
- **rollback**：`train/spike.py` 在 FSDP2 下用 full-state gather/broadcast 快照，
  spike 决策由 rank0 broadcast；不恢复 dataloader 位置。
- **sequence-length curriculum**：`train/curriculum.py` 分阶段 2K→4K→8K
  runner，分别 prepare 数据、累计 step、resume 上一阶段 checkpoint。
- **数据**：`WeightedBatchMixer`（`data.sources`）、`ParagraphDeduper`、
  语言/script、vendored、binary、math formula/answer 过滤。
- **指标**：`domain_routing_metrics`，GRPO 可选 `log_domain_routing`。
- **权重同步**：vLLM generator 版本计数 + 可选 actor/rollout logprob 门禁。
- **未做**：EP/CP/激活重计算、teacher hidden-state cache、GRM（按用户要求
  世界知识专家先不做）。

### D32. rollout/eval/teacher 统一使用 wrapper autocast

- **实际**：`MiniDeepSeekModel` 新增 `autocast_dtype`；CUDA 上 `forward` 自动
  `torch.autocast(bfloat16)`，训练/GRPO/OPD/eval/teacher 都走同一 wrapper。
  权重仍是 fp32 master（teacher 未做 bf16 权重 cast/offload）。
- **依据**：roadmap §5 要求 bf16；之前只有 Trainer 有 autocast，rollout/actor
  精度不一致。
- **影响**：CUDA 上训练与采样算术精度一致；CPU 测试保持 fp32。teacher 的
  显存节省仍需要权重 bf16 或 hidden-state cache（未做）。

### D33. GRPO/OPD 支持 FSDP2，不再只有单进程

- **实际**：
  - `RLConfig.distributed` 复用 `DistributedSettings`；
  - `build_grpo_trainer` / `build_opd_trainer` 在 world>1 时 `apply_parallelism`
    包装 actor/student，prompt/record 按 rank 分片；
  - 梯度裁剪使用 FSDP2-aware `clip_grad_norm`；save/resume 走 gather + 手工
    state load；rank0 才写 checkpoint/日志；
  - `torchrun` 启动：`NPROC_PER_NODE>1 ./scripts/train_grpo.sh/opd.sh`。
- **依据**：roadmap 的阶段 4/5 最终是多卡；verl 是生产路径，但本项目也需要
  自己的小规模多卡正确性路径。
- **验证**：CPU/Gloo 2 进程 `scripts/test_rl_distributed.py --method grpo|opd`
  各跑 1 step 并保存；`tests/test_rl_distributed.py` world=1 FSDP2 覆盖。
- **限制**：vLLM backend 与 FSDP2 RL 未接线（显式拒绝）；teacher 仍每 rank
  复制；各 rank 局部 loss 等权平均，不等 token 数加权。

### D34. hash 层不保留 balance bias；哈希表改为确定性生成

- **原方案**：所有 MoE 层都注册 `expert_bias_E`，`update_expert_biases` 统一
  更新；`tid2eid` 每次 `init_weights()` 用全局 RNG 重建。
- **新方案**：
  - hash 层不注册 `expert_bias_E`（属性为 `None`）。`update_expert_biases`
    只更新 learned 层，但**所有层**的 `tokens_per_expert_E` 照常清零；
  - `tid2eid` 由 `(HASH_SEED=0, layer_id)` 经局部 `torch.Generator` 生成，
    同参数必得同表，跨进程/rank/设备一致，且不消耗全局 RNG；
  - `init_weights()` 不再重建 `tid2eid`：它是数据，不是初始化参数。
- **依据**：`_select_experts` 的 hash 分支不读 bias，更新它是无人读取的死状态；
  `init_weights()` 在 load 之后被调用会静默换表——loss 正常、行为全变。
- **验证**：`tests/test_trainer.py::test_hash_layers_have_no_expert_bias`、
  `test_expert_bias_update`（计数清零 + 只更新 learned 层）、
  `tests/test_routing.py::test_hash_table_is_deterministic`、
  `test_init_weights_does_not_rebuild_hash_table`。
- **影响**：
  - 旧 checkpoint 里 hash 层的 `expert_bias_E` 成为多余键，`strict=True` 加载
    会报 unexpected key，需重新生成或手动删除这些键；`tid2eid` 仍在
    checkpoint 中，不受影响；
  - 新建模型的哈希表与改前不同（旧的走全局 RNG）；已有 checkpoint 加载后
    行为不变；
  - 建表不再消耗全局 RNG，同 seed 下其余参数的初始化序列与改前不同。

## 2026-11-05 审计修复带来的配置/行为修正

### D35. `batch_tokens` 与 `prompts_per_step` 明确为 global 语义

- **原方案**：配置和文档按 global（例如 `full_6b` 注释 0.5M tokens/step）描述，
  但 loader/RL 实际按每 rank 使用；8 卡时 batch 与 token 预算被放大 8 倍。
- **新方案**：`train/run` 按 `shard_world_size` 平分 `batch_tokens` /
  `batch_tokens_final`；GRPO/OPD 按 DP size 平分 `prompts_per_step`。world=1
  行为不变。
- **影响**：所有既有配置无需逐卡改数；若 `batch_tokens < world*seq_len`，
  每 rank 至少保留 1 个 pack，global 会略高于配置（可训练性下限）。

### D36. 阶段切换的 resume 语义

- **新增** `TrainSettings.resume_data` / `resume_optimizer` / `resume_step`：
  - 同一次训练中断后恢复：三者默认 `true`；
  - pretrain → SFT、SFT/pretrain → GRPO/OPD：三者均为 `false`（新数据、新优化器、新 step）；
  - curriculum 的每个 stage：`resume_data=false`，但保留 optimizer 与累计 step。
- **依据**：直接在 pretrain checkpoint 上 resume SFT 会把 pretrain 的 step
  (约 12000) 带进 SFT，导致 `while step < 500` 直接跳过训练；同时会继承
  pretrain 的 `base_lr` 和已消费 batch 数。
- **影响**：从 base 初始化 specialist/RL 时必须使用这三项 false；同 run
  续训保持默认 true。

### D37. 运行期 index top-k 写入 checkpoint metadata

- **实际**：Trainer/GRPO/OPD 保存时在 `extra.index_topk` 记录当前运行期
  top-k；eval、HF export、teacher 加载优先使用该值，再回退静态配置。
- **依据**：`index_topk` 不在 state_dict 中；SFT 末期调度到 64，而静态
  config 可能仍是 1024，直接按 config 重建会静默改变模型行为。
- **影响**：旧 checkpoint 没有该字段时仍回退 config；配置中显式写 top-k
  用于没有运行时元数据的场景。

### D38. MTP 头在 RL/eval/teacher 加载时显式忽略

- **实际**：`load_checkpoint` 增加 `ignore_prefixes`，RL resume、teacher、
  eval 显式传 `("mtp_layers.",)`；主权重仍 strict。
- **依据**：pretrain/SFT 保留 MTP，RL/eval 为 `n_mtp_layers=0`；strict load
  会因 MTP key 报错。
- **影响**：只忽略该前缀，其他 key 不匹配仍报错；RL resume 不再加载旧优化器
  （D22），避免覆盖 RL 学习率。

### D39. 数据文件改为 `.npy` sidecar + mmap

- **原方案**：`np.savez_compressed` + `np.load(mmap_mode="r")`。
- **实际**：compressed/uncompressed npz 成员都不能真正 mmap；6B 数据会整块
  解压进内存（约 54GB/rank）。改为 `<stem>.{input_ids,positions,target_mask,offsets}.npy`
  + JSON meta；旧 `.npz` 仍可读取。
- **影响**：新建数据集的逻辑路径仍是 `*.npz`，但实际文件是 sidecar；外部
  直接 `np.load(path)` 的脚本需要改用 `load_flat_dataset`。

### D40. MiniMind tool 数据解析与透传

- **实际**：`normalize_conversation` 解析字符串形式的 `tools`/`tool_calls`；
  SFT 从 system message 提取 tools；GRPO/eval rollout 透传 `resolved_tools()`。
- **依据**：MiniMind `sft_t2t_mini.jsonl` 的 `tool_calls` 是 JSON 字符串，
  旧代码会被 Jinja 当字符串逐字符迭代而崩溃；system tools 也会丢失。
- **影响**：非法 tool JSON 在 SFT build 中跳过；合法记录保留完整 tool schema。

### D41. Rollout/eval 打开 think 块；format bonus 检查闭合标签

- **实际**：rollout/eval 的 generation prompt 使用 `open_thinking=True`，
  让模型在 `<think>` 内生成推理；`math_reward` 的 format bonus 检查响应里
  是否出现 `</think>`。
- **依据**：SFT 数据包含 think；原 generation prompt 预先闭合空 think，
  模型无法在该位置生成推理，`format_bonus` 恒为 0。
- **影响**：GRPO/eval 的 prompt 与响应格式变化；格式奖励现在可达。

### D42. 优化器分组按官方 recipe 修正

- **实际**：`router.gate.weight` 归 Muon；`compressor.ape` /
  `indexer.compressor.ape` 归 AdamW。
- **依据**：官方 torchtitan recipe 的 Muon 正则包含 router gate，且注释明确
  “additive gate-logit biases (ape) stay with AdamW”。
- **影响**：与旧训练的优化器轨迹不同；参数集合不变。

### D43. Anticipatory Routing snapshot 时序

- **实际**：`observe_step` 在 `optimizer.step()` 之前记录 θ_t；自动 AR 的
  持续步数同时受 `loss_spike_ar_steps` 与 `AnticipatoryRoutingSettings.max_steps` 限制。
- **依据**：原先记录更新后的 θ_{t+1} 却按 `step-delta` 读取，delta=1 的
  fallback 实际没有 stale；max_steps 是未读取配置。
- **影响**：AR fallback 现在使用真正的历史 gate；显式 `enabled=True` 的 AR
  仍持续开启（max_steps 只限制 spike 触发的自动 AR）。

### D44. D27 的 indexer aux loss 除以 world_size 已撤销

- **修正**：删除 `MiniDeepSeekModel.aux_loss_world_size` 及 `compute_loss`
  中的额外除法。
- **依据**：indexer loss 已按 local tokens 归一化；FSDP/DDP 的梯度平均本身
  就是 global per-token，再除 world 会让 aux 项缩小 W 倍。
- **验证**：2 进程 `build_trainer`，world=1/2 权重 digest 一致，indexer loss
  日志一致（4.0274e-6）。

### D45. 训练/验证 loader 使用可重建 factory

- **实际**：`load_dataset` 返回 factory；Trainer 每轮重建 iterator，验证集
  每次从同一 factory 重置；resume 通过 checkpoint `extra.data_batches` 跳过
  已消费 micro-batch。
- **依据**：原 generator 耗尽会抛 StopIteration；val iterator 耗尽后返回 NaN，
  且每次验证的是不同子集。

## 2026-11-06 三个新模型档位与配置外置

### D46. 用 mini_220m_a80m / mini_145m_a60m / mini_60m_dense_d768 取代旧三档

- **原方案**：`mini_60m`（d=640, L=12, 6+1 experts）、`mini_110m`、
  `mini_200m`（d=1024, L=16）；dense 对照指“去掉 compressor/indexer 的 dense
  attention”。
- **实际**：采用 `doc/Parameter_Setting.md` 的三个设计，统一 d=768、L=7、
  MTP depth 1、tied embeddings、SWA+CSA+HCA：
  - `mini_220m_a80m`：216,652,538（216.65M）总参 / 82.17M 激活；
  - `mini_145m_a60m`：145,841,946（145.84M）总参 / 60.91M 激活；
  - `mini_60m_dense_d768`：60,876,570（60.88M）总参 / 60.88M 激活。
- **依据**：用户提供的参数设计；实测 `sum(p.numel())` 与设计值一致（误差
  在舍入范围内）。固定宽度/深度使 220m、145m、dense 三档只改变 FFN/MoE
  与稀疏度，便于消融。
- **影响**：`mini_220m_a80m` 成为默认训练模型；旧 `mini_60m/110m/200m`
  文件删除，`dense_control_1b.yaml` 改为 dense-FFN 对照。旧的“dense attention
  对照”（compress_ratios 全 0）不再单独保留；需要时可在新 YAML 上改
  `compress_ratios`。

### D47. 架构数值外置到 configs/model/*.yaml

- **原方案**：`model/config.py` 里的 `_MINI_PRESETS` / `_DEBUG_PRESET`
  硬编码，训练配置只写 `model.flavor`。
- **实际**：`configs/model/*.yaml` 是完整的 `ModelSettings`；训练配置可用
  `model_file:` 指向任意架构文件，`model:` 下的字段覆盖本次运行。CLI
  增加 `--model-config` 覆盖。未写 `model_file` 时，`model.flavor` 也会解析
  到同名 YAML（debug 配置因此无需改）。
- **依据**：以后训练不同参数设计只应改数据，不应改 Python 分支；同时保留
  flavor 名称用于 checkpoint metadata / HF export / 一致性重载。
- **影响**：新增架构只需新增 YAML；`build_model_config(flavor)` 仍可用于
  CLI/测试，按文件加载。`ModelSettings` 默认值仍是 debug 规格，便于直接构造
  小模型做单测。

### D48. seq_len 与 max_context 分离

- **原方案**：`build_model_config(..., seq_len=X)` 同时把训练长度和
  RoPE/generation 上限设为 X。
- **实际**：模型 YAML 同时写 `seq_len: 4096` 与 `max_context: 8192`；
  `ModelConfig.max_context_length` 取 `max_context`，训练数据仍按 `seq_len`。
  旧的按 flavor 构建路径在只给 `seq_len` 时仍让 context 跟随它（向后兼容）。
- **依据**：`doc/Parameter_Setting.md` 的三档都要求 4K 训练、8K 最大上下文；
  RoPE cache 按 8K 建，训练/评估可按 4K/8K 不同长度运行。
- **影响**：训练模型生成上限为 8192；HF 导出新增 `max_context` 字段并优先
  从 checkpoint metadata 读取。

### D49. 单 expert 的 dense 控制固定权重 1.0

- **问题**：原 router 对单 expert 仍返回 `softplus(gate).sqrt() * route_scale`，
  这不是“等价 dense FFN”，会给 FFN 输出乘一个可学习的缩放。
- **实际**：`num_experts == 1` 时权重固定为 `route_scale`（模型文件为 1.0），
  top-k 恒为 0；`sequence_balance_loss` 在 `num_experts <= 1` 时返回 0。
- **依据**：`doc/Parameter_Setting.md` 明确“单专家，路由权重恒为 1 / 无需
  负载均衡”。保留 1 个 gate 参数（约 0.6K/层）以复用 MoE 结构，但不再影响
  前向数值。
- **影响**：dense 控制组现在严格等价于 SwiGLU FFN；参数统计中 router 仍有
  少量参数，不影响 60.88M 的舍入结果。

### D50. index top-k 由模型文件派生

- **实际**：模型 YAML 的 `index_topk: null` 表示
  `min(2048, max(seq_len // 4, 64))`；4K→1024、8K→2048。显式整数（debug
  model=16）或 `ModelSettings.index_topk` 覆盖优先。训练末期的
  `index_topk_final` 调度保持不变。
- **依据**：`doc/Parameter_Setting.md` 的 “1024 @4K / 2048 @8K”；避免在
  每个 training config 重复写同一个数，也避免 8K 课程忘记改。
- **影响**：2K 课程阶段派生 512；旧配置里显式 `index_topk: 1024` 已删除，
  8K stage 仍显式写 2048。

### D51. head_dim=256 的显存风险（未在 GPU 验证）

- **问题**：220m 档为 16 heads × 256 dim；当前 `gather_attention` 是
  torch 参考实现，会把 `[T, window+topk, head_dim]` 的 KV gather 实体化。
  4K token 时每层约 2.4GB（bf16），7 层约 17GB；8K 翻倍。多卡切分后可接受，
  单卡大 batch 或 4 卡 64K tokens/micro-batch 很可能 OOM。
- **状态**：CPU 上三个配置的 forward/backward 与 1-step 已通过；**未在多卡
  CUDA 实测**。后续需要 fused gather attention / activation checkpointing /
  按 token 分块重算中的至少一种，才能按 full_6b 的最终 batch 规模训练 220m。
- **依据**：`gather_attention` 当前是 correctness-first 的 torch 实现；
  项目阶段只验证代码逻辑。

### D52. 参数统计 bucket 修复

- **问题**：`parameter_report` 的 `moe_routed/router/shared` 统计了 MTP 层，
  又从 `mtp` 桶里减一次，`mini_145m_a60m` 的 `attention_and_other` 会变成
  负数（-4.94M）。
- **实际**：`moe_*` 只统计主 Transformer 层，`mtp` 统计整个 MTP 栈；
  各 bucket 相加等于 total。`active_estimate` 仍按文档口径把 MTP routed
  experts 也按 `top_k/num_experts` 计入稀疏比例（220m=82.17M、
  145m=60.91M），与设计值一致。
- **影响**：仅报告数值；训练/checkpoint 不受影响。

### D53. 训练监控、可视化与 checkpoint 保留策略

- **实际**：
  - `metrics.jsonl` 新增 `tokens_seen`、`tokens_per_sec`、`step_time_s`、
    `eta_seconds`、`loss_ema`、`mem_allocated/reserved/peak_mb`、
    `mtp_loss_d{depth}`；val 记录新增 `val_ppl`、`val_main_loss`、
    `val_mtp_loss`、`val_indexer_loss`、`val_tokens`。
  - 新增 `src/minideepseek/viz/`（原生 SVG + 静态 HTML 报告）与
    `scripts/plot_training.py`；`TrainSettings.report_every > 0` 时训练中
    自动刷新 `report.html`。
  - `save_best` 默认开（val 改善写 `best/`）；`keep_last_ckpt` 默认 0
    （保留全部）；NaN/Inf 时写 `nan_step_*/`（含 `nan_report.json`）后退出。
- **依据**：参考 `mini_deepseek_v4` 的 benchmark/监控/checkpoint 实践，
  补齐本项目缺失的实验基建；所有默认值不改变训练数学，也不新增依赖
  （可视化用原生 SVG，不引入 matplotlib）。
- **影响**：旧 `metrics.jsonl` 仍可解析（字段只增不减）；开 `val_every`
  且 val 改善时多一个 `best/` 目录；`keep_last_ckpt=0` 时不删除任何
  checkpoint（与旧行为一致）。

### D54. 按 block 的激活重算（`train.activation_checkpointing`）

- **问题**：`gather_attention` 把 `[T, window+top-k, head_dim]` 实体化，4K 时
  145M 档 saved 激活 12.3 GiB，8K 约 25 GiB，24 GB 卡上的 8K 阶段必然 OOM；
  加卡无效（激活不随卡数切分）。
- **实际**：`DeepSeekV4Model.grad_checkpointing` + `_run_block()`，用
  `torch.utils.checkpoint(..., use_reentrant=False)` 包每个 transformer block
  和 MTP block；由 `train.activation_checkpointing`（或课程阶段的同名字段）
  打开，默认关闭。实测 145M@4K saved 激活 12.27 GiB → 0.53 GiB。
- **依据**：README 早已把“激活重计算”列为未实现；这是 24 GB 卡跑 8K 的最小改动。
- **影响**：开启后前后向多约 30% 计算；router 的 `tokens_per_expert_E` 在重算时
  会再统计一次（`update_expert_biases` 只用 `sign(counts.mean()-counts)`，
  对 load-balance 更新无影响，只有日志里的 expert-load 计数翻倍）。

### D55. 数据打包改用 `array` / `bytearray` 缓冲

- **问题**：`pack_documents` 用 Python `list[int]` 累积 token，2B token 语料约需
  70 GB 内存，`data.prepare` 在正式语料上会直接 OOM（此前只有 debug 小数据）。
- **实际**：输入/positions 用 `array("i")`、mask 用 `bytearray`，最后
  `np.frombuffer` 零拷贝转 numpy；每个 token 约 9 B（2B token ≈ 18 GB）。
- **依据**：B9 只修了读取侧的 mmap，写入侧仍是 Python list。
- **影响**：pack 结果逐字节不变（现有 `test_data.py` 全过）；`data.prepare`
  在 64 GB 机器上可处理 2B token，32 GB 机器需用 `--max-tokens` 限流。

### D56. 本地语料进 `data/raw/`，manifest 自包含

- **问题**：`data/manifests/pretrain_6b.yaml` 指向 `../minimind/...` 与多个未下载
  语料，服务器上既拿不到也不符合“项目自包含”的约束。
- **实际**：`minimind` 的 `pretrain_t2t.jsonl`（8.5M 文档、约 2.3B token）与
  `pretrain_t2t_mini.jsonl` 复制到 `data/raw/`；新增
  `data/manifests/pretrain_local.yaml`（1.2B token 预算、单源、1 epoch），
  6B manifest 里的 minimind 路径也改到 `data/raw/`。
- **依据**：当前阶段不下载 6B 混合语料；单源通用语料足够跑通方法与曲线。
- **影响**：配置只引用 `data/...` 相对路径；语料本身被 `.gitignore` 忽略，
  服务器上按 `doc/server_runbook.md` 用 rsync 传。

### D57. 课程阶段内的断点续跑

- **问题**：`run_curriculum` 只把上一阶段的 `final` 传给下一阶段；某个阶段中途
  崩溃后重跑会从该阶段起点重新训练（数小时白跑）。
- **实际**：新增 `latest_stage_checkpoint()`，每个阶段优先续自己的
  `final`/`step_*`，再由 `Trainer.train()` 的 `while self.step < until_step`
  停在累计目标上；已完成的阶段重跑是 no-op。`CurriculumStage` 同时新增
  `activation_checkpointing` 字段，让 8K 阶段单独开启重算。
- **依据**：`until_step` 本来就是累计语义，续跑无需改训练循环。
- **影响**：同一条课程命令可反复执行；阶段切换仍用 `resume_data=False`。

### D58. specialist 路线改为 general / tool-use / code

- **原方案**：world / code / math 三个 specialist。
- **新方案**：general / tool-use / code；math 不再单独做 specialist，world/medical
  不再作为 teacher。code 数据与 GRPO reward 暂缺，先训练 general + tool-use。
- **依据**：用户 2026-11-07 决策；MiniMind 的通用能力、RLAIF、tool/agent 数据
  都已具备，code 需要外部数据与单测 reward，先不硬做。
- **影响**：OPD 先 2-teacher（general + tool）；code 数据补齐后加
  `sft_code` / `grpo_code`，再切 3-teacher。world/medical 代码保留但不在流水线。

### D59. 数据平铺 + MiniMind 直接来源

- **原方案**：`data/raw`、`data/processed/{sft,grpo,opd}`、`data/tokenized`。
- **新方案**：`data/*.jsonl` 平铺；`data/tokenization/` 单独放 packed 中间产物；
  `data/manifests/` 放来源/统计。`data/build.py` 从 `minimind/dataset` 复制/处理：
  - `pretrain_t2t.jsonl`、`sft_general.jsonl` 原样复制；
  - `sft_tool.jsonl` = SFT tool 子集 + 25% general；
  - `grpo_general_*` = `rlaif.jsonl` prompt（丢末尾空 assistant，保留多轮历史）；
  - `grpo_tool_*` = `agent_rl.jsonl` + `agent_rl_math.jsonl` 的 tools+gt 子集；
  - `opd_prompts_*` 从 GRPO train prompt 派生并均衡。
- **依据**：用户要求平铺、上传 ModelScope、服务器一键；`minimind/dataset` 只读。
- **影响**：旧 `data/raw`、`data/tokenized`、`data/processed` 已删除；配置统一
  指向平铺路径与 `data/tokenization/`。

### D60. general/tool reward：MiniMind 设计 + 可验证主信号

- **general**：`internlm2-1_8b-reward` 主分数 + MiniMind 的长度/思考/重复 shaping
  + `reward_clip=3`；`rlaif` 没有 gt，不能只用规则。
- **tool**：MiniMind `train_agent.py` 的 tool 对齐 + GT 2.5× + 未完成 -0.5 +
  标签平衡 + shaping；tool 分支不调用 RM，无工具分支可用 RM。
- **math/code**：保留 exact match / 单测作为主信号，shaping 默认 0，避免改变
  既有 reward 语义；需要时在 YAML 打开。
- **依据**：MiniMind 的 RM 路线解决通用 prompt 的奖励稀疏；我们的 tool 有
  gt/工具结果，保留可验证信号更可靠。

### D61. tool-use 多轮 rollout 与 OPD observation mask

- **实现**：`rl/agent_rollout.py` 移植 MiniMind `train_agent.py` 的
  assistant→tool_call→observation 循环；`RolloutSample.token_mask` 把
  observation 标 0，GRPO loss 与 OPD KL 只算 assistant token。
- **OPD**：`ToolRolloutEngine` 同时服务单轮 general 与多轮 tool prompt；
  `OPDTrainer` 用同一 `response_mask`，teacher 在 observation 上下文上打分。
- **依据**：MiniMind 的多轮 tool RL 是当前数据可复用的最短路径；不引入
  vLLM/tool server 依赖，先保证 CPU/CUDA torch 路径正确。

### D62. tokenization 路径与去重

- **路径**：`data/tokenization/pretrain_{2k,4k,8k}.npz`、`sft_{general,tool}.npz`；
  `.npz` 是逻辑路径，实际是 `.npy` sidecar + `.json` meta。
- **去重**：`data/tokenize.py` 默认关闭内置 MinHash。2.3B 语料上纯 Python
  MinHash 约需数十 GB 内存/数十分钟；正式去重需要独立的 exact/分布式实现。
- **依据**：D45 已把 pretrain 打包缓冲降到 ~9 B/token；SFT builder 也改成
  `array`/`bytearray`，604M token 约 6 GB 而不是 ~50 GB。

### D63. 预训练预算 1.2B → 3B，并把 code/tool 加进混料

- **原方案**：`curriculum_145m_4x4090.yaml` 用 MiniMind `pretrain_t2t.jsonl` 单源，
  1.2B token（2.2B 可用，1 epoch）。
- **新方案**：3B token 七源混料——中文通用 40%、英文教育 15%、中文教育 15%、
  **code 15%**、合成教材 10%、**tool 3%**、math 2%（见 `doc/data_plan_145m.md` §3.2）。
- **依据**：实测现有语料 general 96.1% / code 2.1% / **tool 0%**；三专家路线下
  code 与 tool 必须进预训练打底。145M 上 3B 是甜点，5B 以上边际收益低。
- **影响**：`pretrain_3b.yaml` 待建；`data/tokenization/pretrain_*.npz` 需按新混料
  重新打包（当前 tokenization 只覆盖 MiniMind 单源）；`available_tokens` 需实测替换。

### D64. 三专家 SFT 数据量按"饱和点"而非"按数据集大小"

- **原方案**：`configs/sft/*.yaml` 统一 500 step × 32,768 token = 16.4M token/专家。
  （**更正**：`batch_tokens` 是全局 micro-batch，一个 optimizer step 还要乘
  `grad_accum=8`，实际是 500 × 32,768 × 8 = **131M token/专家**；16.4M 的说法漏乘了累积步数。）
- **新方案**：general 40–60k 条（15–20M token）、code 25–35k（12M）、
  tool 5–10k（4–6M，2–3 epoch），比例约 4:2:1。
- **依据**：general 多样性无界；code 受 145M 容量限制（不是"好学"）；
  tool 是窄格式技能，少而重复优于多而杂（见 `doc/specialist_data_145m.md` §3）。
- **影响**：各专家 `steps` 不再统一；tool 需要多 epoch 采样；
  general 数据须先从 `sft_t2t.jsonl` 过滤掉 code/math/tool 再抽样。

### D63. code specialist 配置与 code reward 首版

- **背景**：三个 specialist 定为 general / tool-use / code；code 数据与
  reward 尚未最终确定，但配置和代码槽位先落地，避免后续再改训练主链。
- **实际**：
  - `configs/{sft,grpo,eval}/code.yaml` 改为 145M 档、curriculum resume；
  - `RewardSettings` 增加 `code_pass_reward`、`code_compile_bonus`、
    `code_syntax_penalty`、`code_format_bonus`，`code_reward` 按这些权重
    计算，默认值与旧行为一致；
  - `data/build.py` 增加 `--code-sft/--code-grpo`，自动生成
    `sft_code.jsonl`、`grpo_code_train/eval.jsonl`、`opd_prompts_code.jsonl`；
  - `configs/pipeline/minimind_3teacher.yaml` 预留 sft_code / grpo_code /
    3-teacher OPD / eval_code。
- **依据**：用户要求先写 reward 设置，数据后补；不希望在补数据时改训练代码。
- **影响**：缺 code 数据时 `minimind_3teacher.yaml` 会在 tokenize 前明确报错；
  当前可运行的仍是 `minimind_145m.yaml`（general + tool）。

### D64. 训练循环对齐 MiniMind 的两处细节

- **thinking ratio**：单轮 rollout 原来固定 `open_thinking=True`；现在
  `TorchRolloutEngine` 按 `rl.thinking_ratio` 随机打开，general 用 0.9、
  tool 用 0.1，和 MiniMind 的 `RLAIFDataset` / `train_agent.py` 一致。
- **loss type**：`rl.loss_type` 支持 `grpo` / `cispo`；general/tool 配置用
  `cispo` + `epsilon_high=5.0`（MiniMind 默认），旧配置默认仍为 `grpo`。
- **未照搬**：MiniMind 的 KL(ref) 项与 PPO 未引入；当前 GRPO 仍是
  correctness-first 的 torch 实现。

### D65. 领域同质 batch：预训练默认开，OPD 默认关

- **原方案**：`data.sources` 用 `WeightedBatchMixer` 按 micro-batch 交错，`trainer` 只记全局
  router 指标，分不出"哪个领域用了哪些专家"。
- **新方案**：
  - `data.lockstep`（默认 `true`）：mixer 的配额按 **optimizer step** 分配，每个 slot 连续产出
    `grad_accum` 个同源 batch；`trainer` 每步记录 `domain_id`（`data.sources` 下标，-1 为单源）。
  - `rl.teacher_lockstep`（默认 `false`）：OPD 每步只从一个 teacher 抽 prompt。
- **依据**：
  - 分化收益来自 batch 内 router 统计一致性，不需要连续多步同域（`doc/batching.md` §4）；
  - OPD 的 teacher-pure step **没有公开证据**，且多 teacher 混 batch 是主流写法，
    因此只提供开关、不默认启用，等 A/B 结果。
- **影响**：多源训练的 batch 顺序语义改变（同域连续 `grad_accum` 个），旧 run 的 batch 序列不可
  直接对比；`metrics.jsonl` 多一列 `domain_id`；单源训练路径不受影响。

### D66. 外部语料下载口径：按 token 预算取 shard，不整包拉取

- **原方案**：`doc/data_plan_145m.md` §3.2/§4 列出了数据源，但没有规定取多少、
  shard 怎么选；上游 corpora 全量是 0.3–5.8 TB，整包下载不现实。
- **实际**（详见 `download/README.md`）：
  - fineweb-edu 取 `sample/10BT` 前 4 shards：`int_score≥4` 只保留 13%，
    约 641M token（计划 450M）；
  - chinese-fineweb-edu 取 v1 `map/00022` 1 shard，约 1.23B token（计划 450M）；
  - codeparrot-clean 取 16/56 shards：按 §3.3 的 `≤200 行 + permissive + 非自动生成`
    保留 33.9%，约 750M token（计划 450M）；
  - cosmopedia-v2 2 shards（902M）、finemath-4plus 2 shards（419M）；
  - code SFT 新增 `newfacade/LeetCodeDataset`（2,641 train / 228 test），
    目标是“LeetCode 风格函数题”，比通用 instruct code 更贴题；
  - TACO 只取 train 4/9 shards，EASY 且 `input_output` 非空 3,940 题，够 1–2k GRPO 池。
- **依据**：
  - 下载带宽实测 3–10 MB/s，按计划的 1.4–3× 余量取 shard 即可，不必全量；
  - 145M 模型的 code 能力上限是“函数级（MBPP/LeetCode Easy 档）”，仓库级长文件
    对预训练没有额外收益，≤200 行过滤后 750M token 已覆盖；
  - 预训练代码用 §3.3 的 `≤200 行`，SFT 用 §1.1 的 `≤60 行 / ≤400 token`，两者不混用。
- **影响**：
  - 数据源少了 xLAM（已 gated，匿名 401），tool 预训练文本改由 ToolACE + glaive +
    本地 `data/sft_tool.jsonl` 的 `tools` 字段拼；
  - codeparrot 许可混杂（permissive 约 55%，GPL 系约 32%），预处理必须按
    `license` 白名单 + `autogenerated=False` 过滤，否则会把 copyleft 代码吃进训练集；
  - `python_code_instructions_18k_alpaca` 与 open-r1 的 verifiable 集没有 license 声明，
    正式训练前需确认，当前只作内部研究用途。
