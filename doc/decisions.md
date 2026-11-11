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
- **2026-11-10 修订**：`index_topk` 的默认公式作废，见 D74（改为常量 64）。

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

### D65. 领域同质 batch 只用于预训练

- **原方案**：`data.sources` 用 `WeightedBatchMixer` 按 micro-batch 交错，`trainer` 只记全局
  router 指标，分不出"哪个领域用了哪些专家"。
- **新方案**：`data.lockstep`（默认 `true`）——mixer 的配额按 **optimizer step** 分配，每个 slot
  连续产出 `grad_accum` 个同源 batch；`trainer` 每步记录 `domain_id`
  （`data.sources` 下标，-1 为单源）。
- **OPD 不改**：曾加过 `rl.teacher_lockstep`（每步只从一个 teacher 抽 prompt），已于
  2026-11-08 回滚。理由：KL 按 teacher 分组已是正确性要求（`_group_by_teacher`），
  而"teacher-pure step"没有公开证据支持更好，且与 OPD"融合多个老师"的目标相悖。
- **依据**：分化收益来自 batch 内 router 统计一致性，不需要连续多步同域
  （`doc/batching.md` §4）；OPD 保持主流的多 teacher 混 batch 写法。
- **影响**：多源预训练训练的 batch 顺序语义改变（同域连续 `grad_accum` 个），旧 run 的
  batch 序列不可直接对比；`metrics.jsonl` 多一列 `domain_id`；单源训练与 OPD 路径不受影响。

### D66. 预训练改为 3B / 262k-per-step，课程三段按 token 预算重算

- **原方案**：`pretrain_145m.yaml` 1.2B token；`curriculum_145m_4x4090.yaml` 1400/6400/7800 步，
  每步 131k token（`batch_tokens 16384` × `grad_accum 8`），单源 `tokenized_path`。
- **新方案**：
  - 每步 262,144 token（`grad_accum 16`，`batch_tokens` 仍是全局 micro-batch 16384，**显存不变**）；
  - 总预算 3B，课程 1717 / 8011 / 11444 步 = 0.45B / 1.65B / 0.9B；
  - 每阶段用 `data.sources`（general 0.82 / code 0.15 / tool 0.03）+ `lockstep: true`，
    三段的 packed 数据 token 区间互不重叠（`--skip-tokens` 递进）。
- **依据**：3B = 20.6 token/总参（Chinchilla 量级）；262k 与「按参数缩放的临界批量」
  （V4-Flash 75.5M × (145M/284B)^0.74 ≈ 276k）一致。
- **影响**：`pretrain_145m.yaml` 不再是 1.2B 单源；旧 checkpoint 的 global step 含义变化
  （每步 token 翻倍）；`curriculum_145m_4x4090.yaml` 的 `until_step` 全部改写。

### D67. 用流式打包替代内存打包

- **问题**：`build_pretrain_dataset` 把整份语料读进内存（2B token ≈ 18 GB），本机 16 GB
  无法构建 3B 语料；且三长度各跑一次会重复 tokenize。
- **方案**：新增 `scripts/pack_domain.py`：逐文档 tokenize + 追加写 `.npy`（回填 header），
  内存 O(seq_len)，输出与 `load_flat_dataset` 完全兼容；多输入按文档轮转。
- **依据**：实测 1.2M tok/s，3B 语料约 40 分钟；同语义（`pack_documents` 的 position/mask 规则）。
- **影响**：`prepare pretrain` 仍保留（小语料/调试用）；正式 3B 打包走新脚本。

### D68. 本机不做 tokenize；源数据精确去重并裁到 3B

- **原方案**：本机用 `pack_domain.py` 把 2K/4K/8K 的 npz 全部打出来（约 27 GB、
  40 分钟），`data/pretrain_general.jsonl` 与 `data/pretrain_t2t.jsonl` 并存。
- **新方案**：
  - 本机只保留**源数据**，打包移到训练服务器（`scripts/build_pretrain_packs.sh`）；
  - `pretrain_t2t.jsonl` 合并进 `pretrain_general.jsonl`；
  - 合并前做**跨源精确去重**（规范化文本 blake2b-128），去重后 4.56B token，
    按目标 2.46B 做哈希均匀裁剪（保留 54%）；
  - code 923M → 450M、tool 89M → 90M，三个文件合计 3.0B。
- **依据**：用户明确不在本机训练；本机 16 GB 内存 + 生成 27 GB 中间产物没有意义。
- **影响**：`data/tokenization/` 只剩 debug 数据；配置里的 packed 路径改由服务器生成；
  `build_pretrain_packs.sh` 的 general 输入改为单一 `data/pretrain_general.jsonl`。

### D67. 8K 阶段的 top-k 收缩从“全程最后 5%”改到“阶段开头 10%”

- **原方案**：`pretrain_145m.yaml` 的 `index_topk_warmup_ratio: 0.05` 按
  `schedule_steps`（全程 11,444 步）计算，实际只在 step 10,872–11,444 收缩；
  而 8K stage 的 `index_topk: 2048` 在 8K 正好是全部 CSA 压缩项（8192/4），
  等于 **dense-over-compressed**。结果：8K 阶段 95% 的步数仍是 dense，
  top-k=64 的稀疏路径只跑了约 5%（~150M token），与 roadmap §3.5 的
  “4K→8K 预热 2–5% tokens，8K 用 sparse top-k=64 跑 15–20%”不符。
- **新方案**：
  - `IndexTopKSchedule` 支持显式窗口 `start_step`/`end_step`（优先于旧的
    end-of-run `warmup_ratio`）；
  - `TrainSettings` 增加 `index_topk_start_step`/`index_topk_end_step`；
  - `CurriculumStage` 增加 `index_topk_final` + `index_topk_warmup_ratio`：窗口
    从**本阶段首步**起算，按阶段步数比例收缩；
  - `curriculum_145m_4x4090.yaml` 的 8K 阶段：前 10%（343 步 / ~90M token）
    从 2048 线性收到 64，其余 3,090 步（~810M token）保持 top-k=64。
- **依据**：roadmap §3.5 的稀疏预热与验证目标；4K 阶段仍保持 dense（1024），
  不做收缩。
- **影响**：8K 阶段约 27% 的全程 token 在 sparse 路径上（原为 5%）；4K 阶段
  行为不变；`IndexTopKSchedule` 的旧 ratio 语义保留，standalone 的
  `pretrain_145m.yaml` 仍按 end-of-run 收缩。

### D69. 断点续训的语义：`final` 只代表“阶段完成”，信号触发优雅停

- **问题**：`final/` 原来是训练循环结束就写，被 SIGTERM/Ctrl-C 打断时进程直接退出、
  什么都不存（只能回退到最近一次 `step_*`）；而 pipeline 用
  `stage_02_seq8192/final/meta.json` 是否存在判断“跑完了”，所以只要中断时写了
  `final`，没跑完的 8K 阶段就会被当成完成、直接进 SFT。
- **实际**：
  - `Trainer.train()` 安装 SIGTERM/SIGINT 处理器（仅主线程），第一次信号只置
    `_stop_requested`：跑完当前 step、写 checkpoint、返回；第二次信号才抛
    `KeyboardInterrupt` 立即中止；
  - 被打断时写 `step_NNNNNN/`，**不写 `final`**；`final` 仍只由“到达
    `until_step`”的完整运行写出；
  - 新增 `train.auto_resume`（默认 false，`pretrain_145m.yaml` 打开）：
    `resume` 未设置时从 `out_dir` 里 step 最大的 `final`/`step_*` 续训；
  - `latest_stage_checkpoint()` / `run_curriculum()` 已优先续阶段自己的
    checkpoint，并在 `summary["interrupted"]` 时停止推进到下一阶段。
- **依据**：pipeline 的“按 final 跳过”与“中断要能续”必须共存，只能让
  `final` 承担“完成”语义。
- **影响**：中断后重跑同一条命令即可；硬杀（OOM/掉电）最多回退 `ckpt_every`
  步（当前 200 步 ≈ 5200 万 token）；`step`/LR/index top-k/MTP 权重都按全局
  step 续，数据从该阶段 shard 起点重排（`resume_data: false`）。

### D67. 8K 阶段每步 token 翻倍的问题与卡数自适应配置

- **原方案**：`curriculum_145m_4x4090.yaml` 三个 stage 共用 base 的 `grad_accum: 16`，
  按“262,144 token/step”估算阶段步数（2K 1717 + 4K 6294 + 8K 3433 = 11,444 步）。
- **问题**：数据加载器每个 rank 至少取一个 pack（`size = max(1, per_rank_tokens // seq_len)`）。
  8K stage 配 `batch_tokens: 16384`、4 卡时每 rank 只有 4096 token，小于一个 8K pack，
  被 `max(1, ...)` 抬成 **8192 token/rank**，实际全局 micro-batch 是 32,768，
  ×16 = **524,288 token/step**。8K 阶段会吃掉 1.8B 而不是 0.9B，总预算变成 ~3.9B。
- **修复**：
  - `CurriculumStage` 增加 `grad_accum` 字段（`train/curriculum.py`），按 stage 覆盖累积步数；
  - 8K stage 用 `grad_accum: 8`（4 卡），保持 262,144 token/step、11,444 总步数；
  - 新增 `scripts/autoconfig_145m.py`：按 `torch.cuda.device_count()` 计算每个 stage 的
    `batch_tokens = gpus × packs_per_rank × seq_len` 与 `grad_accum = 262144 // micro`，
    生成 `configs/pretrain/generated/*_g{N}.yaml`；卡数不是 1/2/4/8/16 时直接报错。
- **依据**：用户要求“卡数变化时 262k 不能变”。梯度累积按 target token 数加权
  （commit `fade758`），所以 grad_accum 从 16 变 8 不改变优化语义。
- **影响**：8K 阶段显存不变（每 rank 仍是一个 8K pack）；2K/4K 阶段的 `grad_accum`
  随卡数变成 64/N，总步数与 LR schedule 不变。若继续用旧配置，8K 阶段会多消耗 0.9B token。

### D68. attention 快路径：可选的 attn_gym fused kernel

- **原方案**：`gather_attention` 为纯 torch 参考实现（docstring 标注“可由 fused kernel 替换”）。
- **实际**：
  - 新增 optional extra `kernels = ["attn_gym>=0.0.17"]`（BSD-3-Clause，PyPI）；
  - `gather_attention` 在 CUDA 上优先走 `attn_gym` 的 Triton fused（前向+反向），
    失败或 CPU 时自动回退 torch 参考；
  - `MINIDEEPSEEK_ATTN_BACKEND=torch|auto|attn_gym` 可强制切换，便于 A/B。
- **依据**：torchtitan 的 V4 参考实现同样调用 `attn_gym.sparse.gather_attn`；
  DeepSeek 官方的 TileKernels / FlashMLA / DeepGEMM 都要求 SM90/SM100，
  4090（SM89）不可用（TileKernels README 明确 SM90/SM100 + CUDA 13.1）。
- **实测**（4090，4K，`index_topk=64`，bf16，fwd+bwd 训练口径）：
  - torch：fwd 144 ms / bwd 480 ms → **6,571 tok/s**
  - attn_gym：fwd 196 ms / bwd 188 ms → **10,674 tok/s（1.62x）**，峰值显存 8.0→6.7 GiB
  - `benchmark.py`（含 optimizer）：4,490 → 5,894 tok/s
  - CUDA 等价测试：输出/LSE/Q/KV/sink 梯度一致，稀疏 KV 梯度在原子累加容差内
- **影响**：attn_gym 的 Triton 前向在 4090/D=64 上效率很低（~0.14 TFLOPS），
  是新的前向瓶颈；SDPA + 带状 mask 实测 0.61 ms/层（4K），说明还有约一个数量级
  的空间，但需要重写 attention 才能同时保留 sink 与 LSE 语义。

### D69. tied embedding 的初始化修正（首步 loss 757 → 9.2）

- **现象**：145M 训练首步 `main loss ≈ 752`、`mtp ≈ 104`、`|g| ≈ 29.7`。
- **定位**：torchtitan 的 V4 是 **untied**：embedding `normal_(std=1.0)`、lm_head 用
  `trunc_normal_(std=dim^-0.5)`。本项目 145M 配置开了 `tie_word_embeddings: true`，
  共享矩阵继承了 embedding 的 std=1.0，导致输出 logits 量级 ~750、CE ≈ 757。
- **修复**：`DeepSeekV4Model.init_weights` 在 tied 时用输出投影的初始化
  （`trunc_normal_(std=dim^-0.5)`）；untied 仍按上游（embedding std=1.0 + head std^-0.5）。
- **影响**：首步 `main loss` 757 → **9.25**、mtp 104 → 9.23，梯度量级回到正常；
  参数量不变（仍 tied）；与上游的唯一差异是 tied 时共享矩阵的初始分布，已在源码注释说明。

### D70. 训练吞吐三处优化：micro-batch 8192 / attention 边界 bf16 / MoE 排序 dispatch

- **背景**：4090 实测 4K fwd+bwd 10.5k tok/s（MFU ~4%），低于 `hardware_budget.md`
  的 15–20% 预算；单步 kernel 数上万，launch/同步占比高。
- **micro-batch 4096→8192**：24GB 卡实测 4K×2 = **11.6 GiB**，4K×4 = 16384 token
  在 20.4 GiB 处 OOM，故 8192 是安全上限（8K 阶段本来就是 8192/rank）。
  `benchmark.py`（含 optimizer）4K×1 6,220 → 4K×2 **9,304 tok/s（+50%）**；
  纯 fwd+bwd 口径 +22%（trainer 里 optimizer 已被 32 步梯度累积摊薄）。
  `scripts/autoconfig_145m.py` 的 `--per-rank-tokens` 默认改为 8192，三阶段
  grad_accum 均为 32，每步 token 数仍恒为 262,144。
- **attention 边界 cast bf16**：RoPE/RMSNorm 会把 q/kv 升到 fp32，attn_gym 的
  fp32 路径不用 tensor core（同一 kernel 20 ms vs bf16 0.27 ms）。只在 kernel
  边界 cast，softmax/LSE 仍在 kernel 内 fp32，残差流与优化器状态保持原精度。
  模型内 A/B：364 → 327 ms（+10%）。新增 fp32 输入下的 fused-vs-reference 测试。
- **MoE 排序 dispatch**：一次 `argsort` + `bincount` + 单次 `index_add`，替代
  6 次 mask/index_select/index_add 以及每个专家一次 `.any()` 的 GPU→CPU 同步。
  数学等价，路由与权重不变；新增按 token 显式循环的等价性测试（fp32/bf16）。
- **验证**：本地 `pytest -q` 242 passed；4090 上的 CUDA 等价测试与 A/B 见
  `doc/PROCESS.md` 2026-11-10。
- **未做**：trainer 每个 micro-batch 的 8 次 `float()` 同步、mHC Sinkhorn 融合、
  MoE grouped GEMM（6 个专家仍各一次 GEMM，只是不再有 mask 开销）。

### D71. 训练循环去同步 + mHC Sinkhorn 编译（D70 的续）

- **背景**：D70 之后 4K×2 benchmark 到 11.9k tok/s，但训练端仍受每 micro-batch
  大量 host 同步拖累。profiler/代码检查发现同步点远不止 loss 标量：
  `swiglu_monitor` 每次激活调用 4 次 `.item()`（每 micro-batch 上百次）、
  `routing_metrics` 每层 3 次 `float()`、trainer 8 次 loss 标量转换。
- **改动**：
  - SwiGLU 统计改为设备侧累加，monitor 退出时一次 `tolist()`；
  - `routing_metrics` 全部 stack 后一次 `tolist()`；
  - trainer 的 loss/mtp/balance/indexer 合并成一次 `tolist()`；
  - `HcSplitSinkhorn` 的 20 次归一化在 CUDA 上走 `torch.compile(dynamic=True)`，
    带异常回退和 `MINIDEEPSEEK_COMPILE_MHC=0` 开关。
- **实测**：mHC 单次 1.21 ms → 0.557 ms（2.17x），端到端 11,910 → 11,987 tok/s
  （+0.6%，在噪声内）；同步次数 ~230 → ~4 / micro-batch，但 benchmark 不经过
  trainer 的 metrics 路径，收益要在真实训练 step 上确认。
- **风险**：`torch.compile` 首次调用多 ~10 s、每个新 T 可能再编译一次；失败会自动
  回退到 eager，不改变数值（误差 1.8e-7）。

### D72. 训练路径的实测取舍：fused vs reference、冷编译、routing 同步

- **reference 不能作为 fallback**：把 `MINIDEEPSEEK_ATTN_BACKEND=torch` 跑在训练
  的真实 batch（2K 阶段 4096 token/rank、64 累加）上，step 0 = **203–235 s**
  （~870 tok/s）；同一配置用 fused 是 **27.8 s**（~7.3k tok/s），差 **8.4x**。
  之前误判为“reference 只慢 35%”是 benchmark 的 batch 形状（1 doc）造成的。
- **fused 的冷编译约 28 分钟**：16:06 启动、16:34 出 step 0，中间是 attn_gym
  Triton 逐个变体编译；热缓存之后瞬时。evaluating一个新 stage 前要预留这段时间，
  中途 kill 会丢掉整段编译。
- **`MINIDEEPSEEK_ROUTING_GPU=1`**：`RoutingRecord.add` 原来在每个 router 的
  forward hook 里把 3 个张量 `.to("cpu")`（每 micro-batch 21 次中途同步），
  改为在 GPU 上保留、指标仍设备侧计算；不改变数值，只影响 pretraining 的诊断路径
  （replay/compare 仍走 CPU 副本）。
- **`MINIDEEPSEEK_ATTN_BF16` 默认关闭**：bf16 的 Triton lowering 在本机冷编译明显
  更慢（多次 >30 分钟未完成），留作缓存预热后再打开；fp32 fused 已是当前可用的
  最快路径。
- **mHC `torch.compile` 默认关闭**：单模块 2.17x，但整步只有 +0.6%，还要付首次
  编译和 recompile 风险，训练时用 `MINIDEEPSEEK_COMPILE_MHC=0`。

### D73. 冷编译不再是"启动的一部分"：持久缓存 + 显式 prewarm

- **问题**：`MINIDEEPSEEK_ATTN_AUTOTUNE=0` 用 `configs[:1]` 砍掉 autotune，
  但砍剩下的第一个 tile 是给 Hopper/Blackwell 调的（`GENERIC_FORWARD_TILES` 的
  `shared_rows=256, num_stages=3`），在 4090/SM89 上是病态编译：>40 分钟不产出。
  同时 Triton 和 attn_gym 的缓存都在 `$HOME`，换实例就整段丢掉。
- **决定 1：删掉这个开关，不砍候选**。上游 `attn_gym` 的 `TritonTuner` 本来就有
  prune + 落盘 winner 缓存（`<cache>/winners`），砍候选是在跟上游的设计对抗。
- **决定 2：缓存进仓库**。`paths.configure_compile_cache()` 把 `TRITON_CACHE_DIR`
  / `TORCHINDUCTOR_CACHE_DIR` / `ATTN_GYM_CUTE_CACHE_DIR` 指到 `PROJECT_ROOT/.cache/`
  （已 gitignore），显式环境变量仍然优先。`train_curriculum.sh` / `train_pretrain.sh`
  也导出同一组变量，`-m minideepseek.train.run` 这条路径同样命中；
  `_load_attn_gym()` 在 import attn_gym 前再兜一次（Triton 在 import 时读该变量）。
- **决定 3：编译变成一次性、可见的阶段**。`scripts/prewarm_attn.py` 按 curriculum
  每个 stage 的真实 shape（`seq_len` × `index_topk` × dtype）跑一步 `fwd+bwd`，
  逐变体打印 `compiling ... / compiled in Ns`，并把 `--warn-seconds` 的超时变体
  单独标出来。它强制 `MINIDEEPSEEK_ATTN_BACKEND=attn_gym`：编不过就当场失败，
  而不是悄悄退到慢 8x 的 reference。
- **代价**：第一次仍要付 20–40 分钟，但只有一次，且不再藏在 step 0 里；换 stage
  只补新 `index_topk` 的那几个变体。

### D74. index_topk 收到 64，shrink 计划量化到 2 的幂（2026-11-10，4090 实测）

- **问题**：145M 跑不起来，不是环境问题，是 `index_topk` 和 attn_gym 的 kernel 设计
  撞车。该 kernel 的稀疏选择是 `for slot in tl.static_range(0, TOPK)`，`TOPK` 是
  constexpr → **选择循环按 top-k 全展开**。4090 实测：`K=64` forward 编译 89 s，
  `K=128` 459 s，`K=512` >10 min（夜里那次 28 min）。
- **第二个问题**：`min(2048, max(seq_len // 4, 64))` 在本规模上**恰好等于压缩后长度**
  （2K→512/512、4K→1024/1024、8K→2048/2048）：CSA 100% 全选，退化成
  dense-over-compressed，既没有稀疏性，又付了最大的编译代价。V4 的 512 是 1M
  context 下 256K 压缩条目里的 0.2%，不是一回事。
- **第三个问题（真凶）**：`IndexTopKSchedule` 线性插值**每步返回一个新整数**，而
  top-k 是 kernel 的 constexpr → 训练窗口内**每一步重编译一次**。夜里卡在 step 0
  不是"某个变体慢"，是这个。
- **决定**：
  1. `mini_145m_a60m` 的 `index_topk` 固定 64（2K/4K/8K 分别是 12.5% / 6% / 3%）；
  2. `topk_at` 量化到 2 的幂，一条 schedule 最多 `log2(start/final)` 个 specialization，
     可用 `prewarm_attn.py --index-topk` 提前暖；
  3. 8K stage 的 2048 "dense 预热"删除（roadmap §3.5 的前提是 1M 规模）。
- **保留**：HCA 的 top-k 仍由 `(local+1)//compress_ratio` 推导（2K→64、4K→128、
  8K→256），即"覆盖当前文档全部压缩段"，这是 D3 的语义，没有改；代价是 8K 的
  `K=256` 变体仍要单独编一次（本机 ~15–30 min，一次性，进缓存）。
- **验证**：`tests/test_spike.py::test_index_topk_schedule_visits_a_bounded_ladder`
  （2048→64 全程只出现 6 个值）；4090 上 `K=64` 冷编译 89.2 s，之后同 specialization
  的调用 `compiled in 0.0s`。

### D75. HCA 不封顶：8K 覆盖全部压缩段，代价是一次性的 K=256 编译

- **背景**：HCA 的语义是"覆盖当前文档到当前位置的**全部**压缩段"（D3：压缩率 32，
  无 indexer）。`_hca_topk` 的宽度就是 kernel 的 `TOPK` constexpr，而 attn_gym 把
  选择循环 `tl.static_range(0, TOPK)` 全展开，编译时间随 TOPK 超线性增长。
- **D74 曾经的做法**：`hca_topk_max: 128`，把 8K 的宽度从 256 压到 128，
  以复用 4K 已编好的 kernel。**这是语义改动**：128 个压缩段 × 32 token/段 =
  最近 4096 token，8K 文档的**前半段被直接丢掉**，而不是被压缩。
- **决定**：去掉封顶（`hca_topk_max: null`）。宽度回到静态 `ceil(seq_len / ratio)`
  = 2K:64 / 4K:128 / 8K:256，三档都等于压缩段总数，**任何 stage 都不截断**。
- **代价**：8K 的 `K=256` 变体第一次要编一次（实测 >1 h，一次性、进缓存）。
  2K/4K 不受影响（K=64/128 已缓存）。
- **何时付**：8K stage 从 step 8011 开始（按当前 step_time 约 3–4 天后）。
  在那之前找一段 GPU 空闲窗口跑
  `prewarm_attn.py --curriculum ... --max-seq-len 8192`，不能在训练占着 12 GiB
  显存时跑（8K 预检本身要 ~12 GiB）。
- **备选（未采用）**：把 HCA 压缩率从 32 改成 64（更接近官方的 128），8K 就只需要
  `K=128`、可直接复用缓存；代价是所有 stage 的长程分辨率减半，属于比"付一次编译"
  更大的结构性偏离。留作消融。

### D76. bf16 attention 默认打开：443ms → 17ms（2026-11-11，4090 实测）

- **背景**：D72 把 bf16 边界 cast 默认关掉了，理由是“bf16 的 Triton lowering 在
  本机冷编译多次 >30 分钟未完成”。这条在 2K/4K 上**不成立**：今天在干净缓存上
  实测 K=64 **55.2s**、K=128 **372.3s**，都比 fp32 的同变体（89.2s / 459.3s）还快。
  真正病态的只有 8K 的 K=256（fp32 当年 >1h，bf16 未编完，主动中止）。
- **实测（同进程、同 seed、同一条真实 batch，CUPTI kernel 时间）**：
  | | fwd+bwd kernel | attention | main loss |
  |---|---|---|---|
  | fp32 | 1092 ms | 443 ms（40.6%） | 9.177011 |
  | bf16 | 666 ms | **17 ms（2.6%）** | 9.177383 |
  attention **26x**，整步 kernel **-39%**，loss 差 3.7e-4。
- **线上效果**：`step_time` 17.7 → **13.6s**，tok/s 11.2k → **14.1k（+26%）**，
  GPU 利用率 86% → **100%**，功耗 216W/450W。
- **为什么不到 -39%**：bf16 只砍掉 attention 那一块，剩下的每 micro-batch 固定开销
  （MoE dispatch、mHC、rope、~千次 launch）没变；现在 GPU 100% 忙但只有 216W，
  说明是 kernel 开销/访存受限，不是算力受限。下一个杠杆是把 micro-batch 装满
  （见下面的 fill rate 条目），而不是继续压 attention。
- **代价**：8K 阶段（step 8011）首次调用要现编 K=256，必须在那之前找窗口
  `MINIDEEPSEEK_ATTN_BF16=1 prewarm_attn.py --max-seq-len 8192`。没预热就开始跑
  会表现为“卡住一小时”。
- **默认值**：`scripts/train_curriculum.sh` 里 `MINIDEEPSEEK_ATTN_BF16` 默认 1。

### D77（待决）. pack 填充率 75.7% / 81.3% / 95.3%，实际只训到 2.54B token

- **现象**：`pack_domain.py` 是 next-fit——文档放不下就 flush，从不切分。2K 阶段
  平均文档 1,549 token，于是 2048 的 pack 只有 75.7% 满；三阶段合计
  198.3k / 213.1k / 249.8k token 每 step，11444 步只有 **2.54B**，比计划的 3B 少 15.3%。
- **不是 padding**：pack 里没有 padding，GPU 没白算 FLOPs；亏的是每 micro-batch
  固定开销的摊薄，以及 token 预算本身。
- **三个候选**（当前决定：先不动，只做了 D76）：
  1. **loader 侧虚拟重排**（推荐）：`__getitem__` 按 `index * seq_len` 切平坦流，
     positions 在切点归零，`__len__` 改成 `ceil(total / seq_len)`。约 20 行，
     不动数据文件、零磁盘开销、可单测；
  2. 离线重排数据：已在 `pretrain_tool_2k` 上验证（token 流 sha256 与源相同、
     fill 99.996%、800/800 pack 结构合法、mask 与段首一致），但**磁盘只剩 14 GB**，
     4K 那份要 ~15 GB，只能按数组就地原子替换（峰值额外 8.3 GB）；
  3. 加 `until_step`（×1.3）：零风险，+15% 墙钟。
- **额外发现**：`pack_domain.py` 对 `len(ids) > seq_len` 的文档是**截断**而不是切块
  （`ids = ids[:seq_len]`），所以 2K/4K/8K 三份数据是同一语料的不同截断版本。

### D78. pack 填充率：loader 侧按 seq_len 重切（2026-11-11）

- **问题**（D77 的落地）：`pack_domain.py` 是 next-fit，2K 阶段平均文档 1,549 token，
  于是 2048 的 pack 只有 75.7% 满；三阶段每 step 实际 198.3k / 213.1k / 249.8k token，
  11444 步只有 2.54B，比计划的 3B 少 15.3%。
- **决定**：在 `PackedPretrainDataset` 上加 `full_packs`，按 `index * seq_len` 切平坦
  流、切点处让 positions 归零；`data.full_packs` 由配置打开，**只作用于训练集**，
  验证集保留原来的文档对齐网格，这样 val_loss 跨这次切换仍然可比。
- **不动数据文件**：token 流一个字节都不改，磁盘零开销（当时只剩 14 GB，放不下第二份
  15 GB 的 4K 数据）。
- **语义**：跨切点的文档在切点处被拆开，后半段 positions 从 0 重新开始——
  和 `tokenize_documents` 对超过 `seq_len` 的文档的处理完全一致。顺带记下
  `pack_domain.py` 对超长文档是**截断**而不是切块，这是另一个数据问题，本次没动。
- **验证**：
  - `tests/test_data.py::test_full_packs_recut_without_moving_tokens`：token 流逐 id
    相同、除最后一个 pack 外都恰好 seq_len、每段 positions 从 0 递增、`cu_seqlens`
    与段首一致、`target_mask` 与段首一致；
  - 真实 2K general（369M token）：`len` 239,818 → **180,177**，抽样 572 个 pack
    结构违规 **0**，token 流逐 id 相同，单条 128 µs（每 step 约 16 ms，可忽略）；
  - 真实配置跑一步：**202,399 → 262,144 token**（正好等于预算）。
- **代价**：resume 时按 micro-batch 数跳过数据位置，切到新网格后会落在数据流的
  另一个位置（2K 阶段约 47% 而不是 36%）；不影响正确性。

### D79. curriculum 的 validation 从来没跑过：`_val_factory` 只认 `val_path`

- **现象**：三个 run 目录的 `metrics.jsonl` 里一条 `val_loss` 都没有。680 步、约 1.7 亿
  token 的训练没有任何验证信号。
- **根因**：`Trainer._val_factory` 只从 `val_batches` 构建，而 `run()` 只在
  `train.val_path` 存在时才填 `val_batches`。curriculum 用的是 `train.val_sources`
  （每域一个文件），于是 `_val_factory is None`，`train.val_every` 那个条件
  （`... and self._val_factory is not None`）永远不成立，连后面的逐域 eval 也一起跳过。
- **决定**：`run()` 在 `val_path` 缺省时用 `_interleave_sources(val_sources)` 造一个
  聚合流。必须**轮转**而不是拼接：`evaluate` 只取 `val_steps` 个 batch，拼接的话 10 个
  batch 全是第一个域。
- **验证**：
  - 新单测 `tests/test_val_plumbing.py`：轮转顺序/可重复性；只有 `val_sources` 时
    `_val_factory` 非空、两步训练各写一条 `val_loss`、`val_tokens > 0`、eval 后
    恢复 train 模式；
  - 真实 curriculum 配置：三个 val 集各 100 万 token 都能加载，10 个 batch 轮转后
    65,387 token（覆盖三个域）；
  - `pytest tests/test_trainer.py tests/test_curriculum.py tests/test_data.py
    tests/test_val_plumbing.py` → **61 passed**。
- **代价**：每次验证多花约 265k token（聚合 65k + 三个域各 65k），约一个训练 step
  的量，每 200 步一次（~1 小时），可以忽略。

### D80. 8K 的 K=256 编译：`tl.static_range` → `tl.range`，15 分钟变 3 秒

- **根因**（不是候选数、不是显存）：`attn_gym/.../triton/forward.py` 里
  `for selected_slot in tl.static_range(0, TOPK)` 会被 Triton **完全展开**。
  TOPK=256 时 IR 里有 256 份 online-softmax 循环体，SM89 上 `make_ttgir` 跑
  >15 分钟不产出、也不写 cache；而 backward 的同名循环作者用的是 `tl.range`
  （动态 scf.for），所以一直只有 forward 编不出来。
- **修法**：`scripts/patch_attn_gym.py` 把两处换成 `tl.range(0, TOPK)`（幂等、
  留 `.orig` 备份、`--restore` 可回滚、`--check` 可判定）。语义不变：循环顺序、
  累加顺序、mask 都一样，只是不再展开。
- **实测**（8K 形状：T=8192, S=256, K=256, bf16, 4090）：
  | | 打补丁前 | 打补丁后 |
  |---|---|---|
  | 首次调用（含编译） | >900 s，未产出 | **3.0 s** |
  | backward（含编译） | 未测到 | 9.9 s |
  | 稳态 fwd | — | 3.28 ms/launch |
  | 数值 vs torch reference | — | max 7.8e-3 / mean 1.5e-4（bf16 噪声级） |
  | 峰值显存 | — | 1.95 GiB |
- **副作用**：kernel 源码变了，Triton 的 cache key 全变，三个 stage 的 attention
  变体都要重编一次——但有了这个补丁是**秒级**，所以 prewarm 从「必须提前几小时约窗口」
  降级成「可有可无」。正在跑的进程不会被改动生效，必须在 8K 阶段（step 8011）前重启一次。
- **待办**：这是对已安装依赖的补丁，不是 fork。换机器/升级 attn_gym 后要重跑一次
  `python scripts/patch_attn_gym.py`；上游把 forward 也改成 `tl.range` 之后可以删掉。

### D81. 145M 的多卡并行用 ddp，不用 fsdp2（2026-11-11，2×4090D 实测）

- **问题**：`pretrain_145m.yaml` 一直写 `distributed.mode: fsdp2`。在 2×4090D 上
  续训时实测 **19.1 s/step、13.7k tok/s**，和单卡 4090 的 19.2 s/step、13.7k
  **完全一样**——加了一张卡，吞吐零增长。
- **原因**：fsdp2 + `reshard_after_forward: true` 在每个 micro-batch 的
  forward/backward 前都要重新 all-gather 全部参数。145.8M 参数 bf16 约 292 MB，
  每 micro-batch 14 次 gather（7 层 × fwd/bwd），grad_accum 16 → **每步 ~65 GB**
  集合通信；`NCCL_P2P_DISABLE=1` 只能走主机内存、且两张卡跨 NUMA（`nvidia-smi
  topo` 是 `SYS`），实测把省下的 9.6 s 算力全吃掉还多。
- **改法**：145M 这种规模不需要参数分片，切 `mode: ddp`——每步只做一次
  583 MB 的梯度 all-reduce（`all_reduce_gradients`），参数全量常驻每张卡。
- **实测**（同配置、同数据、同 checkpoint 续训，2×4090D）：

  | 指标 | fsdp2 | ddp |
  |---|---:|---:|
  | s/step（262,144 token） | 19.12 | **8.86** |
  | 全局 tok/s | 13,708 | **29,580** |
  | 峰值显存/卡 | 14.5 GiB | **12.8 GiB** |
  | 3B tokens ETA | 2.5 d | **1.3 d** |

  显存反而更低：fsdp2 的 gather buffer 比参数副本更占地方。
- **适用边界**：ddp 把参数/梯度/优化器状态全量复制，145M 在 24 GB 卡上没问题；
  220M（216.7M 参数）能不能同样切要按显存实测决定，`full_6b.yaml` 暂留 fsdp2。
- **验证**：`pytest tests/` 292 passed / 2 skipped；线上从 step 1068 的 checkpoint
  续训，loss 3.71 接得上，1,070/1,080 两步均 29.6k tok/s。

### D82. rollout 不用 vLLM，改项目内 KV-cache 引擎（2026-11-11，4090 实测）

- **原方案**：roadmap 的 `verl + vLLM`；`rl/backends/vllm.py` 也按
  “导出 HF 目录 + `vllm.LLM` 采样”写好了。
- **核查结果**（vLLM 0.33.0 wheel，`vllm/models/deepseek_v4/`）：
  - 只有两类 sparse-MLA 后端：`FLASHMLA_SPARSE_DSV4`（`supports_compute_capability`
    要求 major ∈ {9,10}，kernel 内 `head_dim_v=512`、FP8 KV）和
    `FLASHINFER_MLA_SPARSE_DSV4`（SM100/120）；
  - 我们的 4090 是 **SM89**；mini 模型是 **head_dim=64、bf16**；
  - 即使换 H100，`head_dim_v=512` 也与 64 不匹配。
  - 结论：vLLM 原生 V4 无法服务 mini-145M；要让 vLLM 跑，只能自写自定义
    attention backend（工期与风险都不可控）。
- **改法**：在项目内实现与训练前向数学一致的 slot 寻址 KV cache（`model/cache.py`），
  compressor/indexer 的增量状态机按 official `inference/model.py` 移植；
  `rl/cached_rollout.py` 做批量 prefill + group fork + 压缩解码；
  `rl/graph_decode.py` 做分桶 CUDA graph。保留 `rollout_backend` 的 vLLM seam，
  `vllm` 仍是可选 extra。
- **实测**（4090，145M bf16，batch 64）：
  - decode eager **49 ms/step**（CPU launch 主导，GPU 仅 ~5 ms）；
  - decode CUDA graph **5.4 ms/step**（≈12k tok/s）；
  - fp32 CPU 逐 token 与全量前向 max|Δ|=2e-6；bf16 rollout/actor logprob 差
    ~4e-2（kernel 顺序/批量形状，见 D83）。
- **影响**：
  - 不再期待 `rollout_backend=vllm` 在当前模型上可用；文档与 preflight 都写明；
  - 多卡只用 DDP（D81）：每 rank 一份完整模型 + 一份 cache；
  - 需要为 tool 多轮补 `prefill_append`（当前仍回退全量重算）。

### D83. GRPO 的 old logprob 由 actor 重算（2026-11-11）

- **原方案**：`GRPOTrainer.compute_loss` 用 rollout 采样时记录的
  `sampling_logprobs` 作 old logprob。
- **问题**：rollout 走 decode 引擎（bf16 bmm、batch 形状不同），actor 走全量
  packed forward；逐 token logprob 实测差 **~4e-2**。直接拿采样值当分母，会让
  每个 ratio 从一开始就带 4% 量级的系统偏差，clip 也会误触发。
- **改法**：新增 `rl.recompute_old_logprobs`（默认 true），old = 当前 actor
  forward 的 `response_logprobs.detach()`；ratio 从 1 开始，等价于 on-policy
  GRPO/REINFORCE+baseline。仍保留采样 logprob 用于诊断。
- **依据**：verl 的 actor 侧 `compute_log_prob` 同样在 rollout 后重算 old
  logprob；对 145M 的 forward 成本（~0.1 s）远小于 decode（~5 s/step）。
- **影响**：
  - 一致性门禁不再用 sampling-vs-actor 的 1e-3 阈值卡 cached 引擎；
  - 若将来做多 epoch PPO，需要在第一个 epoch 保存 old，后续 epoch 复用。

### D84. RL/OPD 全程强制 torch reference attention（2026-11-11）

- **问题**：`CachedRolloutEngine` 只在 rollout 的 prefill 阶段把
  `MINIDEEPSEEK_ATTN_BACKEND` 临时切到 `torch`；rollout 返回后环境恢复，
  GRPO 的 actor forward / OPD 的 teacher forward 又走 attn_gym。attn_gym 的
  Triton kernel 按 `(packed layout, topk)` 特化，RL 每个 step 的 response
  长度不同 → 每步都可能重新编译；OPD 第一次 teacher forward 直接卡在
  `py-spy` 里看到的 `attn_gym ... make_cubin`，数分钟不出一 step。
- **改法**：`rl/run.py` 在导入后设置
  `os.environ.setdefault("MINIDEEPSEEK_ATTN_BACKEND", "torch")`；
  `train_grpo.sh` / `train_opd.sh` 也默认导出；SFT/预训练仍走 attn_gym。
- **影响**：
  - RL/OPD 的 actor/teacher forward 变慢（reference 是 fp32 gather），但
    没有编译悬崖，step time 可预测；GRPO 实测 actor fwd+bwd 8.9 s
    （49k token，含 mini-batch 16）。
  - 数值上 reference 与 cache 解码的差异仍按 D83 处理（old logprob 重算）。
  - 若将来 RL 要走 attn_gym，需要按固定长度分桶 + 预编译，而不是按实际
    packed 长度。

### D85. 预训练从 `lockstep: true` 改为 `false`：optimizer step 不再单领域（2026-11-11）

- **原方案**：`data.lockstep: true`，每个 optimizer step（262,144 token = 16 个
  micro-batch × 16384）**整步只取一个领域**，目的是让每步的 router 指标天然是
  分域的。
- **问题**（145M run step 2393 实测）：
  - 日志里的 `loss` 在 **1.47 ↔ 3.85** 之间锯齿跳，`loss_ema` 永远收敛不到
    一条线上：它只是"这一步抽到了哪个领域"的加权，不是模型质量；
  - Muon 动量 0.95 有 ~20 步记忆，而领域块平均只有 ~11 步（82/15/3），
    动量把上一个领域的梯度方向拖进下一个领域的更新里——这解释了
    "权重 RMS 每 130 步移动 19%、loss 却不动"；
  - router 被逼成按步特化：在 code token 上 layer.2 的分布是
    `[9.6, 40.1, 33.9, 0.2, 16.2, 0.0]`，两个专家 0.0%/0.2%；
  - 连主领域的路由也在退化：general 上的 per-layer min-share
    step 2000 → 2200 → 2393 依次是 **14.2% → 10.4% → 9.3%**，
    max-share 18.9% → 22.4% → 23.3%。
- **新方案**：`lockstep: false`，调度按 **micro-batch** 建表并打乱，一个
  optimizer step 的 16 个 micro-batch 按 82/15/3 混合。
- **代价 / 取舍**：
  - 每步的领域构成会浮动（13/3/0 或 14/2/0），所以步级 loss 仍有方差，
    这就是新增分域 EMA（`train_main_loss_<domain>`）的原因；
  - `domain_id` / `domain_name` 只在**整步同域**时才写，混合步写 -1/""；
  - 路由特化本身是 MoE 想要的，混合不会消灭它（routing 是逐 token 的），
    只是去掉了"强制交替"这个放大器。要让 code/tool 的极端化收敛，还得
    调 `balance_loss_coeff`（现在是 1e-4，见 D88 的观察项）。
- **验证**：重启后 `step 2400 | lm 2.2625 | general 2.4153 code 1.3959
  tool 0.9745`，三个曲线同时可读，且都低于改造前的 running mean。

### D86. cosine 衰减从 80% 提前到 20%（2026-11-11）

- **原方案**：`decay_start_ratio: 0.8` → 峰值 6e-4 一直保持到 step 9,155
  （全程 11,444 步的 80%），只有最后 20% 才衰减。
- **问题**：4K 阶段全程跑在峰值 LR 上，而 step ~1,300 起 general 的
  train main loss 就压在 2.55 不动、验证集三域同时抬升
  （tool 0.909 @1599 → 1.295 @2199）。权重每 130 步移动 19% RMS 却不降
  loss，是"高 LR 噪声地板"的典型特征。
- **新方案**：`decay_start_ratio: 0.2` → 衰减起点正好落在 step 2,288 =
  当前进度，剩余 9,155 步 cosine 从 6e-4 退到 6e-5。
- **同时修的坑**：`resume()` 会从 checkpoint 恢复 optimizer 的
  `param_groups`，连带恢复 `base_lr`，于是**改配置里的 `optim.lr`
  在续训时是静默无效的**。新增 `Trainer._apply_config_learning_rates()`，
  在恢复后让配置覆盖 checkpoint，并打印 `config LR overrides ...`。
  否则 D86 这类实验会看起来毫无效果。
- **未验证**：没有做 6e-4 vs 2e-4 的单变量 A/B（2×4090D 显存只剩 ~1 GB，
  训练期间无法并行跑对照）。峰值是否仍然偏高，要看接下来 500–1000 步
  分域 loss 的斜率再判断。

### D87. 重建验证集：40M/16M/6M 的独立尾部切片（2026-11-11）

- **原方案**：每个领域从语料**开头**取 1M token 当 val（`skip_tokens: 0`）。
- **问题**：1M token 是 general 语料的 0.07%，而且不具代表性——在
  `val_general_4k` 上 layer.3 的某个专家只拿到 0.8% 的 token，同样的权重
  在训练流上拿 17.2%。`val_steps: 10` 只抽 10 个 micro-batch（~160k
  token），比"打包长度"带来的差异还小，曲线基本是噪声。
- **新方案**：从三个 8K 训练区间的**尾部**切出 general 40M / code 16M /
  tool 6M 作为 held-out（`scripts/make_val_split.py`），并把它们从 8K
  训练区间里裁掉，保证与所有训练区间不相交；`val_steps` 10 → 64。
- **为什么从 8K 区间切**：8K 阶段（step 8011 起）还没开始，裁它的尾部
  不会动到已经消费过的数据；4K 区间正在被消费，改它会让续训的 token
  偏移对不上。
- **配套**：`make_val_split.py` 的三条安全约束——只切尾部、`trim` 先写
  暂存目录再 `os.replace`（同一个 inode 上的 memmap 会被 `np.save` 的
  `open("wb")` 直接截断，这个坑真的踩了，见 PROCESS 2026-11-11）、
  文档长度 > `seq_len` 时显式打印被裁掉的 token 比例。
- **验证**：新旧 val 在同一 checkpoint（step 2393）上
  general 2.148 / code 1.351 / tool 1.181（新）vs 2.619 / 1.439 / 1.295
  （旧，step 2199）。数值不可比（不同文本），但新集合内部一致、且每域
  都有 3.9k–9.8k 个 pack 可选。

### D88. 更正："loss 不降"约六成是测量假象，四成是 general 学得慢（2026-11-11）

先用同一段评测代码（CPU fp32、同一批数据）跨 checkpoint 复测，这是唯一口径一致的
对比（训练日志里的数字被 bf16 + pack 重切抬高）：

| checkpoint | train_general | val_general_v2 | train_code | val_code_v2 |
|---|---:|---:|---:|---:|
| step 2200 | 2.3034 | 2.1660 | 1.4350 | 1.5289 |
| step 2393 | 2.2844 | 2.1315 | 1.3428 | 1.4351 |
| 变化 / 193 步 | **−0.019** | **−0.035** | **−0.092** | **−0.094** |

**三个域都在降，没有一个在退化。** 之前"停滞"的判断有四个测量来源：

1. **headline 里含 MTP**：`loss = main + 0.3·mtp`，MTP 一贯比 main 高 0.25 左右，
   所以盯 3.4 看不出一条 2.55 的曲线在动（D85 已改控制台口径）；
2. **lockstep 让步级 loss 在 1.5 ↔ 3.9 之间跳**，把 /1000 步量级的趋势完全盖住；
3. **训练侧数字被抬高约 0.24 nats**：训练用 bf16 autocast，且 `data.full_packs`
   把 pack 网格重切到 4096（文档被切断、续段从 position 0 冷启动），而验证集
   按 `train_full_packs=False` 保留文档对齐的 pack 网格。所以
   **控制台的 train loss 与 val loss 不可直接比较**（同一 checkpoint：控制台
   general ≈ 2.52，CPU fp32 实测同分布 2.284）；
4. 我之前读的分域验证字段是 `val_loss_<domain>`（含 MTP），不是 main head。

**仍然成立的**：general 是三个域里最慢的（−0.02/193 步 ≈ −0.1/1000 步），
按这个斜率到 2.0 还要约 2,800 步；D85–D87 的改动仍然必要——它们让上面这些
区别变得看得见，而不是改变了一个没有问题的训练。

**因此没有下调峰值 LR**：grad norm 只有 0.3（`grad_clip=1.0` 从未触发），
6e-4 对 145M/262k-token 步属于常规区间，"LR 过高"缺少直接证据。
下一步的判据是 step 2599 / 2799 两个新验证点是否继续下降；若 general 的斜率
在 400 步内明显转平，再考虑降到 3e-4（`base_lr` 覆盖已修，改配置即可生效）。

### D89. 8K general 数据从本地备份完整恢复（2026-11-11）

- **背景**：D87 的 `trim_source` 第一版原地重写 memmap 源文件，把
  `pretrain_general_8k.input_ids.npy`（738M token / 2.95 GB）截断成 128 字节。
  当时的临时处置是用 4K 的 token 流在 8192 打包替代，代价是 8K 阶段的
  general 有约 55% 是 stage 1 已见过的 token（原估 35% 是错的：stage 1 正好
  消耗完 4K 区间一整遍，699M/1353M = 55%）。
- **恢复来源**：本地工作区 `data/tokenization/` 存着同一份打包产物。
  先证明同一性：4 个共享边车（含对文档边界极敏感的 `offsets.npy`）md5 逐项
  相同、两个 token 流前 1 MiB md5 相同、general_8k 的 6 项元数据与损坏前一致。
- **恢复判据**：`val_general_8k_v2` 是损坏**前**从原文件切出的，用恢复后的
  文件重切同样的 40M 尾部，四个数组 md5 全部一致 —— 逐字节精确。
- **替换方式**：`gzip -1` + SSH（2.62×、6.9 MB/s、6.64 GB / 约 3 分钟）传到
  `.incoming/`，校验 md5 后**先 `rm` hardlink 再 `mv`**。当时 general_8k 与
  general_4k 是同一 inode（links=2），用 `cat >` 写穿会连带毁掉正在使用的
  4K 数据；替换后 4K 的 inode/links/前缀 md5 均未变。
- **结果**：`pretrain_general_8k` = 698,000,234 token / 87,932 packs，
  `skip_tokens` 回到真实的 1,723,000,000，与 curriculum 里 8K 阶段的
  698M general 预算一致。18 个数据集全部可加载。
- **教训**：任何"就地重写数据集"的路径都不允许存在——包括看起来安全的
  `rm`+`mv`（必须先确认目标不是别人的 hardlink）。数据集产物应留一份
  校验和清单，且**源语料不要在同一块盘上删到只剩一份**：这次的恢复完全
  依赖本地那份备份。
