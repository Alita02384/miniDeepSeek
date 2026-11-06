# mini_220m_a80m

| 字段 | 值 | 描述 |
|---|---|---|
| dim | 768 | 模型隐藏维度（residual stream 宽度） |
| n_layers | 7 | 主 Transformer 层数（不含 MTP） |
| n_mtp_layers | 1 | MTP 深度（额外 1 个预测层） |
| n_heads | 16 | 注意力 query head 数 |
| head_dim | 256 | 每个 head 的维度（Q/K/V） |
| rope_head_dim | 32 | RoPE 维度；32/256 = 12.5%，与 V4 的 64/512 同比例 |
| q_lora_rank | 288 | Q 的 MLA 低秩压缩维度 |
| o_lora_rank | 288 | 输出投影的 MLA 低秩压缩维度 |
| n_groups | 2 | MLA 输出分组数（16 heads / 2 groups = 8 heads/组） |
| compress_ratios | (0, 0, 4, 32, 4, 32, 4) | 每层注意力类型：0=SWA，4=CSA，32=HCA |
| window_size | 128 | SWA 滑窗大小 |
| index_n_heads | 8 | CSA indexer 的 query head 数 |
| index_head_dim | 64 | CSA indexer 的 head 维度 |
| index_topk | 1024 @4K / 2048 @8K | CSA 稀疏注意力的 top-k |
| moe_inter_dim | 1216 | 每个 routed expert 的 FFN 中间维度 |
| num_experts | 8 | routed expert 数量 |
| num_shared_experts | 0 | shared expert 数量（本配置无 shared） |
| top_k | 2 | 每 token 激活的 routed expert 数 |
| n_hash_layers | 2 | 前 2 个 MoE 层使用 token-id hash routing |
| route_norm | True | 是否对 top-k 路由权重归一化 |
| route_scale | 1.5 | 路由权重缩放系数 |
| load_balance_coeff | 1e-3 | loss-free 负载均衡 bias 的更新系数 |
| swiglu_limit | 10.0 | SwiGLU 激活的 clamp 上限 |
| hc_mult | 4 | mHC 的 hyper-connection 倍数 |
| sinkhorn_iters | 20 | mHC Sinkhorn 归一化迭代次数 |
| vocab_size | 6400 | 词表大小（MiniMind tokenizer） |
| tie_word_embeddings | True | embedding 与 lm_head 是否共享权重 |
| seq_len | 4096 | 主训练序列长度 |
| max_context | 8192 | 最大上下文长度 |
| 总参 | 216.7M | 含 MTP depth 1 的总参数量 |
| 激活 | 82.2M | 每 token 激活参数量（含 MTP） |

# mini_60m_dense_d768

| 字段 | 值 | 描述 |
|---|---|---|
| dim | 768 | 模型隐藏维度（residual stream 宽度） |
| n_layers | 7 | 主 Transformer 层数（不含 MTP） |
| n_mtp_layers | 1 | MTP 深度（额外 1 个预测层） |
| n_heads | 12 | 注意力 query head 数（Q 总维度 = d） |
| head_dim | 64 | 每个 head 的维度；12×64 = 768 = d |
| rope_head_dim | 32 | RoPE 维度（head_dim 64 的一半） |
| q_lora_rank | 288 | Q 的 MLA 低秩压缩维度 |
| o_lora_rank | 288 | 输出投影的 MLA 低秩压缩维度 |
| n_groups | 2 | MLA 输出分组数（12 heads / 2 groups = 6 heads/组） |
| compress_ratios | (0, 0, 4, 32, 4, 32, 4) | 每层注意力类型：0=SWA，4=CSA，32=HCA |
| window_size | 128 | SWA 滑窗大小 |
| index_n_heads | 8 | CSA indexer 的 query head 数 |
| index_head_dim | 64 | CSA indexer 的 head 维度 |
| index_topk | 1024 @4K / 2048 @8K | CSA 稀疏注意力的 top-k |
| moe_inter_dim | 2304 | FFN 中间维度（3.0d）；单专家即 dense FFN |
| num_experts | 1 | 单专家（等价 dense FFN） |
| num_shared_experts | 0 | 无 shared expert |
| top_k | 1 | 每 token 激活唯一专家（dense） |
| n_hash_layers | 0 | dense，无 hash routing |
| route_norm | False | 单专家，无需归一化 |
| route_scale | 1.0 | 单专家，路由权重恒为 1 |
| load_balance_coeff | None | 单专家，无需负载均衡 |
| swiglu_limit | 10.0 | SwiGLU 激活的 clamp 上限 |
| hc_mult | 4 | mHC 的 hyper-connection 倍数 |
| sinkhorn_iters | 20 | mHC Sinkhorn 归一化迭代次数 |
| vocab_size | 6400 | 词表大小（MiniMind tokenizer） |
| tie_word_embeddings | True | embedding 与 lm_head 是否共享权重 |
| seq_len | 4096 | 主训练序列长度 |
| max_context | 8192 | 最大上下文长度 |
| 总参 | 60.9M | 含 MTP depth 1 的总参数量 |
| 激活 | 60.9M | dense，每 token 全部参数激活（含 MTP） |

# mini_145m_a60m

| 字段 | 值 | 描述 |
|---|---|---|
| dim | 768 | 模型隐藏维度（residual stream 宽度） |
| n_layers | 7 | 主 Transformer 层数（不含 MTP） |
| n_mtp_layers | 1 | MTP 深度（额外 1 个预测层） |
| n_heads | 12 | 注意力 query head 数（Q 总维度 = d） |
| head_dim | 64 | 每个 head 的维度；12×64 = 768 = d |
| rope_head_dim | 32 | RoPE 维度（head_dim 64 的一半） |
| q_lora_rank | 288 | Q 的 MLA 低秩压缩维度 |
| o_lora_rank | 288 | 输出投影的 MLA 低秩压缩维度 |
| n_groups | 2 | MLA 输出分组数（12 heads / 2 groups = 6 heads/组） |
| compress_ratios | (0, 0, 4, 32, 4, 32, 4) | 每层注意力类型：0=SWA，4=CSA，32=HCA |
| window_size | 128 | SWA 滑窗大小 |
| index_n_heads | 8 | CSA indexer 的 query head 数 |
| index_head_dim | 64 | CSA indexer 的 head 维度 |
| index_topk | 1024 @4K / 2048 @8K | CSA 稀疏注意力的 top-k |
| moe_inter_dim | 1152 | 每个 routed expert 的 FFN 中间维度（1.5d） |
| num_experts | 6 | routed expert 数量 |
| num_shared_experts | 0 | shared expert 数量（本配置无 shared） |
| top_k | 2 | 每 token 激活 2 个 routed expert（33% 激活率；激活 FFN 宽度 3.0d） |
| n_hash_layers | 2 | 前 2 个 MoE 层使用 token-id hash routing |
| route_norm | True | 是否对 top-k 路由权重归一化 |
| route_scale | 1.5 | 路由权重缩放系数 |
| load_balance_coeff | 1e-3 | loss-free 负载均衡 bias 的更新系数 |
| swiglu_limit | 10.0 | SwiGLU 激活的 clamp 上限 |
| hc_mult | 4 | mHC 的 hyper-connection 倍数 |
| sinkhorn_iters | 20 | mHC Sinkhorn 归一化迭代次数 |
| vocab_size | 6400 | 词表大小（MiniMind tokenizer） |
| tie_word_embeddings | True | embedding 与 lm_head 是否共享权重 |
| seq_len | 4096 | 主训练序列长度 |
| max_context | 8192 | 最大上下文长度 |
| 总参 | 145.8M | 含 MTP depth 1 的总参数量 |
| 激活 | 60.9M | 每 token 激活参数量（含 MTP） |

## 落地状态（2026-11-06）

三个方案均已实现为 `configs/model/*.yaml`，并由 `ModelSettings` /
`build_model_config_from_settings` 加载；`mini_220m_a80m` 是默认训练模型。

实测（`sum(p.numel())`，含 MTP depth 1、tied embeddings）：

| 配置 | 总参 | 每 token 激活 |
|---|---|---|
| `mini_220m_a80m` | 216,652,538（216.65M） | 82.17M |
| `mini_60m_dense_d768` | 60,876,570（60.88M） | 60.88M |
| `mini_145m_a60m` | 145,841,946（145.84M） | 60.91M |

验证：三档 CPU forward/backward 与参数校准单测通过；`mini_220m_a80m`
1-step smoke + checkpoint 回读通过。head_dim=256 的 4K/8K 真实显存未在 GPU
验证，见 `decisions.md` D51。
