# 145M MoE 数据方案（code / tool / general 三专家）

> 对象：`mini_145m_a60m`（145.84M 总参 / 60.91M 激活）、6400 词表、4K 主训上下文。
> 专家划分：**code、tool-use、general**（原 math / world 已取消，见 `doc/decisions.md`）。
> 目标：数据量不大，但每一条都要能用。
>
> 配套：`doc/specialist_data_145m.md`（现有语料实测 + SFT 甜点量 + 能否直接提取）、
> `doc/moe_specialization.md`（专家分化与数据排布）。
>
> 与路线图的关系：`miniDeepSeek_technical_roadmap.md` §4 是 60M / 6B token 的基线，
> 本文是 145M / 3B / 三专家的收敛版；预算、领域、难度三处偏离记录在 `doc/decisions.md`。

---

## 0. 结论速览

| 阶段 | 预算 | 主料 | 一句话口径 |
|---|---|---|---|
| 预训练 | **3B tokens** | MiniMind 中文 + fineweb-edu + chinese-fineweb-edu + **code 15%** + cosmopedia + **tool 3%** | 现有 2.2B 语料 code 仅 2.1%、tool 为 0，必须补 |
| general SFT | 40–60k 条（15–20M tok） | 从 `sft_t2t.jsonl` 过滤 | 剔除代码/数学/tool 后再抽 |
| code SFT | 25–35k 条（12M tok） | 现有语料抽 10–30k + Magicoder + self-oss | 现有只能到"代码问答"档，要补可执行代码 |
| tool SFT | 5–10k 条（4–6M tok，2–3 epoch） | 已有 `data/sft_tool.jsonl` | 窄格式技能，少而重复 |
| code GRPO | 1–2k 题 | MBPP + open-r1 + TACO EASY | 先改 reward 部分分，否则恒 0 |
| tool GRPO | 3–5k prompt | 现有 + xLAM / ToolACE | 已有 reward，补 prompt 多样性 |
| general GRPO | 现有池 | `grpo_general_train.jsonl` | 保持 |

---

## 1. 三个硬约束

### 1.1 词表只有 6400：实测 fertility

同一批样本，tok/byte（越小越省）：

| tokenizer | vocab | emb@768 | 中文 | 英文 | Python | GSM CoT |
|---|---|---|---|---|---|---|
| **minimind-6400（当前）** | 6,400 | 4.9M | **0.245** | 0.283 | 0.426 | 0.444 |
| SmolLM-135M | 49,152 | 37.7M | 0.549 | **0.183** | **0.290** | 0.317 |
| Qwen2.5-0.5B | 151,665 | 116.5M | **0.207** | 0.179 | 0.241 | 0.309 |
| TinyLlama-32k | 32,000 | 24.6M | 0.410 | 0.193 | 0.339 | 0.335 |
| GPT-2-50k | 50,257 | 38.6M | 0.709 | 0.194 | 0.455 | 0.259 |

缩进与标识符的实测成本：

```text
"    return value\n"  -> 5 tokens      "  return value\n" (2 空格) -> 4 tokens
"\treturn value\n"    -> 6 tokens
"def compute_mean(values):" -> 11 tokens（comput|e|_|me|an 被切碎）
```

**结论**：

1. 代码数据**只做 Python**（C++ 比 Python 再贵 ~23%，JS 贵 ~6% 但收益不值）；
2. 代码样本**限短**（≤60 行、≤400 token 输出）；
3. 数学/工具数据避开 LaTeX 与超长 JSON schema；
4. 中英混料别为省 token 全用中文——英文数据质量高得多。

### 1.2 难度天花板

145M 总参 / 61M 激活约等于 100M 级 dense：

| 任务 | 现实预期 | 该不该进训练 |
|---|---|---|
| general 指令跟随 | 可用 | ✅ 主战场 |
| 函数级 Python（MBPP 档） | 5–15% | ✅ code 主战场 |
| 单/双工具调用（给定 schema） | 20–50% | ✅ tool 主战场 |
| 多轮 agent / 复杂 API 编排 | ≈0 | ❌ 不要 |
| Codeforces / APPS competition | ≈0 | ❌ 不要 |

### 1.3 预算（按现有配置算）

- 预训练：`configs/pretrain/curriculum_145m_4x4090.yaml` 现为 **1.2B**；本文建议提到 **3B**
  （2.2B 现有 + 新增 code/tool/英文语料），145M 上 5B 以上边际收益很低。
  有效 batch 建议 131k → **262k**（`grad_accum` 16，显存不变），3B ÷ 262k ≈ 11.4k 步；
  课程三段的 `until_step` 要按新 token/step 重算——见 `doc/batching.md` §1–§3。
- SFT：`configs/sft/*.yaml` 500 step × `batch_tokens: 32768` × `grad_accum: 8`
  → **131M token/专家**（`batch_tokens` 是全局 micro-batch，per-step 要再乘 grad_accum）。
  注意这与 §0 的 40–60k 条数据（约 15M token）差 8 倍，等于跑 8 个 epoch；
  **要么把 steps 降到 60–80，要么把数据扩到 40 万条**，建议前者。
- GRPO：`configs/grpo/*.yaml` 200 step × `prompts_per_step: 32` → **6,400 次 prompt 采样**。

---

## 2. 词表：保持 6400

只比 tok/byte 会得出"换大词表更省"的错误结论，要把 **LM head 算力**一起算：
每 token head 是 `2 × 768 × vocab`，而 body（61M 激活）约 `2 × 60.9M ≈ 122 MFLOPs`。

| vocab | head MFLOPs/token | 中文 MFLOPs/byte | 英文 | Python |
|---|---|---|---|---|
| 6,400 | 9.8 | **32.3** | 37.3 | 56.2 |
| 49,152 | 75.5 | 108.4（3.4×） | 36.1（≈打平） | 57.3（≈打平） |
| 151,665 | 233.0 | —（embedding 116.5M，不可行） | | |

再加上 `rl/opd.py` 用 **full-vocabulary reverse KL**（文件头注释明确写了依赖 6400 词表），
GRPO 也要全词表 logprob——大词表在 RL 阶段是 8–24 倍的 logits 量。

**结论：不换公开词表。** 真要改进，只能现在（预训练开跑前）自己训 16k BPE，
预计中文 ~0.21 / 英文 ~0.24 / Python ~0.36、embedding 12.3M；这属于对
AGENTS.md §1.1/§7 的偏离，需写 `doc/decisions.md`。

---

## 3. 预训练：3B 混料

### 3.1 现状与缺口【实测】

`data/pretrain_t2t.jsonl`（8.28 GB，~2.2B token）抽样 3 万条：

| 类别 | 占比 | 说明 |
|---|---|---|
| general | 96.1% | 中文合成指令/问答文风 |
| code | 2.1% | "教你怎么写代码"的对话，不是源码 |
| math | 1.8% | 解题对话 |
| **tool** | **0.0%** | 完全没有 |

### 3.2 配比（已落成三个文件）

数据在**数据集这一层**就按域拆好，训练侧只做 batch 级混料。三个文件都经过
**精确去重 + 按预算裁剪**（`scripts/dedup_and_split.py`）：

| 文件 | 内容 | 文档数 | 估计 token | 去重删掉 |
|---|---|---|---|---|
| `data/pretrain_general.jsonl` | MiniMind `pretrain_t2t` + fineweb-edu(`int_score≥4`) + chinese-fineweb-edu + cosmopedia-v2 + finemath(1/7) | 5,362,052 | **2.46B** | 10,645（0.1%） |
| `data/pretrain_code.jsonl` | codeparrot-clean（license 白名单 / 非自动生成 / ≤200 行 / ≤8KB / alpha≥0.3） | 310,244 | **0.45B** | 6,364 |
| `data/pretrain_tool.jsonl` | ToolACE + glaive 的 schema→call 文本 | 114,737 | **0.09B** | 9,237 |

> 去重是**跨源精确去重**：对规范化文本取 blake2b-128，两个来源共用一个哈希集合。
> general 的重复率只有 0.1%，说明两批语料基本不重叠；去重后 4.56B token，
> 按目标 2.46B 做**哈希均匀裁剪**（保留 54%），所以域内比例不变。
> 估计口径：MiniMind `tok/char=0.743`、外部 general `0.335`、code `0.432`、tool `0.336`
> （实测样本），**没有在本机跑 tokenizer**——打包在训练服务器上做。

**总量 3B**，2K/4K/8K 三段的 token 区间**互不重叠**（`--skip-tokens` 递进），
所以课程跑完每个 token 只被看一次：

| 阶段 | token | general / code / tool |
|---|---|---|
| 2K | 0.450B | 369M / 67.5M / 13.5M |
| 4K | 1.650B | 1353M / 247.5M / 49.5M |
| 8K | 0.900B | 738M / 135M / 26M |

打包产物（每域每长度一套 `.npy` sidecar）：

```text
data/tokenization/
├── pretrain_{general,code,tool}_{2k,4k,8k}.npz   # 训练
├── val_{general,code,tool}_{2k,4k,8k}.npz        # 各 1M token，训练跳过
```

生成命令：`bash scripts/build_pretrain_packs.sh`；ETL：`python scripts/build_external_data.py`。

### 3.3 过滤

沿用 `data/quality.py`（实测 Python 源码 `accepted=True`），额外加：

1. code：`license` 白名单、`autogenerated=False`、`line_max ≤ 200`、`alpha_frac ≥ 0.3`、≤8KB；
2. tool：只保留"函数名 + 参数 + 一句描述"，schema 深度 ≤ 2、参数 ≤ 6 个；
3. 全部：MinHash 文档级 + 段落级去重；13-gram 对评测集查污染；
4. 抽样 2k 条跑 `prepare fertility`，确认三类 tok/byte 不劣化。

### 3.4 注意：数据是成块的

现有语料按来源成块拼接（pretrain 在 58% 处有一段 code 15.2%）。
**不要按文件顺序训练**，会变成"先学通用、最后学 tool"；按领域分组 + 打乱 batch 顺序，
详见 `doc/moe_specialization.md` §4。落地方式是 `data.sources` + `data.lockstep: true`
（每 step 同域、`metrics.jsonl` 记 `domain_id`），见 `doc/batching.md` §5。

---

## 4. code 专家（缺口最大）

### 4.1 SFT：目标 25–35k 条

| 来源 | 规模 | 许可 | 用法 |
|---|---|---|---|
| 现有 `sft_t2t.jsonl` 提取 | ~154 万条含代码块，过滤后 10–30k | 随 MiniMind（CC-BY-NC） | **起步用**，但只到"代码问答"档 |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 75,197 | MIT | 按 `lang=="python"` 过滤，主料 |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 50,661 | ODC-By | 已做执行过滤，质量最稳 |
| `m-a-p/CodeFeedback-Filtered-Instruction` | 156,526 | Apache-2.0 | 取简单子集 |
| `iamtarun/python_code_instructions_18k_alpaca` | 18,612 | 需核对 | 只留含 `def` + docstring |
| `google-research-datasets/mbpp` train | 374 / 120 | CC-BY-4.0 | "写函数"范式锚点 |

**过滤规则**（缺一不可）：

```text
ast.parse 通过 且 至少一个 def
单文件、≤60 行、≤400 token
只 import 标准库（numpy/pandas/torch/requests 全丢）
禁 input() / 网络 / 写文件
按函数名 + AST 归一化去重
```

**注意**：从现有语料抽出来的样本抽样看，混杂 CSS 问答、架构描述、伪代码，
必须强过滤；它们覆盖不到"函数签名 + 单元测试"这一档，所以 GRPO 仍需外补。

### 4.2 GRPO：目标 1–2k 题

| 来源 | 规模 | 说明 |
|---|---|---|
| `MBPP`（full + sanitized train） | 494 | 函数级、测试现成，**先跑通这条** |
| `open-r1/verifiable-coding-problems-python_decontaminated-tested` | 15,068 | 自带 `fn_name/input/output`，过滤短题取 500–1k |
| `BAAI/TACO` EASY | 25,443 中筛 | 需 `input_output` 非空、无图、题面 <800 字符 |
| 自建 oracle（codeparrot 反推） | 无限 | 真实函数→删函数体→用原函数生成测试 |

### 4.3 前置条件：改 `code_reward`

当前 `rl/reward.py::code_reward` 是"整段测试 exit code 0 → 1.0，否则 0.0"。
145M 上会恒 0、无梯度。**必须先改成按测试给部分分**（路线图 §4.4.2 本来就写"部分通过按比例"），
并逐条 assert 跑。

---

## 5. tool 专家（数据已有，补多样性）

### 5.1 现状

`data/sft_tool.jsonl`（158 MB）与 `data/grpo_tool_train.jsonl`（89 MB）已就绪，
来源是 MiniMind `sft_t2t_mini` 尾部 13.5 万条 tool 块 + `agent_rl`。

---

## 6. general 专家

### 6.1 SFT：40–60k 条

`minimind/dataset/sft_t2t.jsonl`（14.10 GB，约 857 万条）实测构成：
**general 72.4% / code 18.0% / math 9.5%**，末尾还有 tool 块。
所以**不能整份直接当通用专家数据**，否则通用专家会吃进 18% 代码、和 code 专家抢路由。

过滤规则（按内容，不按行号——数据是成块的）：

```python
drop if "```" in record
drop if "\\frac" / "\\boxed" / "解方程" in record
drop if message.get("tools") or message.get("tool_calls")
drop if len(turns) < 2 or not assistant content
```

过滤后仍有百万级，抽 40–60k 即可。

### 6.2 预训练

见 §3.2 的 fineweb-edu / chinese-fineweb-edu / cosmopedia 三组，共 1.2B。

---

## 7. GRPO 的数据量口径

GRPO 的瓶颈不是题目数，是 **rollout 次数**和**难度分布**。按每步 32 prompt × group 8：

| 训练步数 | prompt 抽样 | rollout 条数 | 建议 in-band pool | 原始池 |
|---|---|---|---|---|
| 200（当前） | 6,400 | 51,200 | 2–4k | 7–10k |
| 500 | 16,000 | 128,000 | 5–8k | 20k |
| 1,000 | 32,000 | 256,000 | 10–15k | 35–50k |
| 2,000 | 64,000 | 512,000 | 20–30k | 70k+ |

要点：

1. **"2–4k"是过滤后的 in-band 数量**；数学/代码原始池都很大，稀缺的是"当前能做对 10–90%"的题；
2. **复用 prompt 不丢人**：on-policy 下策略每步在变，同一题下一轮就是新样本；
   超过 8–10 epoch 才开始背题；
3. **代码池是真稀缺**：MBPP 只有 494，open-r1 过滤后 1–2k，要上万只能自建。

---

## 8. 划分与污染

| 集合 | 用途 |
|---|---|
| MBPP `test`/`sanitized test`、HumanEval、HumanEval+、LiveCodeBench | ❌ 禁止训练 |
| APPS `test`、TACO `test`、Codeforces | ❌ 禁止训练 |
| BFCL / τ-bench / ToolBench 测试集 | ❌ 禁止训练（tool 评测） |
| C-Eval / CMMLU | ❌ 禁止训练 |
| GSM8K test（若保留 math 能力评估） | ❌ 禁止训练 |

落盘前跑：

```bash
python -m minideepseek.data.prepare contamination \
    --train data/processed/sft/code/train.jsonl \
    --eval  data/processed/eval/humaneval.jsonl --n 13
```

注意跨集泄漏：`sft_t2t` 里已含代码/数学，用于 general 时必须先过滤；
MBPP 的 train 与 test 不可混。

---

## 9. 与本仓库的对接

| 阶段 | 命令 | 输入格式 |
|---|---|---|
| 预训练 | `prepare curriculum --input … --out-dir data/tokenized/hq` | `{"text","domain","source"}` |
| SFT | `prepare sft --input … --output data/tokenized/sft_code.npz` | `{"messages":[…]}` |
| GRPO | `prepare grpo --input … --domain code --verifier unit_test` | `{"prompt","reference_answer","rubric"}` |
| 混料 | `data/manifests/*.yaml` + `manifest` | `sources[{name,path,domain,license,weight,available_tokens,max_epochs}]` |

需要新增/修改：

- `data/manifests/pretrain_3b.yaml`（§3.2 七源）
- `data/manifests/specialists.yaml`：加 `code` 段（当前注明 "code expert is intentionally absent"）
- `configs/{sft,grpo}/tool.yaml`、`general.yaml`；`math/world.yaml` 归档
- `data/processed/eval/code.jsonl`（HumanEval/MBPP test）

---

## 10. 实施顺序

1. **code SFT**：先 Magicoder + self-oss 组 25k，跑 100 step 看生成格式；
2. **改 `code_reward` 为部分分**，再接 MBPP 494 题跑通 GRPO 闭环；
3. **pre-train 升到 3B**：加 code 15% + tool 3%，必须在重新 tokenize 前定；
4. **general 过滤**：从 `sft_t2t.jsonl` 抽 40–60k，验证不含代码/数学/tool；
5. **tool 补强**：xLAM/ToolACE 进 GRPO prompt 池；
6. **分化实验**：按 `doc/moe_specialization.md` 做领域 batch + 平衡损失退火。

---

## 附：数据集速查

| 名称 | 类型 | 规模 | 许可 | 字段 |
|---|---|---|---|---|
| `sft_t2t.jsonl`（本地） | general SFT | ~857 万条 | CC-BY-NC-4.0 | `conversations` |
| `sft_t2t_mini.jsonl`（本地） | general + tool | 90.6 万条 | CC-BY-NC-4.0 | `conversations` |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | code SFT | 75,197 | MIT | `lang`, `problem`, `solution` |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | code SFT | 50,661 | ODC-By | `instruction`, `response` |
| `m-a-p/CodeFeedback-Filtered-Instruction` | code SFT | 156,526 | Apache-2.0 | `query`, `answer`, `lang` |
| `google-research-datasets/mbpp` | code SFT/GRPO | 374 / 120 | CC-BY-4.0 | `text`, `code`, `test_list` |
| `open-r1/verifiable-coding-problems-python_decontaminated-tested` | code GRPO | 15,068 | 核对 | `problem`, `verification_info.test_cases` |
| `BAAI/TACO` | code GRPO | 25,443 | Apache-2.0 | `question`, `input_output`, `difficulty` |
| `Salesforce/xlam-function-calling-60k` | tool SFT/GRPO | 60k | CC-BY-4.0 | `query`, `tools`, `answers` |
| `Team-ACE/ToolACE` | tool SFT | 11,300 | Apache-2.0 | `system`, `conversations` |
| `glaiveai/glaive-function-calling-v2` | tool SFT | 112,960 | Apache-2.0 | `system`, `chat` |
| `nvidia/When2Call` | tool GRPO | 15k sft | CC-BY-4.0 | 调用决策 |
| `HuggingFaceFW/fineweb-edu` | pretrain | 10BT 子集 | ODC-By | `text`, `int_score` |
| `opencsg/chinese-fineweb-edu` | pretrain | 1.2M 行 | Apache-2.0 | `text` |
| `HuggingFaceTB/smollm-corpus` | pretrain | 39.1M docs | ODC-By | `text` |
| `codeparrot/codeparrot-clean` | pretrain code | — | permissive | `content`, `license`, `autogenerated` |
| `HuggingFaceTB/finemath` | pretrain math | 6.7M（4plus） | ODC-By | `text`, `int_score` |
