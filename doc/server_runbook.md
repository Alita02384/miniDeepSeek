# 服务器运行手册：general + tool-use 专家 → 2-teacher OPD（4×24 GB）

> 当前路线：**general / tool-use / code** 三个 specialist。code 数据与 GRPO
> reward 暂缺，先训练 general + tool-use，OPD 跑 2-teacher；code 补齐后加
> `sft_code` / `grpo_code` 并切 3-teacher。
>
> 数据是平铺 JSONL，tokenization 在服务器生成；本地只打包上传 JSONL。

---

## 0. 本地：生成数据包

```bash
cd miniDeepSeek
./scripts/setup_env.sh
bash scripts/build_data.sh          # minimind -> data/*.jsonl
bash scripts/package_dataset.sh dist/minideepseek-data
```

`data/tokenization/` 不打包；`dist/minideepseek-data` 里只有 JSONL、manifests、
`SHA256SUMS` 和 dataset card。上传到 ModelScope 数据集仓库。

## 1. 服务器：环境 + 下载

```bash
git clone <repo> miniDeepSeek && cd miniDeepSeek
DATA_REPO_URL=https://www.modelscope.cn/datasets/<ns>/<name>.git \
  bash scripts/server_setup.sh
```

脚本会：建 venv、装 CUDA torch（默认 cu124，可 `TORCH_INDEX_URL=...` 覆盖）、
装 `.[hf]`、下载数据、校验 `SHA256SUMS`、跑 CPU 测试。

判断成功：

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
PY
ls data/pretrain_t2t.jsonl data/sft_general.jsonl data/grpo_tool_train.jsonl
```

## 2. 一键流水线

```bash
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh \
  configs/pipeline/minimind_145m.yaml --dry-run

NPROC_PER_NODE=4 bash scripts/run_pipeline.sh \
  configs/pipeline/minimind_145m.yaml
```

阶段：

```text
tokenize pretrain 2K/4K/8K + sft_general/sft_tool
→ pretrain curriculum（全部可用 token，step 自动算）
→ SFT general / SFT tool
→ GRPO general（rlaif + 1.8B reward model）
→ GRPO tool（agent_rl* 多轮工具 rollout）
→ OPD 2-teacher（general + tool）
→ eval general / eval tool
→ export HF
```

已完成的 stage 会跳过；`--from sft_tool`、`--to grpo_tool` 只跑子段；`--force`
重跑。运行状态在 `runs/minimind_145m/state.json`，日志在 `runs/.../logs/`。

## 3. 数据与预算

- pretrain：`pretrain_t2t.jsonl`，tokenize 后实际 token 数写入
  `data/tokenization/pretrain_*.json`；pipeline 按 2K/4K/8K = 15%/55%/30% 分配，
  `step = tokens / (batch_tokens_final × grad_accum)`，不写死 6B。
- SFT general：`sft_general.jsonl`（约 6 亿 token），默认 1 epoch。
- SFT tool：`sft_tool.jsonl`（tool 子集约 5400 万 + general 约 1360 万 input token）。
- GRPO tool：`grpo_tool_train.jsonl`，约 3.5 万条多轮工具 prompt；每 step
  4 组 × 16 prompt，rollout 成本比 general 高。
- OPD：`opd_prompts_mixed.jsonl`，general 17k + tool 20k；teacher 为
  `checkpoints/grpo_general/final`、`checkpoints/grpo_tool/final`。

磁盘：2K/4K/8K 三份 tokenized 约 20 GB/份（完整语料）；checkpoint 另计。
内存：tokenize 串行跑，单进程约 20 GB 量级；不要三份并行。

## 4. 上卡前检查

```bash
.venv/bin/python scripts/measure_activation_budget.py --device cuda \
  --flavor mini_145m_a60m --seq-len 2048 4096
.venv/bin/python scripts/measure_activation_budget.py --device cuda \
  --flavor mini_145m_a60m --seq-len 8192 --activation-checkpointing
```

- 2K/4K 默认不开重算；8K stage 在 curriculum config 里打开
  `activation_checkpointing: true`。
- 如果 8K OOM：先降 `index_topk`（2048→1024），再考虑去掉 8K stage；
  不要靠加卡解决激活。

## 5. Reward model

general/tool 的配置使用 MiniMind 的 `internlm2-1_8b-reward`：

```yaml
reward:
  kind: general   # 或 tool
  reward_model_path: ../internlm2-1_8b-reward
```

把 reward model 放在代码根同级目录（`../internlm2-1_8b-reward`）。没有它时
general GRPO 只有 shaping、没有主信号；tool 分支仍可用工具+GT reward，但
无工具分支会退化。下载与许可按 InternLM2 官方仓库执行。

本地如果 `~/.cache` 不可写，运行前设：

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python scripts/check_reward_model.py --path ../internlm2-1_8b-reward
```

## 6. 断点 / 常见问题

- **续跑**：再执行同一条 `run_pipeline.sh`；stage final 存在即跳过。
- **OOM**：先看是 tokenize、pretrain 8K 还是 tool rollout。tool rollout 的
  prompt 平均约 1.2k token，`max_turns=3`，必要时降 `prompts_per_step`。
- **NCCL 卡住**：4 卡消费级主板先 `export NCCL_P2P_DISABLE=1`。
- **loss NaN**：pretrain 先降 lr 到 3e-4 重跑；GRPO 看 reward 方差，
  全零/全同会退化为 0 advantage。
- **reward model OOM**：1.8B fp16 约 3.6 GB；与 145M actor 同卡一般够，
  不够就放 CPU（很慢）或换 HTTP judge。

## 7. code 专家补齐后

1. 把 code SFT/GRPO 数据放进 `data/`（`sft_code.jsonl`、
   `grpo_code_train.jsonl`）；
2. 新增 `configs/sft/code.yaml`、`configs/grpo/code.yaml`；
3. 在 pipeline 里加 `sft_code`、`grpo_code`，OPD 加 `code` teacher 与
   `opd_prompts_code.jsonl`；
4. 重跑 `opd`：`--from opd --force`。
