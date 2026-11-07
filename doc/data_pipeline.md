# 数据与一键训练流水线

> 当前路线（2026-11-07）：**general / tool-use / code** 三个 specialist。
> code 数据与 GRPO reward 暂缺；当前可训练 general 与 tool-use，OPD 先跑
> 2-teacher，code 数据补齐后再切 3-teacher。

## 1. 目录

`data/` 只放平铺的、可直接读取的 JSONL；tokenization 单独放
`data/tokenization/`，不进入上传包。

```text
data/
├── pretrain_t2t.jsonl
├── sft_general.jsonl
├── sft_tool.jsonl
├── grpo_general_train.jsonl     eval_general.jsonl
├── grpo_tool_train.jsonl        eval_tool.jsonl
├── opd_prompts_general.jsonl
├── opd_prompts_tool.jsonl
├── opd_prompts_mixed.jsonl
├── manifests/{specialists.yaml,stats.json}
└── tokenization/                # 服务器生成：pretrain_{2k,4k,8k}.npz, sft_{general,tool}.npz
```

code 相关文件（`sft_code.jsonl`、`grpo_code_train.jsonl`、
`opd_prompts_code.jsonl`）当前不存在；补齐后由同一 build 脚本加入。

## 2. 生成

```bash
.venv/bin/python -m minideepseek.data.build \
  --minimind-root ../minimind/dataset --out data
```

- `pretrain_t2t.jsonl`：从 `minimind/dataset/pretrain_t2t.jsonl` 复制/硬链；
- `sft_general.jsonl`：从 `sft_t2t_mini.jsonl` 复制；
- `sft_tool.jsonl`：`sft_t2t_mini` 中带 `tools`/`tool_calls` 的子集 + 25% general；
- `grpo_general_train/eval`：`rlaif.jsonl` 的 prompt，保留多轮历史，去掉末尾空 assistant；
- `grpo_tool_train/eval`：`agent_rl.jsonl` + `agent_rl_math.jsonl` 的 tools+gt 子集，多轮；
- `opd_prompts_*`：从 GRPO train prompt 派生，按 teacher_key 均衡。

`minimind/dataset` 只读；不要在那边改文件。

## 3. Tokenization（训练服务器）

```bash
.venv/bin/python -m minideepseek.data.tokenize \
  --data-dir data --out-dir data/tokenization
```

- 2K/4K/8K 各一份 packed 数据集；完整 2.3B 语料约 20 GB/份。
- `--smoke --max-records 200` 可在 CPU 上验证。
- 内置 MinHash 默认关闭；2.3B 语料上它会占数十 GB 内存，去重需要单独实现。

## 4. 上传 / 下载

```bash
bash scripts/package_dataset.sh dist/minideepseek-data
# 将 dist/minideepseek-data 上传到 ModelScope 数据集仓库
```

服务器：

```bash
DATA_REPO_URL=https://www.modelscope.cn/datasets/<ns>/<name>.git \
  bash scripts/server_setup.sh
```

`server_setup.sh` 安装 CUDA torch、项目、`.[hf]`，下载数据并跑 CPU 测试。

## 5. 一键训练

```bash
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh \
  configs/pipeline/minimind_145m.yaml --dry-run
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh \
  configs/pipeline/minimind_145m.yaml
```

阶段：tokenize → pretrain 2K→4K→8K（按实际 token 数自动算 step）→
SFT general/tool → GRPO general/tool → 2-teacher OPD → eval → HF export。

`--from/--to` 只跑子阶段；已有 `final/meta.json` 的 stage 会跳过，`--force` 重跑。

## 6. Reward 与参考实现

- math/code：可验证 reward（exact match / 单测）；
- general：MiniMind 的 `internlm2-1_8b-reward` + 长度/思考/重复 shaping；
- tool：MiniMind `train_agent.py` 的工具对齐 + GT + 未完成 + shaping；
- 多轮 tool rollout：`rl/agent_rollout.py`，observation 不参与 loss；
- 来源与改动见 `doc/sources.md`，决策见 `doc/decisions.md`。
