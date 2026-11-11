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

## 2. 一键流水线（145M 模板，按需使用）

> 当前 3B 预训练与 145M 后训练都以单阶段脚本为准；pipeline 是整理好的
> 145M 模板。`--dry-run` 会检查 pretrain/SFT 输入、GRPO resume、OPD data 和
> teacher checkpoint，缺一个就失败。
>
> **不要把它当成 3B 预训练的入口**；hb01 的 pretrain job 不要用 pipeline 重启。

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

- pretrain：分域 `pretrain_{general,code,tool}.jsonl`，tokenize 后实际 token 数写入
  `data/tokenization/pretrain_{domain}_{2k,4k,8k}.json`；pipeline 按
  2K/4K/8K = 15%/55%/30% 分配，并读取 base config 的 `grad_accum`（16）：
  `step = tokens / (batch_tokens_final × grad_accum)`，不写死 6B。
- SFT general：`sft_general.jsonl`（约 6 亿 token），默认 1 epoch。
- SFT tool：`sft_tool.jsonl`（tool 子集约 5400 万 + general 约 1360 万 input token）。
- GRPO tool：`grpo_tool_train.jsonl`，约 3.5 万条多轮工具 prompt；每 step
  4 组 × 16 prompt，rollout 成本比 general 高。
- OPD：2-teacher 用 `opd_prompts_mixed.jsonl`（general 17k + tool 20k），
  3-teacher 用 `opd_prompts_three.jsonl`（+ code 3.1k）；teacher 为
  `checkpoints/grpo_{general,tool,code}_145m/final`。

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
  reward_model_path: internlm2-1_8b-reward
```

把 reward model 放在代码根目录下（`internlm2-1_8b-reward`，约 3.2 GB）。没有它时
general GRPO 只有 shaping、没有主信号；tool 分支仍可用工具+GT reward，但
无工具分支会退化。下载与许可按 InternLM2 官方仓库执行。

### 5.1 服务器上下载（HF 连不上）

服务器连不上 `huggingface.co` 和 `github.com`；`hf-mirror.com` 的小文件能拿，
大文件会 307 到 `api/resolve-cache` 之后卡死。实测可用的只有 ModelScope 的
repo 文件 API：

```bash
DEST=/root/rivermind-data/miniDeepSeek/internlm2-1_8b-reward
BASE='https://www.modelscope.cn/api/v1/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward/repo?Revision=master&FilePath='
mkdir -p "$DEST" && cd "$DEST"
: > urls.txt
for f in config.json configuration_internlm2.py modeling_internlm2.py \
         model.safetensors.index.json special_tokens_map.json \
         tokenization_internlm2.py tokenization_internlm2_fast.py \
         tokenizer.model tokenizer_config.json \
         model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  printf '%s%s\n\tout=%s\n' "$BASE" "$f" "$f" >> urls.txt
done
aria2c -i urls.txt -x8 -s8 -j2 -c --auto-file-renaming=false
```

3.4 GB，8 连接实测约 15 MB/s（2–3 分钟；单连接只有 7 MB/s）。
`reward_bench_results/`、`README.md`、`.gitattributes`、`configuration.json`
训练用不到，可不下。

### 5.2 校验

```bash
cd /root/rivermind-data/miniDeepSeek/internlm2-1_8b-reward
sha256sum model-00001-of-00002.safetensors model-00002-of-00002.safetensors config.json
# b0cbcdad5899d93dd653a4bf7fdf65e1d9668aaf61c0d5d41be1529e96b902ea  model-00001 (1,981,392,544 B)
# 6421a9ec59b5e7b127c1b99fb80b88e5e013a9f2769a2f3000b8c99c99ad4630  model-00002 (1,417,790,344 B)
# a50c39e11d22fb4250c2fa82724dbd59c8b2b1885fa85ca4565b4f710dc0c957  config.json
du -sh .
```

### 5.3 依赖（`[hf]` 不够用）

只装 `pip install -e '.[hf]'` 会栽两次，都是 2026-11 在 gpuhome 上实测到的：

- InternLM2 的 `trust_remote_code` modeling 文件 `import einops`，`[hf]` 里原来没有，
  报 `ImportError: This modeling file requires ... einops`；
- venv 是 `--system-site-packages`，pip 认为镜像自带的 protobuf 7.35.1 已满足
  `protobuf>=4` 就不装，新 sentencepiece 的 gencode 7.36.1 比它新，报
  `VersionError: ... gencode 7.36.1 runtime 7.35.1`。

`scripts/setup_env_gpu.sh` 现在把两件事都处理了（`[hf]` 带 `einops>=0.8`，脚本结尾
`pip install --ignore-installed 'protobuf>=5'` 并 import 自检）。手工装的话：

```bash
.venv/bin/pip install -e '.[dev,hf]'
.venv/bin/pip install --ignore-installed 'protobuf>=5'
```

### 5.4 打分自检

本地如果 `~/.cache` 不可写，运行前设：

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python scripts/check_reward_model.py --path internlm2-1_8b-reward
```

期望输出形如 `.../internlm2-1_8b-reward reward=-0.812751`（CPU fp32；fp16 会差
~0.002）。不带 GPU 的实例上约 5 s 加载、1–2 s 打分。

## 6. reward model 资源

实测（CPU/M5，16GB）：加载后最大 RSS 约 **3.15GB**；单条短回答打分约
**2–4s**。模型参数 **1.70B**：

| 精度 | 权重 | 说明 |
|---|---:|---|
| fp16/bf16 | **3.4GB** | 配置默认 |
| fp32 | 6.8GB | 不建议 |

单次 forward 约 `2 × 1.7B × tokens` FLOPs：

| 序列 token | TFLOP |
|---:|---:|
| 256 | 0.87 |
| 512 | 1.74 |
| 1024 | 3.48 |

24GB 卡上按 fp16 估算，每个 rank 约 **5–6GB**（权重 3.4GB + 激活/CUDA
上下文）。注意当前实现是**每个 rank 各加载一份 RM**，4 卡就是 4 份；
actor 训练状态另算。general GRPO 每 step 约 6×32=192 次打分，按 4090
10–20% MFU 估算约 **20–60s/step 仅 RM**；CPU 按 2–4s/条则要 6–13min/step，
不可训练。tool 分支只在无工具调用时用 RM，调用数更少。

如果要省显存/时间：把 RM 放独立卡并在 rank0 汇总打分，或把 `get_score`
改成 `get_scores` 批量打分（当前逐条调用）。

## 7. 断点续训 / 常见问题

**pretrain 课程（2K→4K→8K）支持断点续训**，三层的语义是：

| 层 | 机制 |
|---|---|
| pipeline | 已有 `final/meta.json` 的 stage 直接跳过（`--force` 可强制重跑） |
| curriculum | 每个 stage 优先续自己的 `final` 或最新 `step_*`；`until_step` 是累计值，跑满即停 |
| trainer | 每 `train.ckpt_every` 步写 `step_NNNNNN/`（`keep_last_ckpt: 3`）；收到 SIGTERM/SIGINT 时**跑完当前 step** 再写 `step_NNNNNN` 退出 |

所以中断后**重跑同一条命令**即可：

```bash
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh configs/pipeline/minimind_145m.yaml
# 只重跑 pretrain：
NPROC_PER_NODE=4 bash scripts/train_curriculum.sh configs/pretrain/curriculum_145m_4x4090.yaml
```

几个必须知道的点：

- **`final` 只表示“这一阶段跑完了”**。被信号打断的阶段只写 `step_NNNNNN`，
  不会写 `final`，否则 pipeline 会把没跑完的 8K 阶段当成完成、直接进 SFT。
- **硬杀（OOM kill、机器掉电）最多回退 `ckpt_every` 步**。当前 200 步 ×
  262,144 token ≈ 5,200 万 token ≈ 4×4090 上 6–9 分钟。想更细就调小
  `ckpt_every`（每个 checkpoint 实测约 1.1 GB：模型 557 MB + Muon momentum，
  `keep_last_ckpt: 3` 约占 3.5 GB）。
- **单 stage 直接跑**（`train_pretrain.sh configs/pretrain/pretrain_145m.yaml`）
  打开 `train.auto_resume: true` 后同样会自动续，不用手写 `train.resume`。
- **多卡要所有 rank 都收到信号**（torchrun 前台 Ctrl-C、Slurm
  `--signal=B:TERM`、k8s preStop 都满足）。如果只是单卡 Xid/OOM 掉线，
  其他 rank 会卡在集合通信里：先杀进程，再重跑。
- **后台跑时别只 kill pipeline 的 PID**：`run_pipeline.sh` 用 `subprocess.run`
  起训练，SIGTERM 只发给父进程会留下孤儿训练进程。要么前台 Ctrl-C（信号发给
  整个进程组），要么 `pkill -TERM -f train_curriculum.py` / `kill -- -<PGID>`。
- **恢复后 step / LR / index_top-k / MTP 权重全部接着走**（都按全局 step 计算）；
  数据顺序会从该阶段 shard 的起点重排（`resume_data: false`），不重复 token
  也不跳 token，但同一 step 看到的数据与未中断时不同。
- **SFT 也走同一个 trainer**，信号续训同样生效；GRPO/OPD 是各自的 trainer，
  中断后从配置里的 `resume`（SFT/base checkpoint）重跑该 stage。
- **OOM**：先看是 tokenize、pretrain 8K 还是 tool rollout。tool rollout 的
  prompt 平均约 1.2k token，`max_turns=3`，必要时降 `prompts_per_step`。
- **NCCL 卡住**：4 卡消费级主板先 `export NCCL_P2P_DISABLE=1`。
- **loss NaN**：pretrain 先降 lr 到 3e-4 重跑；GRPO 看 reward 方差，
  全零/全同会退化为 0 advantage。
- **reward model OOM**：1.8B fp16 约 3.6 GB；与 145M actor 同卡一般够，
  不够就放 CPU（很慢）或换 HTTP judge。

## 8. code 专家补齐后

code 配置和 pipeline 槽位已经写好，只差数据：

1. 准备 code SFT（conversations）与 code GRPO（`prompt` + `rubric.tests`）；
2. 生成 code 数据：

```bash
.venv/bin/python -m minideepseek.data.build \
  --minimind-root ../minimind/dataset --out data \
  --code-sft /path/to/code_sft.jsonl \
  --code-grpo /path/to/code_grpo.jsonl
```

3. 跑 3-teacher pipeline：

```bash
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh \
  configs/pipeline/minimind_3teacher.yaml
```

`configs/grpo/code.yaml` 里的 `code_pass_reward`、`code_compile_bonus`、
`code_syntax_penalty`、`code_format_bonus`、`code_timeout_s` 是首版 reward
权重，可直接按数据调整；`data/opd_prompts_mixed.jsonl` 会自动包含 code。
