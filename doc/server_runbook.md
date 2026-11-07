# 服务器运行手册：145M MoE + 2K→4K→8K（4×4090/3090 24 GB）

从一台空 Linux 服务器到跑完一个预训练课程，按顺序执行。每一步都写了**怎么判断
成功**和**失败先看什么**。

约定：

- 代码根 `miniDeepSeek/`，数据都在 `data/` 下，配置里只写 `data/...` 相对路径；
- 单机 4 卡用 `torchrun --standalone`，不要 `--nnodes`；
- 显存按 24 GB 规划：2K/4K 不开重算，8K 开按 block 激活重算。

---

## 0. 本地打包上传（在你自己的机器上）

```bash
# 代码（不含 .venv / .git / 已生成的 tokenized 数据）
rsync -av --exclude=.venv --exclude=.git --exclude='data/tokenized*' \
  miniDeepSeek/ user@server:/data/pjm/miniDeepSeek/

# 语料：8.5M 文档、约 2.3B tokens，8.3 GB
rsync -av miniDeepSeek/data/raw/ user@server:/data/pjm/miniDeepSeek/data/raw/
```

`data/raw/pretrain_t2t.jsonl` 是训练语料，`pretrain_t2t_mini.jsonl` 只用来做
数据管线 smoke。

---

## 1. 环境

```bash
cd /data/pjm/miniDeepSeek
python3.12 -m venv .venv
.venv/bin/pip install -U pip
# CUDA 版 torch（cu124 对应 4090/3090 驱动 >= 550；按服务器驱动选 cu121/cu124/cu126）
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install safetensors tokenizers jinja2 numpy pyyaml pytest ruff
```

判断成功：

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

应看到 `4` 张卡、`bfloat16` 可用。**不要用 `./scripts/setup_env.sh`**：它走 `uv`
且不保证装 CUDA 版 torch。

先跑 CPU 单元测试确认代码完好（约 40 s，不占 GPU）：

```bash
.venv/bin/python -m pytest -q
```

---

## 2. 数据：2K / 4K / 8K

三个长度各建一份 packed 数据集，**并行跑**（每个进程约 12 GB 内存，合计 ~36 GB；
内存不足就串行）。

```bash
mkdir -p logs data/tokenized
for pair in "2048 2k" "4096 4k" "8192 8k"; do
  set -- $pair
  nohup .venv/bin/python -m minideepseek.data.prepare pretrain \
    --input data/raw/pretrain_t2t.jsonl \
    --output data/tokenized/pretrain_$2.npz \
    --seq-len $1 --max-tokens 1200000000 \
    > logs/prepare_$2.log 2>&1 &
done
wait
tail -n 20 logs/prepare_2k.log
```

- `--max-tokens 1200000000`：每个阶段最多取 12 亿 token（语料约 23 亿）。
  想多用/少用就改这个数，**同时改课程配置里的 `until_step`**（见 §4 的换算）。
- 文件名必须是 `pretrain_2k/4k/8k.npz`，课程配置按这个路径找数据。
- 实测速度约 7.5 万 token/s/进程（本机 10 核）；12 亿 token ≈ 4.5 h/阶段，
  三阶段并行 ≈ 4.5 h。磁盘：每阶段约 9 GB，合计约 27 GB。

判断成功：

```bash
.venv/bin/python - <<'PY'
import json, glob
for path in sorted(glob.glob("data/tokenized/pretrain_*.json")):
    meta = json.load(open(path))
    print(path, meta["seq_len"], meta["tokens"], meta["packs"])
PY
.venv/bin/python -m minideepseek.data.prepare manifest \
  --manifest data/manifests/pretrain_local.yaml
```

`tokens` 应该都在 1.2e9 量级，`packs` ≈ tokens / seq_len。manifest 要 `"ok": true`。

---

## 3. 上卡前的检查（5 分钟，能省掉几小时的重跑）

```bash
# 3.1 显存：4K 不开重算 ~14 GiB；8K 开重算应显著下降
.venv/bin/python scripts/measure_activation_budget.py --device cuda \
  --flavor mini_145m_a60m --seq-len 2048 4096
.venv/bin/python scripts/measure_activation_budget.py --device cuda \
  --flavor mini_145m_a60m --seq-len 8192 --activation-checkpointing

# 3.2 真实一步的 tok/s 与峰值显存（走 trainer 同一条路径）
NPROC_PER_NODE=4 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/benchmark.py --config configs/pretrain/pretrain_145m.yaml \
  --distributed fsdp2 --batch-size 1 --steps 5
```

参考值（本机 CPU 侧的 saved-tensor 实测，GPU 峰值会再高 10–20%）：

| 场景 | 145M 的 saved 激活 | 加参数/梯度/优化器后 |
|---|---|---|
| 4K，不重算 | 12.3 GiB | ~14 GiB |
| 4K，开重算 | 0.5 GiB | 峰值取决于单层重算，约 6 GiB |
| 8K，不开重算 | ~25 GiB | **OOM** |
| 8K，开重算 | ~1 GiB | 预估 12–15 GiB，需实测确认 |

如果 8K 仍然 OOM，按顺序降级：① 课程配置里 `index_topk: 2048` → `1024` 或
`512`；② 把 8K 阶段的 `batch_tokens` 保持 32768 但确认每卡只有 1 个 pack；
③ 去掉 8K 阶段（`until_step` 停在 4K）。**不要**试图靠加卡解决：激活不随卡数
切分。

---

## 4. 训练

```bash
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 24 GB 卡减少碎片
NPROC_PER_NODE=4 nohup ./scripts/train_curriculum.sh \
  configs/pretrain/curriculum_145m_4x4090.yaml \
  > logs/train_curriculum.log 2>&1 &
```

阶段与预算（`configs/pretrain/curriculum_145m_4x4090.yaml`）：

| 阶段 | seq | 累计 step | 每步全局 token | 阶段 token | 重算 |
|---|---|---|---|---|---|
| 2K | 2048 | 1400 | 16K×8 = 131072 | 0.18B | 关 |
| 4K | 4096 | 6400 | 16K×8 = 131072 | 0.66B | 关 |
| 8K | 8192 | 7800 | 32K×8 = 262144 | 0.37B | 开 |

`until_step` 是**累计**值：`step = 全局 token 预算 / (batch_tokens × grad_accum)`。
改了 `--max-tokens` 就按这个公式重算三行。

预估时间：4×4090 上约 **3–4 小时**（MFU 15–20%，8K 阶段开重算再慢 ~30%）。
3090 约慢 15%。

判断成功：

```bash
grep -E "stage|loss" logs/train_curriculum.log | tail -20
ls checkpoints/curriculum_145m/stage_*/final
```

日志里 8K 阶段的 loss 应该从 4K 的末端接上（跳到略高再降），不是重新从 10.4 开始。

---

## 5. 断点续跑 / 常见问题

- **续跑**：再执行 §4 同一条命令即可。课程 runner 会优先用每个 stage 自己的最新
  checkpoint（`stage_XX_seqY/step_*` 或 `final`），已经跑完的 stage 直接跳过。
- **OOM**：先看是哪个 stage；8K 按 §3 的降级顺序处理。2K/4K OOM 说明
  `batch_tokens` 被调大了（每卡必须 ≤ 2 个 pack）。
- **NCCL 卡住**：4 卡消费级主板常见，先 `export NCCL_P2P_DISABLE=1` 试一次；
  仍然挂就查 `nvidia-smi topo -m` 确认卡在同一 PCIe switch 下。
- **loss 变 NaN**：`train.loss_spike_*` 默认只记录不处理；小模型 0.6 学习率通常
  稳定，若出现先降到 3e-4 重跑。
- **数据准备太慢**：三阶段并行是最省事的加速；想再快就把 `--max-tokens` 降到
  6 亿（模型小，1.2B 与 0.6B 的差别主要体现在后期 loss）。

---

## 6. 这一步之后

- 课程结束后 `checkpoints/curriculum_145m/stage_02_seq8192/final` 是 base 模型；
  接着跑 SFT/GRPO/OPD（`configs/sft/*`、`configs/grpo/*`、`configs/opd/*`），
  记得把 `train.resume` 指到它。
- 想要 220M 档：换 `model_file: configs/model/mini_220m_a80m.yaml`，但 24 GB 上
  4K 需要重算、8K 需要 chunked attention（见 `doc/hardware_budget.md`）。
