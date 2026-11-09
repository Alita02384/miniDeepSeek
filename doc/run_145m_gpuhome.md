# 145M 预训练上线流程（Mac 打包 → gpuhome 实例 → 切 GPU 开训）

> 适用场景：本地 Mac 做 tokenization，数据盘实例先无卡准备，之后关机切 GPU 开训。
> 关键约定：**代码 / 数据 / checkpoint 放数据盘 `/root/rivermind-data`；
> Python 环境放系统盘 `/opt/mds-venv`，在仓库里软链成 `.venv`。**
>
> 每步 optimizer step 恒为 **262,144 token**，与卡数无关（见 §8）。
> 相关文档：`doc/data_plan_145m.md`（数据来源）、`doc/hardware_budget.md`（显存/吞吐预算）。

---

## 0. 目录与环境约定

| 位置 | 放什么 | 说明 |
|---|---|---|
| `/root/rivermind-data/miniDeepSeek` | 代码（含 `.git`） | 数据盘，可扩容 |
| `/root/rivermind-data/miniDeepSeek/data` | JSONL + `tokenization/` 下的 packs | 数据盘 |
| `/root/rivermind-data/miniDeepSeek/checkpoints` | 训练产物 | 数据盘 |
| `/opt/mds-venv` | Python 环境 | 系统盘，30 GB 足够 |

```bash
# 本文所有命令里的连接参数，先 export 一次
export HOST=hn01-ssh.gpuhome.cc
export PORT=31185
```

> 服务器**连不上 GitHub**，代码一律用 rsync 同步（见 §4）。数据也不能从 ModelScope 拉，
> 全部从 Mac 直接 rsync。

---

## 1. 本机 Mac：tokenize（9 个训练 pack + 9 个验证 pack，约 45–75 分钟）

```bash
cd ~/Workspace/miniDeepSeek/miniDeepSeek
mkdir -p data/tokenization logs

pack () {
  local d=$1 s=$2 skip=$3 cap=$4 tag="$(( $2 / 1024 ))k"
  .venv/bin/python scripts/pack_domain.py \
    --input "data/pretrain_${d}.jsonl" --seq-len "$s" \
    --skip-tokens "$skip" --max-tokens "$cap" --domain "$d" \
    --out "data/tokenization/pretrain_${d}_${tag}.npz" \
    > "logs/pack_${d}_${tag}.log" 2>&1
}

pack general 2048 1000000 369000000 &
pack general 4096 370000000 1353000000 &
pack general 8192 1723000000 738000000 &
pack code 2048 1000000 67500000 &
pack code 4096 68500000 247500000 &
pack code 8192 316000000 135000000 &
pack tool 2048 1000000 13000000 &
pack tool 4096 14000000 48000000 &
pack tool 8192 62000000 26000000 &
wait

# 验证集：每个域从头取 1M token（训练 pack 从 skip 之后开始，两者不重叠）。
# 训练配置的 val_sources 在启动时全部加载，缺了会直接 FileNotFoundError。
pack_val () {
  local d=$1 s=$2 tag="$(( $2 / 1024 ))k"
  .venv/bin/python scripts/pack_domain.py \
    --input "data/pretrain_${d}.jsonl" --seq-len "$s" \
    --skip-tokens 0 --max-tokens 1000000 --domain "$d" \
    --out "data/tokenization/val_${d}_${tag}.npz" \
    > "logs/pack_val_${d}_${tag}.log" 2>&1
}
for d in general code tool; do
  for s in 2048 4096 8192; do pack_val "$d" "$s" & done
done
wait
```

进度：`tail -f logs/pack_general_8k.log`。
打包期间**不要重复执行**（会两个进程写同一批文件）。

验收（18 个 `.json` 全部存在，token 数对得上）：

```bash
.venv/bin/python - <<'PY'
import glob, json
paths = sorted(glob.glob("data/tokenization/pretrain_*_*.json")
               + glob.glob("data/tokenization/val_*_*.json"))
for p in paths:
    m = json.load(open(p))
    print(f"{p:58s} {m['tokens']/1e6:8.1f}M  {m['packs']} packs")
assert len(paths) == 18, f"expected 18 packs, found {len(paths)}"
PY
```

预期训练集：general 369 / 1353 / 738M，code 67.5 / 247.5 / **124.9M**，tool 13 / 48 / 26M；
验证集 9 个各 1.0M（general/code/tool × 2K/4K/8K）。

> `code_8k` 比 cap（135M）少约 10M：`pretrain_code.jsonl` 在 8K 阶段被用完。
> 8K 阶段按 135M 规划配额（515 步 × 8 pack = 4120 pack），实际可用 4298 pack
> （17193/4 分片），**按 pack 数算仍够一轮，不会重放**；只是 code 在 8K 阶段实际
> 消耗比规划少 ~7%。

---

## 2. 本机 Mac：压缩 packs（约 3–5 分钟）

27 GB 的 int32 数组直接传太慢，先压（实测 **3.4 GB**；token id 高字节全 0，zstd
约 7×）。验证 pack 只有 78 MB，单独打一个包，免得为了加验证集重传 3.4 GB：

```bash
cd ~/Workspace/miniDeepSeek/miniDeepSeek
tar -cf packs.tar data/tokenization/pretrain_*
zstd -3 -T0 packs.tar -o packs.tar.zst
rm packs.tar
ls -lh packs.tar.zst

tar -cf val_packs.tar data/tokenization/val_*
zstd -3 -T0 val_packs.tar -o val_packs.tar.zst
rm val_packs.tar
ls -lh val_packs.tar.zst        # 约 8 MB
```

> 可选保险：把预训练 JSONL 也压一份放着（~3.5 GB），需要重建 packs 时再解压。
> ```bash
> tar -cf pretrain_jsonl.tar data/pretrain_general.jsonl data/pretrain_code.jsonl data/pretrain_tool.jsonl
> zstd -3 -T0 pretrain_jsonl.tar -o pretrain_jsonl.tar.zst
> rm pretrain_jsonl.tar
> ```

---

## 3. 新实例：检查 + 建目录

```bash
ssh -p "$PORT" root@"$HOST"
df -h / /root/rivermind-data
nproc
free -g
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
mkdir -p /root/rivermind-data/miniDeepSeek/{data,logs}
```

- 数据盘建议 ≥ 80 GB：packs 27 G + 非预训练 JSONL 2.4 G + ckpt ~6 G；
- 无卡启动时 `nvidia-smi` 不存在、`torch.cuda.is_available()` 为 False 都正常。

---

## 4. 传代码（几秒）

```bash
cd /Users/cjh/Workspace/miniDeepSeek
rsync -avz --partial --progress -e "ssh -p $PORT" \
  --exclude '.venv' --exclude '.cache' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude 'data' --exclude 'download' --exclude 'dist' --exclude 'checkpoints' \
  --exclude 'runs' --exclude 'reports' --exclude 'internlm2-1_8b-reward' \
  --exclude '__pycache__' --exclude '*.zst' --exclude '*.tar' \
  miniDeepSeek/ root@"$HOST":/root/rivermind-data/miniDeepSeek/
```

`internlm2-1_8b-reward`（3.2 GB）是 GRPO 用的奖励模型，预训练不需要，已排除。

---

## 5. 传数据

### 方案 A（推荐，只传 packs + 非预训练 JSONL，约 2.4 GB）

训练只读 `data/tokenization/pretrain_*`，不读 `pretrain_*.jsonl`，所以可以不传它们：

```bash
cd /Users/cjh/Workspace/miniDeepSeek/miniDeepSeek/dist/minideepseek-data
rsync -avz --partial --progress -e "ssh -p $PORT" \
  --exclude 'pretrain_*.jsonl' \
  ./*.jsonl SHA256SUMS README.md \
  root@"$HOST":/root/rivermind-data/miniDeepSeek/data/
```

### 方案 B（全传，约 13 GB / 压缩后 ~4 GB）

```bash
cd /Users/cjh/Workspace/miniDeepSeek/miniDeepSeek/dist/minideepseek-data
rsync -avz --partial --progress -e "ssh -p $PORT" \
  ./*.jsonl SHA256SUMS README.md \
  root@"$HOST":/root/rivermind-data/miniDeepSeek/data/
```

### 传 packs（两种方案都要）

```bash
cd /Users/cjh/Workspace/miniDeepSeek/miniDeepSeek
rsync -av --partial --progress -e "ssh -p $PORT" \
  packs.tar.zst root@"$HOST":/root/rivermind-data/miniDeepSeek/

# 验证 pack（约 8 MB，几秒）
rsync -av --partial --progress -e "ssh -p $PORT" \
  val_packs.tar.zst root@"$HOST":/root/rivermind-data/miniDeepSeek/

# 可选：JSONL 保险包
# rsync -av --partial --progress -e "ssh -p $PORT" \
#   pretrain_jsonl.tar.zst root@"$HOST":/root/rivermind-data/miniDeepSeek/
```

> packs 已经压缩过，这一步**不要加 `-z`**。
> 如果 `packs.tar.zst` 放在 `dist/` 下（本机现状就是这样），把源路径换成
> `dist/packs.tar.zst`，目标路径不变；`val_packs.tar.zst` 同理。

---

## 6. 服务器：解压 + 校验

```bash
cd /root/rivermind-data/miniDeepSeek
[ -f packs.tar.zst ] && tar -I zstd -xf packs.tar.zst && rm packs.tar.zst
[ -f val_packs.tar.zst ] && tar -I zstd -xf val_packs.tar.zst && rm val_packs.tar.zst

ls data/tokenization/pretrain_*.json | wc -l   # 9
ls data/tokenization/val_*.json | wc -l        # 9
(cd data && sha256sum -c SHA256SUMS --ignore-missing)
df -h /root/rivermind-data
```

packs 的 token 数必须和 §1 的验收结果一致；不一致就停下来重传。
**缺 `val_*.json` 时训练会在第一个 stage 启动阶段就报
`FileNotFoundError: flat dataset not found: data/tokenization/val_general_2k.npz`**
（`val_sources` 在启动时全部加载，不是可选项）。

---

## 7. 服务器：装环境（系统盘 + 软链）

```bash
python3 -m venv --system-site-packages /opt/mds-venv
cd /root/rivermind-data/miniDeepSeek
ln -sfn /opt/mds-venv .venv

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_NO_CACHE_DIR=1 .venv/bin/pip install -U pip
PIP_NO_CACHE_DIR=1 .venv/bin/pip install -e . --no-deps
PIP_NO_CACHE_DIR=1 .venv/bin/pip install \
  safetensors tokenizers jinja2 pyyaml pytest ruff \
  transformers sentencepiece protobuf

.venv/bin/python -m pytest -q
```

- 用镜像自带的 CUDA torch（`--system-site-packages`），**不要**跑 `scripts/server_setup.sh`（会重复装 2.5 GB cu124 torch）；
- 这种 venv **没有 `.venv/bin/torchrun`**：`--system-site-packages` 只共享
  site-packages，console script 留在系统 bin 目录。多卡一律用
  `.venv/bin/python -m torch.distributed.run`（`scripts/train_*.sh` 已按此改好），
  可用 `ls .venv/bin/torchrun` 复核（应当不存在）；
- 若 `torch.version.cuda` 为空或导入失败，再补：
  ```bash
  PIP_NO_CACHE_DIR=1 .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```

---

## 8. 卡数自适应 + 启动训练

### 8.1 为什么 262k 不会变

训练器里 `batch_tokens` 是**全局 micro-batch**，按卡数均分；每个 rank 至少取一个 pack：

```text
packs_per_rank = max(1, per_rank_tokens // seq_len)
micro_global   = gpus × packs_per_rank × seq_len
grad_accum     = 262,144 // micro_global
tokens/step    = micro_global × grad_accum = 262,144   ← 恒定
```

| 卡数 | 2K stage（packs/rank=2） | 4K stage（packs/rank=1） | 8K stage（packs/rank=1，强制 8192） |
|---:|---|---|---|
| 1 | micro 4096 / accum 64 | 4096 / 64 | 8192 / 32 |
| 2 | 8192 / 32 | 8192 / 32 | 16384 / 16 |
| 4 | 16384 / 16 | 16384 / 16 | 32768 / **8** |
| 8 | 32768 / 8 | 32768 / 8 | 65536 / 4 |

> 8K 阶段的 `packs_per_rank` 因为 `max(1, 4096//8192)` 被抬成 1，全局 micro-batch
> 是 4×8192=32768，所以 grad_accum 必须是 **8** 而不是 16；否则每步是 524,288 token，
> 8K 阶段会吃掉 1.8B 而不是 0.9B。
>
> `grad_accum` 不同不影响数学：梯度累积按 target token 数加权
> （`train/trainer.py`，commit `fade758`），总步数仍是 11,444。

### 8.2 自动生成配置

```bash
cd /root/rivermind-data/miniDeepSeek

# 先看表（不写文件）
.venv/bin/python scripts/autoconfig_145m.py --dry-run

# 按当前卡数生成 configs/pretrain/generated/curriculum_145m_g{N}.yaml
.venv/bin/python scripts/autoconfig_145m.py
```

脚本会打印三阶段对照表和一行启动命令。卡数不是 1/2/4/8/16 时（比如 3 张）会直接报错，
提示换成受支持的卡数——因为 262,144 除不尽。

### 8.3 切 GPU 后：冒烟

```bash
GPUS=$(.venv/bin/python -c "import torch; print(torch.cuda.device_count())")
echo "gpus=$GPUS"
nvidia-smi

# 单卡显存/吞吐：4K 峰值应约 12.3 GiB
.venv/bin/python scripts/benchmark.py --flavor mini_145m_a60m \
  --seq-len 4096 --batch-size 1 --steps 5 --warmup 1 --device cuda

# 多卡 NCCL + FSDP2
export NCCL_P2P_DISABLE=1
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=$GPUS \
  scripts/benchmark.py --flavor mini_145m_a60m --seq-len 4096 --batch-size 1 \
  --steps 5 --warmup 1 --distributed fsdp2

# 8K 阶段预检：8K + 激活重算的真实峰值从没在 CUDA 上测过（估计 12–15 GiB），
# 在这里花 1 分钟，好过训练到 2.1B token 才发现第 3 阶段 OOM。
.venv/bin/python scripts/measure_activation_budget.py --device cuda \
  --flavor mini_145m_a60m --seq-len 4096 8192 --activation-checkpointing
```

> 该脚本只跑前向（saved-tensor 峰值，`index_topk` 按 8K 自动取 2048），
> backward + 优化器状态还要再加约 1–2 GiB。8K 那行 `cuda_peak_reserved_mib`
> > 20 GiB 就先按 §9 把 8K 阶段的 `index_topk` 降到 1024，再开长跑。

### 8.4 开训

```bash
cd /root/rivermind-data/miniDeepSeek
GPUS=$(.venv/bin/python -c "import torch; print(torch.cuda.device_count())")
.venv/bin/python scripts/autoconfig_145m.py --gpus "$GPUS"
export NCCL_P2P_DISABLE=1

nohup env NPROC_PER_NODE=$GPUS bash scripts/train_curriculum.sh \
  "configs/pretrain/generated/curriculum_145m_g${GPUS}.yaml" \
  > logs/curriculum_145m.log 2>&1 &

# 多卡时脚本内部是 `.venv/bin/python -m torch.distributed.run`（venv 里没有 torchrun 脚本）
```

---

## 9. 监控 / 续跑 / 常见问题

```bash
tail -f logs/curriculum_145m.log
watch -n 5 nvidia-smi
tail -f checkpoints/curriculum_145m_3b/stage_00_seq2048/metrics.jsonl
```

- **可视化**：`train.report_every: 200` 已打开，训练会每 200 步把当前 metrics.jsonl
  渲染成 `<stage_dir>/report.html`（内联 SVG，无外部依赖）。它读的是磁盘上的
  metrics.jsonl，所以续跑不会丢中断前的曲线。旧/已结束的 run 可手动补：

  ```bash
  .venv/bin/python scripts/plot_training.py \
    checkpoints/curriculum_145m_3b/stage_00_seq2048/metrics.jsonl \
    --out checkpoints/curriculum_145m_3b/stage_00_seq2048/report.html
  ```

  每个 stage 各自一个 out_dir，所以是三份 report.html。要一张跨阶段的图，把三个
  jsonl 直接按阶段顺序拼起来再渲染一次即可（step / tokens_seen 本来就是跨阶段累加的）：

  ```bash
  cat checkpoints/curriculum_145m_3b/stage_0*/metrics.jsonl > /tmp/all_stages.jsonl
  .venv/bin/python scripts/plot_training.py /tmp/all_stages.jsonl --out /tmp/all_stages.html
  ```

- **续跑**：重跑 §8.4 同一条命令；已完成的 stage 会跳过，当前 stage 从最新 checkpoint 恢复，
  并按 checkpoint 里的 `data_batches` **精确跳过已消费的 micro-batch**（同一卡数下数据顺序
  与中断前完全一致，不会从 stage 开头重放）；启动时会有约 1 分钟在回放数据，属正常；
- **换卡数续跑**：重新生成配置（`autoconfig_145m.py --gpus N`），`out_dir` 不变即可续；
  但每个 micro-batch 的 token 数变了，`data_batches` 的跳过量只是近似（可能偏移几百 M
  token），要求严格对齐就重跑该 stage；
- **NCCL 卡住**：消费卡保持 `NCCL_P2P_DISABLE=1`；
- **8K OOM**：先看 §8.3 的 `measure_activation_budget` 结果；确认
  `activation_checkpointing: true`（8K 阶段已开）。仍 OOM 就把
  `configs/pretrain/curriculum_145m_4x4090.yaml` 里 8K 阶段的 `index_topk`
  从 2048 改成 1024，再重跑 `autoconfig_145m.py` 重新生成。2K/4K 的
  2.1B token（约 70%）不受影响；OOM 只会在第 3 阶段开头发生，
  前两个阶段的 `final/` 已经落盘；
- **loss NaN**：预训练先把 lr 降到 3e-4 重跑；
- **磁盘不够**：方案 A 已省 13 G；再不够就删 `data/pretrain_*.jsonl` 或调小 `keep_last_ckpt`。

---

## 10. 时间与磁盘预算

| 步骤 | 时间 | 磁盘（数据盘） |
|---|---|---|
| 本地 tokenize（18 并行） | 45–75 min | 本地 +27 GB |
| 压缩 packs | 3–5 min | 本地 +3.4 GB（val +8 MB） |
| 传代码 | < 1 min | — |
| 传数据（方案 A） | 10–20 min | +2.4 GB |
| 传 packs | ~30 min（3.4 GB @ 2 MB/s） | +3.4 GB → 解压后 27 GB |
| 传验证 packs | < 1 min（8 MB） | +78 MB |
| 装环境 | 10 min | 系统盘 ~0.5 GB |
| 冒烟 | 10 min | — |
| 预训练 3B（4×4090） | **6–9 h** | ckpt ~4–6 GB |

> `/root/rivermind-data` **释放实例即丢失**，训练期间不要点“释放”；重要 checkpoint
> 及时 rsync 回本地或传到对象存储。
