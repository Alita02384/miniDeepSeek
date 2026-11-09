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

## 1. 本机 Mac：tokenize（9 个 pack 并行，约 40–70 分钟）

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
```

进度：`tail -f logs/pack_general_8k.log`。
打包期间**不要重复执行**（会两个进程写同一批文件）。

验收（9 个 `.json` 全部存在，token 数对得上）：

```bash
.venv/bin/python - <<'PY'
import glob, json
for p in sorted(glob.glob("data/tokenization/pretrain_*_*.json")):
    m = json.load(open(p))
    print(f"{p:58s} {m['tokens']/1e6:8.1f}M  {m['packs']} packs")
PY
```

预期：general 369 / 1353 / 738M，code 67.5 / 247.5 / **124.9M**，tool 13 / 48 / 26M。

> `code_8k` 比 cap（135M）少约 10M：`pretrain_code.jsonl` 在 8K 阶段被用完。
> mixer 在某个域耗尽时会从头重启该域（`data/mix.py`），所以 8K 阶段最后约 7%
> 的步数会重复约 10M code token，对 3B 总预算影响 < 0.4%。

---

## 2. 本机 Mac：压缩 packs（约 3–5 分钟）

27 GB 的 int32 数组直接传太慢，先压（约 8–9 GB）：

```bash
cd ~/Workspace/miniDeepSeek/miniDeepSeek
tar -cf packs.tar data/tokenization/pretrain_*
zstd -3 -T0 packs.tar -o packs.tar.zst
rm packs.tar
ls -lh packs.tar.zst
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
  --exclude '__pycache__' \
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

# 可选：JSONL 保险包
# rsync -av --partial --progress -e "ssh -p $PORT" \
#   pretrain_jsonl.tar.zst root@"$HOST":/root/rivermind-data/miniDeepSeek/
```

> packs 已经压缩过，这一步**不要加 `-z`**。

---

## 6. 服务器：解压 + 校验

```bash
cd /root/rivermind-data/miniDeepSeek
tar -I zstd -xf packs.tar.zst
rm packs.tar.zst

ls data/tokenization/pretrain_*.json
(cd data && sha256sum -c SHA256SUMS --ignore-missing)
df -h /root/rivermind-data
```

packs 的 token 数必须和 §1 的验收结果一致；不一致就停下来重传。

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
.venv/bin/torchrun --standalone --nproc_per_node=$GPUS scripts/benchmark.py \
  --flavor mini_145m_a60m --seq-len 4096 --batch-size 1 --steps 5 --warmup 1 \
  --distributed fsdp2
```

### 8.4 开训

```bash
cd /root/rivermind-data/miniDeepSeek
GPUS=$(.venv/bin/python -c "import torch; print(torch.cuda.device_count())")
.venv/bin/python scripts/autoconfig_145m.py --gpus "$GPUS"
export NCCL_P2P_DISABLE=1

nohup env NPROC_PER_NODE=$GPUS bash scripts/train_curriculum.sh \
  "configs/pretrain/generated/curriculum_145m_g${GPUS}.yaml" \
  > logs/curriculum_145m.log 2>&1 &
```

---

## 9. 监控 / 续跑 / 常见问题

```bash
tail -f logs/curriculum_145m.log
watch -n 5 nvidia-smi
tail -f checkpoints/curriculum_145m_3b/stage_00_seq2048/metrics.jsonl
```

- **续跑**：重跑 §8.4 同一条命令；已完成的 stage 会跳过，当前 stage 从最新 checkpoint 恢复；
- **换卡数续跑**：重新生成配置（`autoconfig_145m.py --gpus N`），`out_dir` 不变即可续；
- **NCCL 卡住**：消费卡保持 `NCCL_P2P_DISABLE=1`；
- **8K OOM**：确认 `activation_checkpointing: true`（配置里已开），仍 OOM 就把 `index_topk` 降到 1024；
- **loss NaN**：预训练先把 lr 降到 3e-4 重跑；
- **磁盘不够**：方案 A 已省 13 G；再不够就删 `data/pretrain_*.jsonl` 或调小 `keep_last_ckpt`。

---

## 10. 时间与磁盘预算

| 步骤 | 时间 | 磁盘（数据盘） |
|---|---|---|
| 本地 tokenize（9 并行） | 40–70 min | 本地 +27 GB |
| 压缩 packs | 3–5 min | 本地 +9 GB |
| 传代码 | < 1 min | — |
| 传数据（方案 A） | 10–20 min | +2.4 GB |
| 传 packs | 60–90 min（2 MB/s 上行） | +9 GB → 解压后 27 GB |
| 装环境 | 10 min | 系统盘 ~0.5 GB |
| 冒烟 | 10 min | — |
| 预训练 3B（4×4090） | **6–9 h** | ckpt ~4–6 GB |

> `/root/rivermind-data` **释放实例即丢失**，训练期间不要点“释放”；重要 checkpoint
> 及时 rsync 回本地或传到对象存储。
