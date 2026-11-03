# 代码来源与改动

> 规则见 `AGENTS.md` §3.3：保留版权头、只复制最小闭包、记录来源 commit 与改动。
> 本仓库**不修改**任何第三方 checkout。

## 1. torchtitan（BSD-3-Clause）

- 来源：`../torchtitan`，checkout commit `01586df70`（`git -C ../torchtitan rev-parse HEAD`）。
- 用途：DeepSeek-V4 的**模型实现**（attention / compressor / mHC / MoE / MTP /
  sharding / state_dict_adapter）与 debug 配置的对照。运行时通过
  `src/minideepseek/_compat.py` 把该 checkout 加到 `sys.path`；也可显式安装。

### 复制的文件

| 本仓库文件 | 上游文件 | 改动 |
|---|---|---|
| `src/minideepseek/model/flavors.py` | `torchtitan/models/deepseek_v4/flavors.py` | 见下 |

改动明细：

1. 保留原 BSD-3 版权头与文件级说明，追加本项目的来源说明。
2. 相对导入 `from .attention/.compressor/.mhc/.model/.moe/.mtp/.sharding` 改为
   `torchtitan.models.deepseek_v4.*` 绝对导入（实现类留在上游）。
3. 新增 `_MINI_PRESETS` 与 `_mini_model`：mini-60M / 110M / 200M 三档配置；
   `compress_ratios = (0, 0) + (4, 32) * k`，`hca=32`、`index_topk=64`、
   `n_hash_layers=3`、`top_k=2`、`num_experts=6`、`num_shared_experts=1`、
   `swiglu_limit=10.0`、`hc_mult=4`、tied embeddings。
4. 新增 `_mini_60m_dense` / `debugmodel_dense` 与 `_mini_dense_from_debug`：
   `compress_ratios` 全 0、`window_size=seq_len` 的 dense 对照。
5. 新增 `tie_word_embeddings` 与 `swiglu_limit` 运行时覆盖参数
   （上游 debug flavor 默认 `swiglu_limit=None`，本项目测试需要开启 clamp）。
6. `MODEL_FLAVORS` 只注册 debug / mini / dense 六个 flavor；上游 flash/pro 的
   **构建函数仍保留在文件中**用于结构对照，但不在注册表，避免误用 284B 配置。
7. 未改动 `torchtitan/` 下任何文件；开发过程中一次 `pip install -e` 产生的
   `torchtitan.egg-info` 已删除，最终环境不写上游目录。

## 2. minimind（Apache-2.0）

| 本仓库文件 | 上游文件 | 说明 |
|---|---|---|
| `tokenizer/tokenizer.json` | `minimind/model/tokenizer.json` | 6,400 词表，36 个 added tokens；未修改 |
| `tokenizer/tokenizer_config.json` | `minimind/model/tokenizer_config.json` | chat template 原样复制；未修改 |

数据样本（`pretrain_t2t_mini.jsonl` / `sft_t2t_mini.jsonl` / `agent_rl_math.jsonl`）
只在本地脚本中读取，**不复制进 git**。

## 3. verl（Apache-2.0）与 DeepSeek-V4-Flash（MIT）

- **未复制代码**。
- verl：阅读 `docs/algo/opd.md`、`examples/on_policy_distillation_trainer/`、
  `utils/veomni/router_replay.py` 的设计，作为 `rl/opd.py`、`rl/rollout.py`、
  `model/routing.py` 的算法参考。本项目在 CPU 环境下不能安装/运行 verl+vLLM，
  因此实现为原生 torch 版本（见 `doc/decisions.md`）。
- DeepSeek-V4-Flash：阅读 `config.json`、`README.md`、inference 结构，用于对照
  超参和 chat 格式；按要求**没有整体复制** `inference/model.py`。

## 4. 复现来源

```bash
git -C ../torchtitan rev-parse HEAD      # 01586df70...
git -C ../minimind rev-parse HEAD 2>/dev/null || true
sha256sum tokenizer/tokenizer.json      # 见 checkpoint meta / data meta
```

torchtitan 上游若有变动，`flavors.py` 的私有构建函数签名可能变化；升级时先跑
`pytest tests/test_model.py tests/test_tokenizer.py`，再重新核对本文件的改动点。
