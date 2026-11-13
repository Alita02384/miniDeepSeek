#!/usr/bin/env python3
"""SFT-only training dashboard (light theme), separate from the pretraining one.

Reads the ``metrics.jsonl`` of every SFT stage under
``checkpoints/posttrain_e2e_145m/sft_*/runs/*/`` and serves one page with the
loss curves, the four mixed domains (general / code / tool / identity), grad
norm, LR, throughput and memory.  It can also write a self-contained HTML file
so the numbers can be inspected without starting anything.

    # live (default: http://127.0.0.1:8766)
    python scripts/sft_dashboard.py

    # one-shot static report, openable by double-click
    python scripts/sft_dashboard.py --export reports/sft_dashboard.html

The page needs no network access (no CDN, no fonts) and re-reads the metrics
file every 5 s, so it keeps updating while a run is in progress.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Fields copied into the page; the rest stays on disk.
KEEP = (
    "step", "loss", "main_loss", "loss_ema", "main_loss_ema", "mtp_loss", "mtp_loss_d1",
    "balance_loss", "indexer_loss", "grad_norm", "lr", "lr_factor",
    "tokens", "tokens_seen", "tokens_per_sec", "step_time_s", "entropy", "top1_prob",
    "mem_peak_mb", "mem_reserved_mb", "wall_clock",
    "train_loss_general", "train_loss_code", "train_loss_tool", "train_loss_general#2",
    "train_main_loss_general", "train_main_loss_code", "train_main_loss_tool",
    "train_main_loss_general#2",
    # MoE routing health (same keys the pretraining dashboard draws)
    "router_learned_expert_load_max_over_mean", "router_learned_min_relative_load",
    "router_learned_routing_entropy_normalized", "router_learned_experts_used_fraction",
    "router_learned_expert_usage_mean_std",
    "router_hash_expert_load_max_over_mean", "router_hash_routing_entropy_normalized",
    "router_hash_min_relative_load", "router_hash_experts_used_fraction",
)

_STAGE = re.compile(r"(sft_[a-z0-9_]+)")


def _run_name(path: Path) -> str:
    for part in reversed(path.parts):
        match = _STAGE.search(part)
        if match:
            return match.group(1)
    return path.parent.name


def collect(patterns: list[str]) -> dict[str, Any]:
    """Load every matching metrics file, grouped by SFT stage."""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in sorted(glob.glob(pattern)))
    # De-duplicate while keeping order, and ignore empty files.
    seen: set[Path] = set()
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        name = _run_name(path)
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "step" not in record:
                    continue
                rows.append({key: record[key] for key in KEEP if key in record})
        if rows:
            rows.sort(key=lambda r: r["step"])
            runs.setdefault(name, []).extend(rows)
    for rows in runs.values():
        rows.sort(key=lambda r: r["step"])
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
    }


def summarise(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per-stage first/last values for the summary table."""
    table = []
    for name, rows in runs.items():
        last = rows[-1]

        def first(key: str) -> float | None:
            for row in rows:
                if key in row:
                    return row[key]
            return None

        table.append(
            {
                "stage": name,
                "steps": len(rows),
                "last_step": last.get("step"),
                "first_main_loss": first("main_loss"),
                "last_main_loss": last.get("main_loss"),
                "last_loss_ema": last.get("loss_ema"),
                "last_grad_norm": last.get("grad_norm"),
                "tokens_seen": last.get("tokens_seen"),
                "last_step_time": last.get("step_time_s"),
                "last_tokens_per_sec": last.get("tokens_per_sec"),
            }
        )
    table.sort(key=lambda r: r["stage"])
    return {"stages": table}


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#f8fafc; --panel:#ffffff; --line:#e2e8f0; --fg:#0f172a; --dim:#64748b;
    --accent:#2563eb; --good:#16a34a; --warn:#d97706; --bad:#dc2626;
    --c1:#2563eb; --c2:#ea580c; --c3:#16a34a; --c4:#7c3aed; --c5:#d97706;
    --c6:#0891b2; --c7:#dc2626; --c8:#0d9488;
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px 28px 48px; background:var(--bg); color:var(--fg);
         font:14px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }
  header { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 18px; margin-bottom:16px; }
  h1 { font-size:22px; margin:0; letter-spacing:-.01em; }
  .sub { color:var(--dim); font-size:13px; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; margin:6px 0 14px; }
  .tabs button { background:var(--panel); color:var(--dim); border:1px solid var(--line);
                 border-radius:999px; padding:5px 14px; font:inherit; cursor:pointer; }
  .tabs button.on { color:#fff; background:var(--accent); border-color:var(--accent); }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
  .card .label { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .card .value { font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:2px; }
  .card .hint { color:var(--dim); font-size:11px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:14px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .panel h2 { font-size:13px; margin:0 0 2px; font-weight:600; }
  .panel .note { color:var(--dim); font-size:11px; margin-bottom:6px; }
  canvas { width:100%; height:190px; display:block; }
  .legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:6px; color:var(--dim); font-size:11px; }
  .legend i { display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:5px; vertical-align:middle; }
  table { width:100%; border-collapse:collapse; margin-top:10px; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:6px 8px; border-bottom:1px solid #eef2f7; font-size:12px; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--dim); font-weight:600; }
  .empty { color:var(--dim); font-size:13px; padding:40px 0; text-align:center; }
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <span class="sub" id="stamp">加载中…</span>
  <span class="sub" id="progress"></span>
</header>
<div class="tabs" id="tabs"></div>
<div class="cards" id="cards"></div>
<div class="grid">
  <section class="panel"><h2>Loss</h2><div class="note">total（含 MTP/aux）· main（LM head）· EMA</div>
    <canvas id="c-loss"></canvas>
    <div class="legend"><span><i style="background:var(--c1)"></i>main</span>
      <span><i style="background:var(--c2)"></i>total</span>
      <span><i style="background:var(--c6)"></i>main EMA</span></div></section>
  <section class="panel"><h2>分域 loss</h2><div class="note">general · code · tool · identity（sft_identity.npz）</div>
    <canvas id="c-domain"></canvas>
    <div class="legend"><span><i style="background:var(--c1)"></i>general</span>
      <span><i style="background:var(--c2)"></i>code</span>
      <span><i style="background:var(--c3)"></i>tool</span>
      <span><i style="background:var(--c4)"></i>identity</span></div></section>
  <section class="panel"><h2>Grad norm</h2><div class="note">pre-clip，对数轴</div>
    <canvas id="c-grad"></canvas></section>
  <section class="panel"><h2>Learning rate</h2><div class="note">warmup 10 步，cosine 到 10%</div>
    <canvas id="c-lr"></canvas></section>
  <section class="panel"><h2>吞吐</h2><div class="note">tok/s（左）· step 时间（右，秒）</div>
    <canvas id="c-speed"></canvas>
    <div class="legend"><span><i style="background:var(--c3)"></i>tok/s</span>
      <span><i style="background:var(--c7)"></i>step 时间</span></div></section>
  <section class="panel"><h2>辅助损失 / 显存</h2><div class="note">MTP d1 · balance · indexer（log）与峰值显存</div>
    <canvas id="c-aux"></canvas>
    <div class="legend"><span><i style="background:var(--c4)"></i>MTP d1</span>
      <span><i style="background:var(--c5)"></i>balance</span>
      <span><i style="background:var(--c8)"></i>indexer</span>
      <span><i style="background:var(--c2)"></i>mem GB（右）</span></div></section>
  <section class="panel"><h2>预测分布</h2><div class="note">输出 token 概率分布的熵（左，nats）· top-1 概率（右）</div>
    <canvas id="c-dist"></canvas>
    <div class="legend"><span><i style="background:var(--c6)"></i>entropy</span>
      <span><i style="background:var(--c2)"></i>top-1 prob（右）</span></div></section>
  <section class="panel"><h2>MoE 路由健康</h2><div class="note">专家负载 max/mean（越接近 1 越均衡）· 路由熵（右，归一化）</div>
    <canvas id="c-router"></canvas>
    <div class="legend"><span><i style="background:var(--c1)"></i>learned max/mean</span>
      <span><i style="background:var(--c5)"></i>hash max/mean</span>
      <span><i style="background:var(--c6)"></i>entropy（右）</span></div></section>
  <section class="panel"><h2>专家使用</h2><div class="note">最小相对负载 · 被用到专家的比例（右）</div>
    <canvas id="c-experts"></canvas>
    <div class="legend"><span><i style="background:var(--c4)"></i>learned min relative load</span>
      <span><i style="background:var(--c8)"></i>hash min relative load</span>
      <span><i style="background:var(--c3)"></i>used fraction（右）</span></div></section>
</div>
<section class="panel" style="margin-top:14px">
  <h2>各阶段汇总</h2>
  <table id="summary"><thead><tr>
    <th>阶段</th><th>步数</th><th>main loss 首</th><th>main loss 末</th>
    <th>loss EMA</th><th>grad norm</th><th>tokens seen</th><th>s/step</th><th>tok/s</th>
  </tr></thead><tbody></tbody></table>
</section>

<script>
const DATA = __DATA__;
const COLORS = {c1:'#2563eb',c2:'#ea580c',c3:'#16a34a',c4:'#7c3aed',c5:'#d97706',c6:'#0891b2',c7:'#dc2626',c8:'#0d9488'};
const STYLES = getComputedStyle(document.documentElement);
const css = (name, fallback) => (STYLES.getPropertyValue(name) || fallback).trim();
let STATE = null;
let ACTIVE = null;

function fmt(v, d = 3) {
  if (v === undefined || v === null || !Number.isFinite(v)) return '—';
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (abs >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (abs >= 1e4) return (v / 1e3).toFixed(1) + 'k';
  return v.toFixed(d);
}
function dur(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '—';
  const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

function chart(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const live = series.filter(s => s.points.length);
  const pad = { l: 52, r: opts.right ? 52 : 12, t: 10, b: 22 };
  const pw = Math.max(10, w - pad.l - pad.r), ph = Math.max(10, h - pad.t - pad.b);
  ctx.font = '10px ui-monospace, monospace';
  if (!live.length) { ctx.fillStyle = css('--dim', '#64748b'); ctx.fillText('等待数据…', pad.l, pad.t + 14); return; }
  const xs = live.flatMap(s => s.points.map(p => p[0]));
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  const axis = (which) => {
    const vals = live.filter(s => (s.axis || 'left') === which)
      .flatMap(s => s.points.map(p => p[1]))
      .filter(v => Number.isFinite(v) && (!opts.log || v > 0));
    if (!vals.length) return null;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (opts.log) { lo = Math.log10(lo); hi = Math.log10(hi); }
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const padv = (hi - lo) * 0.08; lo -= padv; hi += padv;
    const s = { lo, hi };
    s.map = v => opts.log ? Math.log10(Math.max(v, 1e-12)) : v;
    s.y = v => pad.t + ph - (s.map(v) - lo) / (hi - lo) * ph;
    return s;
  };
  const left = axis('left'), right = opts.right ? axis('right') : null;
  const X = x => pad.l + (x - x0) / (x1 - x0) * pw;
  // grid + left labels
  ctx.strokeStyle = css('--line', '#e2e8f0'); ctx.lineWidth = 1;
  ctx.fillStyle = css('--dim', '#64748b');
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ph * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    if (left) {
      const raw = opts.log ? Math.pow(10, left.hi - (left.hi - left.lo) * i / 4) : left.hi - (left.hi - left.lo) * i / 4;
      ctx.fillText(fmt(raw, raw < 10 ? 2 : 0), 4, y + 3);
    }
    if (right) {
      const raw = opts.log ? Math.pow(10, right.hi - (right.hi - right.lo) * i / 4) : right.hi - (right.hi - right.lo) * i / 4;
      ctx.fillText(fmt(raw, raw < 10 ? 2 : 0), w - pad.r + 6, y + 3);
    }
  }
  // x labels
  ctx.fillText(String(Math.round(x0)), pad.l, h - 6);
  const xm = String(Math.round((x0 + x1) / 2));
  ctx.fillText(xm, pad.l + pw / 2 - ctx.measureText(xm).width / 2, h - 6);
  const xe = String(Math.round(x1));
  ctx.fillText(xe, pad.l + pw - ctx.measureText(xe).width, h - 6);
  // series
  for (const s of live) {
    const ax = (s.axis === 'right' ? right : left);
    if (!ax) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.7;
    ctx.beginPath();
    s.points.forEach(([x, y], i) => {
      if (!Number.isFinite(y) || (opts.log && y <= 0)) return;
      const px = X(x), py = ax.y(y);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
    const last = [...s.points].reverse().find(p => Number.isFinite(p[1]) && (!opts.log || p[1] > 0));
    if (last) {
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(X(last[0]), ax.y(last[1]), 2.2, 0, Math.PI * 2); ctx.fill();
    }
  }
}

const pts = (rows, key) => rows.filter(r => Number.isFinite(r[key])).map(r => [r.step, r[key]]);

function renderCards(rows) {
  const last = rows[rows.length - 1] || {};
  const best = rows.reduce((m, r) => (Number.isFinite(r.main_loss) && r.main_loss < m ? r.main_loss : m), Infinity);
  const done = last.step || 0, total = LAST_TOTAL;
  const remaining = total > done ? (total - done) * (last.step_time_s || 0) : 0;
  const cards = [
    ['step', `${done}`, total ? `/ ${total}` : ''],
    ['main loss', fmt(last.main_loss), `best ${fmt(best)}`],
    ['loss EMA', fmt(last.loss_ema), ''],
    ['grad norm', fmt(last.grad_norm), ''],
    ['lr', fmt(last.lr, 2), ''],
    ['吞吐', fmt(last.tokens_per_sec, 0) + ' tok/s', (last.step_time_s || 0).toFixed(2) + ' s/step'],
    ['已训 token', fmt(last.tokens_seen), ''],
    ['剩余', dur(remaining), ''],
  ];
  document.getElementById('cards').innerHTML = cards.map(([k, v, hint]) =>
    `<div class="card"><div class="label">${k}</div><div class="value">${v}</div><div class="hint">${hint || '&nbsp;'}</div></div>`).join('');
}

function render() {
  const runs = (STATE && STATE.runs) || {};
  const names = Object.keys(runs).sort();
  if (!names.length) {
    document.getElementById('tabs').innerHTML = '<button class="on">无数据</button>';
    document.getElementById('cards').innerHTML = '<div class="empty">还没找到 SFT 的 metrics.jsonl</div>';
    return;
  }
  if (!ACTIVE || !runs[ACTIVE]) ACTIVE = names[names.length - 1];
  document.getElementById('tabs').innerHTML = names.map(n =>
    `<button data-run="${n}" class="${n === ACTIVE ? 'on' : ''}">${n}（${runs[n].length} 步）</button>`).join('');
  document.querySelectorAll('#tabs button[data-run]').forEach(b => {
    b.onclick = () => { ACTIVE = b.dataset.run; render(); };
  });
  const rows = runs[ACTIVE];
  renderCards(rows);
  const has = (k) => rows.some(r => Number.isFinite(r[k]));
  chart(document.getElementById('c-loss'), [
    { points: pts(rows, 'main_loss'), color: COLORS.c1 },
    { points: pts(rows, 'loss'), color: COLORS.c2 },
    { points: pts(rows, 'main_loss_ema'), color: COLORS.c6 },
  ]);
  chart(document.getElementById('c-domain'), [
    { points: pts(rows, 'train_main_loss_general'), color: COLORS.c1 },
    { points: pts(rows, 'train_main_loss_code'), color: COLORS.c2 },
    { points: pts(rows, 'train_main_loss_tool'), color: COLORS.c3 },
    { points: pts(rows, 'train_main_loss_general#2'), color: COLORS.c4 },
  ]);
  chart(document.getElementById('c-grad'), [{ points: pts(rows, 'grad_norm'), color: COLORS.c5 }], { log: true });
  chart(document.getElementById('c-lr'), [{ points: pts(rows, 'lr'), color: COLORS.c1 }]);
  chart(document.getElementById('c-speed'), [
    { points: pts(rows, 'tokens_per_sec'), color: COLORS.c3 },
    { points: pts(rows, 'step_time_s'), color: COLORS.c7, axis: 'right' },
  ], { right: true });
  chart(document.getElementById('c-dist'), [
    { points: pts(rows, 'entropy'), color: COLORS.c6 },
    { points: pts(rows, 'top1_prob'), color: COLORS.c2, axis: 'right' },
  ], { right: true });
  chart(document.getElementById('c-router'), [
    { points: pts(rows, 'router_learned_expert_load_max_over_mean'), color: COLORS.c1 },
    { points: pts(rows, 'router_hash_expert_load_max_over_mean'), color: COLORS.c5 },
    { points: pts(rows, 'router_learned_routing_entropy_normalized'), color: COLORS.c6, axis: 'right' },
  ], { right: true });
  chart(document.getElementById('c-experts'), [
    { points: pts(rows, 'router_learned_min_relative_load'), color: COLORS.c4 },
    { points: pts(rows, 'router_hash_min_relative_load'), color: COLORS.c8 },
    { points: pts(rows, 'router_learned_experts_used_fraction'), color: COLORS.c3, axis: 'right' },
  ], { right: true });
  chart(document.getElementById('c-aux'), [
    { points: pts(rows, 'mtp_loss_d1'), color: COLORS.c4 },
    { points: pts(rows, 'balance_loss'), color: COLORS.c5 },
    { points: pts(rows, 'indexer_loss'), color: COLORS.c8 },
    { points: rows.filter(r => Number.isFinite(r.mem_peak_mb)).map(r => [r.step, r.mem_peak_mb / 1024]), color: COLORS.c2, axis: 'right' },
  ], { log: true, right: true });

  const body = document.querySelector('#summary tbody');
  body.innerHTML = (STATE.stages || []).map(s => `<tr>
    <td>${s.stage}</td><td>${s.steps}</td><td>${fmt(s.first_main_loss)}</td>
    <td>${fmt(s.last_main_loss)}</td><td>${fmt(s.last_loss_ema)}</td><td>${fmt(s.last_grad_norm)}</td>
    <td>${fmt(s.tokens_seen)}</td><td>${fmt(s.last_step_time, 2)}</td><td>${fmt(s.last_tokens_per_sec, 0)}</td></tr>`).join('');
  document.getElementById('stamp').textContent = `更新于 ${STATE.generated_at}` + (LIVE ? ' · 每 5 秒自动刷新' : '');
}

async function refresh() {
  try {
    const res = await fetch('/data', { cache: 'no-store' });
    STATE = await res.json();
  } catch (err) {
    document.getElementById('stamp').textContent = '读取失败: ' + err.message;
    return;
  }
  render();
}

const LAST_TOTAL = DATA && DATA.total_steps ? DATA.total_steps : 0;
const LIVE = !(DATA && DATA.static);
if (LIVE) { refresh(); setInterval(refresh, 5000); }
else if (DATA && DATA.payload) { STATE = DATA.payload; render(); }
</script>
</body></html>
"""


def render_html(title: str, data: dict[str, Any], *, static: bool, total_steps: int) -> str:
    payload = {
        "static": static,
        "total_steps": total_steps,
        "payload": data if static else None,
    }
    return (
        PAGE.replace("__TITLE__", title)
        .replace("__DATA__", json.dumps(payload))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metrics",
        action="append",
        default=None,
        help="metrics.jsonl glob (repeatable); defaults to the posttrain e2e tree",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--title", default="miniDeepSeek SFT")
    parser.add_argument("--total-steps", type=int, default=0, help="每阶段计划步数（用于进度/ETA）")
    parser.add_argument("--export", default=None, help="写出静态 HTML 后退出")
    args = parser.parse_args(argv)

    patterns = args.metrics or [
        str(ROOT / "checkpoints/posttrain_e2e_145m/sft_*/runs/*/metrics.jsonl"),
        str(ROOT / "checkpoints/sft_*_145m/runs/*/metrics.jsonl"),
    ]
    if args.export:
        data = collect(patterns)
        data.update(summarise(data["runs"]))
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            render_html(args.title, data, static=True, total_steps=args.total_steps),
            encoding="utf-8",
        )
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
        return 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:  # noqa: A003
            sys.stderr.write("[sft-dash] " + fmt % a + "\n")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = render_html(args.title, {}, static=False, total_steps=args.total_steps).encode()
                ctype = "text/html; charset=utf-8"
            elif self.path.startswith("/data"):
                data = collect(patterns)
                data.update(summarise(data["runs"]))
                body = json.dumps(data).encode()
                ctype = "application/json; charset=utf-8"
            elif self.path == "/favicon.ico":
                self.send_response(204); self.end_headers(); return
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SFT dashboard on http://{args.host}:{args.port}  (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
