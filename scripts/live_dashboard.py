#!/usr/bin/env python3
"""Live training dashboard: a local bridge that tails a remote metrics.jsonl.

The training box only exposes SSH, so a page opened in a browser cannot read
``metrics.jsonl`` by itself.  This script holds one SSH connection, follows the
file with ``tail -c +N`` (only the new bytes cross the wire), and serves two
things on ``127.0.0.1``:

    /        the dashboard (scripts/live_dashboard.html)
    /data    the records as JSON, ``Access-Control-Allow-Origin: *``

Open either ``http://127.0.0.1:8765`` or the HTML file directly -- the page
polls ``http://127.0.0.1:8765/data`` either way and redraws every few seconds.
The HTML alone shows nothing: a browser cannot open an SSH connection, so this
process has to be running for the page to have data.

    python scripts/live_dashboard.py \
        --host hb01-ssh.gpuhome.cc --port 30136 \
        --remote-root /root/rivermind-data/miniDeepSeek \
        --stage checkpoints/curriculum_145m_3b/stage_00_seq2048

Needs passwordless SSH (``ssh-copy-id -p PORT root@HOST``); otherwise export
``MINIDEEPSEEK_SSH_PASSWORD`` and it will go through ``SSH_ASKPASS``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent

#: ``.../stage_01_seq4096/...`` -> (1, 4096), so a record read straight out of a
#: per-launch file still carries the stage it belongs to.
STAGE_DIR = re.compile(r"stage_(\d+)_seq(\d+)")
HTML_PATH = HERE / "live_dashboard.html"

#: Fields the dashboard draws.  Everything else stays on the server.
#: Fields the dashboard draws (per-domain ``router_domain_*`` keys are added
#: dynamically in ``_slim``).  Everything else stays on the server.
KEEP = (
    # ---- loss ------------------------------------------------------------
    "step", "loss", "loss_ema", "main_loss", "main_loss_ema", "mtp_loss",
    "mtp_loss_d1", "mtp_weight", "balance_loss", "indexer_loss",
    # ---- validation (main head is the comparable one) --------------------
    "val_loss", "val_ppl", "val_main_loss", "val_mtp_loss", "val_tokens",
    "val_loss_general", "val_loss_code", "val_loss_tool",
    "val_main_loss_general", "val_main_loss_code", "val_main_loss_tool",
    # ---- per-domain training loss (needs data.lockstep=false) ------------
    "train_main_loss_general", "train_main_loss_code", "train_main_loss_tool",
    "train_loss_general", "train_loss_code", "train_loss_tool",
    # ---- throughput / schedule -------------------------------------------
    "tokens", "tokens_per_sec", "tokens_seen", "step_time_s",
    "grad_norm", "lr", "lr_factor", "eta_seconds", "index_topk",
    # ---- model behaviour -------------------------------------------------
    "entropy", "top1_prob",
    # ---- router health (learned + hash) ----------------------------------
    "router_learned_expert_load_max_over_mean",
    "router_learned_expert_load_max_over_mean_worst",
    "router_learned_experts_used_fraction",
    "router_learned_routing_entropy_normalized",
    "router_learned_routing_entropy_nats",
    "router_learned_min_relative_load",
    "router_learned_expert_usage_mean_std",
    "router_hash_expert_load_max_over_mean",
    "router_hash_experts_used_fraction",
    "router_hash_routing_entropy_normalized",
    "router_hash_min_relative_load",
    # ---- SwiGLU clamp / memory / misc ------------------------------------
    "swiglu_gate_clamp_rate", "swiglu_up_clamp_rate",
    "swiglu_gate_abs_max", "swiglu_up_abs_max",
    "mem_allocated_mb", "mem_reserved_mb", "mem_peak_mb",
    "domain_name", "loss_spike", "ar_active",
    # ---- provenance: which stage produced the row, and where it came from --
    "stage", "seq_len", "source", "wall_clock",
)


_MAX_TREND_POINTS = 4000


class _State:
    """records / snapshots accumulator shared by the live bridge and the exporter."""

    def __init__(self) -> None:
        self.records: dict[int, dict] = {}
        self.snapshots: list[dict] = []
        self.domain_snapshots: list[dict] = []


def _absorb(state, record: dict, path: str = "") -> None:
    step = record.get("step")
    if step is None:
        return
    step = int(step)
    if path:
        match = STAGE_DIR.search(path)
        if match:
            record.setdefault("stage", int(match.group(1)))
            record.setdefault("seq_len", int(match.group(2)))
        record.setdefault("source", "metrics")
    previous = state.records.get(step)
    if previous is None or len(record) >= len(previous):
        state.records[step] = record
    else:
        return
    layers = record.get("router_layers")
    if layers and (not state.snapshots or state.snapshots[-1]["step"] != int(step)):
        state.snapshots.append({"step": int(step), "layers": layers})
        state.snapshots = state.snapshots[-8:]
    domain_layers = record.get("router_domain_layers")
    if domain_layers and (
        not state.domain_snapshots or state.domain_snapshots[-1]["step"] != int(step)
    ):
        state.domain_snapshots.append({"step": int(step), "domains": domain_layers})
        state.domain_snapshots = state.domain_snapshots[-_MAX_TREND_POINTS:]


def _payload(state, *, stage: str, path: str, updated: float, error: str) -> dict:
    rows = [state.records[key] for key in sorted(state.records)]
    latest_layers = state.snapshots[-1] if state.snapshots else None
    newest = rows[-1] if rows else {}
    return {
        "stage": stage,
        "file": path,
        "updated_at": time.time(),
        "age_s": round(time.time() - updated, 1) if updated else None,
        "error": error,
        "latest": {
            key: newest.get(key)
            for key in (
                "step", "loss", "loss_ema", "main_loss", "main_loss_ema",
                "tokens_per_sec", "eta_seconds", "lr", "lr_factor",
                "grad_norm", "tokens_seen", "mem_peak_mb", "mtp_weight",
                "balance_loss", "entropy", "top1_prob", "domain_name",
                "train_main_loss_general", "train_main_loss_code",
                "train_main_loss_tool", "val_main_loss",
                "router_learned_min_relative_load", "stage", "seq_len", "source",
            )
        },
        "router_layers": latest_layers,
        "router_domain_layers": (
            state.domain_snapshots[-1] if state.domain_snapshots else None
        ),
        "best_val": _best_val(rows),
        "router_series": _router_series(state.domain_snapshots, rows),
        "rows": [_slim(record) for record in _thin(rows)],
    }


class Remote:
    """One SSH hop, one followed file."""

    def __init__(
        self,
        host: str,
        port: int,
        root: str,
        stage: str,
        metrics_file: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.root = root.rstrip("/")
        #: Explicit consolidated file (``merge_metrics.py --name`` output).  The
        #: curriculum run spans three stage directories, so pointing at one
        #: ``metrics_all.jsonl`` is how the page shows steps 0..N in one axis.
        self.metrics_file = metrics_file or ""
        # ``stage`` is either one stage directory or the curriculum root; the
        # glob below covers both, so the view keeps following when the run
        # crosses into the next stage directory.
        self.stage = f"{self.root}/{stage.strip('/')}"
        self._metrics_glob = f"{self.stage}/runs/*/metrics.jsonl"
        self.offset = 0
        self.path = ""
        self.records: dict[int, dict] = {}
        self.snapshots: list[dict] = []
        #: ``{domain: {layer: {"counts": [E]}}}`` per snapshot step.  Kept out
        #: of ``rows`` (~3 KB each) but *all* of them are retained: the expert
        #: view renders the newest one and the trend panel needs the history.
        self.domain_snapshots: list[dict] = []
        self.error = ""
        self.updated = 0.0
        self._lock = threading.Lock()
        self._stop = False

    # -- ssh ------------------------------------------------------------------
    def _ssh(self, command: str, timeout: float = 30.0) -> str:
        argv = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=10",
            "-p", str(self.port),
            f"root@{self.host}",
            command,
        ]
        env = dict(os.environ)
        if os.environ.get("MINIDEEPSEEK_SSH_PASSWORD"):
            askpass = Path("/tmp/mds_askpass.sh")
            askpass.write_text('#!/bin/sh\nprintf "%s\\n" "$MINIDEEPSEEK_SSH_PASSWORD"\n')
            askpass.chmod(0o700)
            env["SSH_ASKPASS"] = str(askpass)
            env["SSH_ASKPASS_REQUIRE"] = "force"
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or f"ssh exited {out.returncode}")
        return out.stdout

    # -- data -----------------------------------------------------------------
    def _all_files(self) -> list[str]:
        found = self._ssh(
            f"ls -t {self._metrics_glob} {self.stage}/*/runs/*/metrics.jsonl "
            "2>/dev/null | sort -u"
        ).split()
        return [path for path in found if path.endswith("metrics.jsonl")]

    def _newest_file(self) -> str:
        files = self._ssh(
            f"ls -t {self._metrics_glob} {self.stage}/*/runs/*/metrics.jsonl "
            "2>/dev/null | head -1"
        ).strip()
        if not files:
            raise RuntimeError(f"no metrics.jsonl under {self.stage}")
        return files

    def _read_file(self, path: str) -> list[dict]:
        text = self._ssh(f"cat {path}")
        return _parse(text)

    def _history_file(self) -> str:
        """Consolidated history: explicit ``--metrics-file`` or ``metrics_all``.

        ``scripts/merge_metrics.py`` consolidates every launch plus the steps
        that only survive in the console log; seeding from it means the page
        shows the whole campaign instead of just the current stage's launches.
        """
        if self.metrics_file:
            found = self._ssh(f"test -s {self.metrics_file} && echo yes").strip()
            if found == "yes":
                return self.metrics_file
        parent = self.stage.rsplit("/", 1)[0]
        for candidate in (
            f"{self.stage}/metrics_full.jsonl",
            f"{parent}/metrics_full.jsonl",
            f"{self.stage}/metrics_all.jsonl",
            f"{parent}/metrics_all.jsonl",
        ):
            found = self._ssh(f"test -s {candidate} && echo yes").strip()
            if found == "yes":
                return candidate
        return ""

    def refresh(self) -> None:
        """Pick up new bytes from the followed file."""
        with self._lock:
            try:
                newest = self._newest_file()
                if newest != self.path:  # first poll, or the run directory moved
                    self.path = newest
                    self.offset = 0
                    self.records = {}
                    self.snapshots = []
                    self.domain_snapshots = []
                    history = self._history_file()
                    if history:
                        for record in self._read_file(history):
                            self._absorb(record, history)
                    for path in self._all_files():
                        for record in self._read_file(path):
                            self._absorb(record, path)
                    self.offset = int(
                        self._ssh(f"stat -c %s {self.path}").strip() or 0
                    )
                else:
                    chunk = self._ssh(f"tail -c +{self.offset + 1} {self.path}")
                    if chunk:
                        self.offset += len(chunk.encode())
                        for record in _parse(chunk):
                            self._absorb(record, self.path)
                self.error = ""
                self.updated = time.time()
            except Exception as error:  # keep serving the last good snapshot
                self.error = str(error)

    def _absorb(self, record: dict, path: str = "") -> None:
        _absorb(self, record, path)

    def payload(self) -> dict:
        return _payload(
            self, stage=self.stage, path=self.path, updated=self.updated, error=self.error
        )


def _router_series(snapshots: list[dict], records: list[dict]) -> dict:
    """Per-domain expert-share history for the trend panel and the scrubber.

    Two sources feed it, because the per-domain detail predates the key that
    carries it:

    * ``router_domain_layers`` -- per-domain x layer x expert counts accumulated
      over a whole ``train.router_every`` window.  Only exists since the field
      was added, so it covers the tail of the run.
    * ``router_layers`` + ``domain_name`` -- one micro-batch's per-layer counts.
      Before ``data.lockstep=false`` a step was domain-homogeneous, so every
      such record *is* that domain's routing; that recovers samples back to a
      few hundred steps.  They are logged as ``sampled`` so the page can say
      which points are a single micro-batch rather than a window.

    Shares are per-mille integers so the payload stays small; rows are aligned
    to the union of snapshot steps, with ``[]`` where a domain had no tokens.
    """
    windows: dict[int, dict[str, dict[str, list[int]]]] = {}
    sampled: set[int] = set()
    for snapshot in snapshots:
        windows[int(snapshot["step"])] = snapshot["domains"]
    for record in records:
        step = int(record["step"])
        if step in windows:
            continue  # a full window always wins over a single micro-batch
        domain = record.get("domain_name")
        layers = record.get("router_layers")
        if not domain or not layers:
            continue
        windows[step] = {domain: layers}
        sampled.add(step)

    steps = sorted(windows)
    layers_seen: dict[str, None] = {}
    domains_seen: dict[str, None] = {}
    for per_domain in windows.values():
        for domain, per_layer in per_domain.items():
            domains_seen.setdefault(domain, None)
            for layer in per_layer:
                layers_seen.setdefault(layer, None)

    shares: dict[str, dict[str, list[list[int]]]] = {}
    for domain in domains_seen:
        per_layer_out: dict[str, list[list[int]]] = {}
        for layer in layers_seen:
            column: list[list[int]] = []
            for step in steps:
                info = windows[step].get(domain, {}).get(layer)
                counts = (info or {}).get("counts") or []
                total = sum(counts)
                column.append(
                    [round(value * 1000 / total) for value in counts] if total else []
                )
            per_layer_out[layer] = column
        shares[domain] = per_layer_out

    order = ("overall", "general", "code", "tool")
    domains = [name for name in order if name in domains_seen]
    domains += sorted(name for name in domains_seen if name not in domains)
    return {
        "steps": steps,
        "layers": sorted(layers_seen),
        "domains": domains,
        "shares": shares,
        "sampled": sorted(sampled),
    }


def _best_val(rows: list[dict]) -> dict | None:
    """Lowest ``val_main_loss`` so far, for the header badge."""
    best = None
    for row in rows:
        value = row.get("val_main_loss")
        if isinstance(value, (int, float)) and (best is None or value < best[0]):
            best = (float(value), int(row["step"]))
    if best is None:
        return None
    return {"value": best[0], "step": best[1]}


def _parse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _slim(record: dict) -> dict:
    out = {key: record[key] for key in KEEP if key in record}
    # Per-domain routing keys are emitted every ``router_every`` steps and name
    # their domain, so they cannot live in the fixed list above.  Only the
    # scalars: ``router_domain_layers`` is the nested per-layer snapshot (~1 KB)
    # and travels on its own, outside ``rows``.
    for key, value in record.items():
        if key.startswith("router_domain_") and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            out[key] = value
    return out


def _thin(rows: list[dict], budget: int = 1500) -> list[dict]:
    """Keep every val record and an evenly spaced tail-sized sample."""
    if len(rows) <= budget:
        return rows
    vals = [row for row in rows if "val_loss" in row]
    stride = max(1, len(rows) // (budget - len(vals))) if len(vals) < budget else 1
    sampled = rows[::stride]
    seen = {id(row) for row in sampled}
    extra = [row for row in vals if id(row) not in seen]
    merged = sorted(sampled + extra, key=lambda row: row["step"])
    return merged[-budget:]


class Handler(BaseHTTPRequestHandler):
    remote: Remote

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
        elif route == "/data":
            self._send(
                200,
                "application/json; charset=utf-8",
                json.dumps(self.remote.payload()).encode(),
            )
        elif route == "/health":
            self._send(200, "text/plain", b"ok\n")
        else:
            self._send(404, "text/plain", b"not found\n")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep the console quiet
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="", help="remote host (required unless --export)")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--remote-root", default="/root/rivermind-data/miniDeepSeek")
    parser.add_argument(
        "--stage",
        default="checkpoints/curriculum_145m_3b",
        help="curriculum root or one stage directory, relative to --remote-root; "
        "the curriculum root keeps following the run across stage boundaries",
    )
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="explicit consolidated metrics.jsonl (e.g. metrics_full.jsonl) to seed from",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="write a static HTML (same layout as the live page) and exit",
    )
    parser.add_argument(
        "--local-metrics",
        default=None,
        help="local metrics.jsonl used by --export (no SSH involved)",
    )
    parser.add_argument("--local-port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    args = parser.parse_args(argv)

    if not args.export and not args.host:
        raise SystemExit("--host is required unless --export is used")
    if args.export:
        if not args.local_metrics:
            raise SystemExit("--export needs --local-metrics <file>")
        source = Path(args.local_metrics)
        if not source.is_file():
            raise SystemExit(f"no such metrics file: {source}")
        state = _State()
        for record in _parse(source.read_text(encoding="utf-8")):
            _absorb(state, record, str(source))
        payload = _payload(
            state,
            stage=args.stage or str(source),
            path=str(source),
            updated=time.time(),
            error="",
        )
        shim = (
            "<script>\nwindow.__DASH_DATA__ = "
            + json.dumps(payload)
            + ";\n(function () {\n"
            "  var data = window.__DASH_DATA__;\n"
            "  var orig = window.fetch.bind(window);\n"
            "  window.fetch = function (url, opts) {\n"
            "    if (String(url).indexOf('/data') !== -1) {\n"
            "      return Promise.resolve({ ok: true, status: 200,\n"
            "        json: function () { return Promise.resolve(data); } });\n"
            "    }\n"
            "    return orig(url, opts);\n"
            "  };\n"
            "})();\n</script>\n"
        )
        html = HTML_PATH.read_text(encoding="utf-8")
        index = html.find("<script")
        if index < 0:
            index = len(html)
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html[:index] + shim + html[index:], encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(payload['rows'])} rows)")
        return 0

    remote = Remote(args.host, args.port, args.remote_root, args.stage, args.metrics_file)
    print(
        f"[live] following {remote.stage}/runs/*/metrics.jsonl on {args.host}",
        flush=True,
    )
    remote.refresh()
    if remote.error:
        print(f"[live] first poll failed: {remote.error}", file=sys.stderr)

    stop = threading.Event()

    def follow() -> None:
        while not stop.wait(args.interval):
            remote.refresh()

    threading.Thread(target=follow, daemon=True).start()

    Handler.remote = remote
    server = ThreadingHTTPServer(("127.0.0.1", args.local_port), Handler)
    url = f"http://127.0.0.1:{args.local_port}/"
    print(f"[live] dashboard: {url}", flush=True)
    print(f"[live] raw data : {url}data", flush=True)
    print("[live] the page polls every 5 s; leave this process running", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[live] bye")
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
