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
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "live_dashboard.html"

#: Fields the dashboard draws.  Everything else stays on the server.
KEEP = tuple(dict.fromkeys(['"step', "loss", "loss_ema", "main_loss", "mtp_loss", "val_loss", "val_ppl", "val_main_loss", '"', '"val_loss_general', "val_loss_code", "val_loss_tool", "val_main_loss_general", '"', '"val_main_loss_code', "val_main_loss_tool", "tokens", "tokens_per_sec", "tokens_seen", '"', '"step_time_s', "grad_norm", "lr", "eta_seconds", '"', '"router_learned_expert_load_max_over_mean', "router_learned_experts_used_fraction", '"', '"router_learned_routing_entropy_normalized', "router_learned_min_relative_load", '"', '"router_hash_expert_load_max_over_mean', "swiglu_gate_clamp_rate", '"', '"mem_peak_mb', "domain_name", "loss_spike", "indexer_loss", 'balance_loss"']))


class Remote:
    """One SSH hop, one followed file."""

    def __init__(self, host: str, port: int, root: str, stage: str) -> None:
        self.host = host
        self.port = port
        self.root = root.rstrip("/")
        # ``stage`` is either one stage directory or the curriculum root; the
        # glob below covers both, so the view keeps following when the run
        # crosses into the next stage directory.
        self.stage = f"{self.root}/{stage.strip('/')}"
        self._metrics_glob = f"{self.stage}/runs/*/metrics.jsonl"
        self.offset = 0
        self.path = ""
        self.records: dict[int, dict] = {}
        self.snapshots: list[dict] = []
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
                    for path in self._all_files():
                        for record in self._read_file(path):
                            self._absorb(record)
                    self.offset = int(
                        self._ssh(f"stat -c %s {self.path}").strip() or 0
                    )
                else:
                    chunk = self._ssh(f"tail -c +{self.offset + 1} {self.path}")
                    if chunk:
                        self.offset += len(chunk.encode())
                        for record in _parse(chunk):
                            self._absorb(record)
                self.error = ""
                self.updated = time.time()
            except Exception as error:  # keep serving the last good snapshot
                self.error = str(error)

    def _absorb(self, record: dict) -> None:
        step = record.get("step")
        if step is None:
            return
        self.records[int(step)] = record
        layers = record.get("router_layers")
        if layers:
            self.snapshots.append({"step": int(step), "layers": layers})
            self.snapshots = self.snapshots[-8:]

    def payload(self) -> dict:
        with self._lock:
            rows = [self.records[key] for key in sorted(self.records)]
            latest_layers = self.snapshots[-1] if self.snapshots else None
            newest = rows[-1] if rows else {}
            return {
                "stage": self.stage,
                "file": self.path,
                "updated_at": time.time(),
                "age_s": round(time.time() - self.updated, 1) if self.updated else None,
                "error": self.error,
                "latest": {
                    key: newest.get(key)
                    for key in (
                        "step", "loss", "loss_ema", "main_loss", "tokens_per_sec",
                        "eta_seconds", "lr", "grad_norm", "tokens_seen", "mem_peak_mb",
                        "domain_name",
                    )
                },
                "router_layers": latest_layers,
                "rows": [_slim(record) for record in _thin(rows)],
            }


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
    return {key: record[key] for key in KEEP if key in record}


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
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--remote-root", default="/root/rivermind-data/miniDeepSeek")
    parser.add_argument(
        "--stage",
        default="checkpoints/curriculum_145m_3b/stage_00_seq2048",
        help="path (relative to --remote-root) of one curriculum stage",
    )
    parser.add_argument("--local-port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    args = parser.parse_args(argv)

    remote = Remote(args.host, args.port, args.remote_root, args.stage)
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
    print(f"[live] dashboard: http://127.0.0.1:{args.local_port}/", flush=True)
    print(f"[live] raw data : http://127.0.0.1:{args.local_port}/data", flush=True)
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
