#!/usr/bin/env python3
"""OpenAI-compatible inference server for a miniDeepSeek checkpoint.

Serves the released Hugging Face directory (``config.json`` + ``model.safetensors``
+ ``modeling_minideepseek.py``) through ``trust_remote_code``, so it works on any
machine with ``torch`` + ``transformers`` -- a CUDA box, an Apple-silicon laptop
(MPS) or plain CPU.

    # local directory produced by scripts/export_hf.py
    python scripts/serve_model.py --model hf/miniDeepSeek-base --port 8000

    # or straight from ModelScope (requires `pip install modelscope`)
    python scripts/serve_model.py --model Alita0429/miniDeepSeek --port 8000

Endpoints (subset of the OpenAI API, enough for the OpenAI SDK / curl):

    GET  /healthz
    GET  /v1/models
    POST /v1/completions
    POST /v1/chat/completions      (uses the tokenizer's chat template)

    curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"用一句话解释 MoE"}],"max_tokens":64}'

The model has no KV cache (it recomputes the packed sequence on every step, the
same forward the trainer uses), so throughput is lower than a cached engine;
keep ``max_tokens`` modest on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

DEFAULT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|end|>")


def _resolve_model(model: str) -> str:
    """Return a local directory for ``model`` (downloading from ModelScope if needed)."""
    path = Path(model)
    if path.is_dir():
        return str(path)
    try:
        from modelscope import snapshot_download
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            f"{model!r} is not a local directory and `modelscope` is not installed.\n"
            "Install it with `pip install modelscope`, or pass a local path."
        ) from error
    print(f"downloading {model} from ModelScope ...", flush=True)
    return snapshot_download(model)


class Engine:
    """Thin generate() wrapper: one model, one tokenizer, greedy/sampling."""

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_context: int | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "auto": None,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[dtype]
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "cpu" and torch_dtype == torch.bfloat16:
            # CPU bf16 matmul is slow on most x86/arm builds; fp32 is faster here.
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, dtype=torch_dtype
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.dtype = torch_dtype
        self.max_context = int(max_context or getattr(self.model.config, "max_context", 8192))
        self.stop_ids = self._stop_ids()
        self.lock = threading.Lock()
        print(
            f"loaded {model_dir} on {device} ({torch_dtype}), "
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params, "
            f"max_context {self.max_context}",
            flush=True,
        )

    def _stop_ids(self) -> list[int]:
        ids: list[int] = []
        eos = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos, int):
            ids.append(eos)
        elif isinstance(eos, (list, tuple)):
            ids.extend(int(t) for t in eos)
        for token in DEFAULT_STOP_TOKENS:
            found = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(found, int) and found >= 0 and found not in ids:
                ids.append(found)
        return ids

    def chat_ids(self, messages: list[dict[str, str]]) -> list[int]:
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover - the shipped tokenizer has a template
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: list[str] | None = None,
        seed: int | None = None,
    ):
        """Yield ``(text_piece, token_id)`` until the model stops."""
        from transformers import StoppingCriteriaList

        room = self.max_context - len(prompt_ids)
        max_tokens = max(1, min(int(max_tokens), max(room, 1)))
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        greedy = float(temperature) <= 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": not greedy,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.stop_ids or None,
            "attention_mask": attention_mask,
        }
        if not greedy:
            kwargs.update(temperature=float(temperature), top_p=float(top_p))
            if int(top_k) > 0:
                kwargs["top_k"] = int(top_k)

        with self.lock:
            if seed is not None:
                torch.manual_seed(int(seed))
            out = self.model.generate(input_ids=input_ids, **kwargs)[0, len(prompt_ids) :]

        text = self.tokenizer.decode(out.tolist(), skip_special_tokens=False)
        # Trim at the first stop token / stop string so streaming and
        # non-streaming responses agree.
        cut = len(text)
        for token in DEFAULT_STOP_TOKENS:
            index = text.find(token)
            if index >= 0:
                cut = min(cut, index)
        for token in stop or []:
            index = text.find(token)
            if index >= 0:
                cut = min(cut, index)
        return text[:cut], out.tolist()


class Handler(BaseHTTPRequestHandler):
    server_version = "miniDeepSeek/1.0"
    engine: Engine

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write("[serve] " + fmt % args + "\n")

    # -- helpers ----------------------------------------------------------
    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _sse(self, chunk: dict[str, Any]) -> None:
        self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
        self.wfile.flush()

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = _PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif self.path in ("/healthz", "/health"):
            self._json({"status": "ok", "device": self.engine.device})
        elif self.path == "/v1/models":
            self._json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "minideepseek-base",
                            "object": "model",
                            "owned_by": "miniDeepSeek",
                        }
                    ],
                }
            )
        else:
            self._json({"error": {"message": f"unknown path {self.path}"}}, 404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_body()
            if self.path == "/v1/completions":
                self._completions(body)
            elif self.path == "/v1/chat/completions":
                self._chat(body)
            else:
                self._json({"error": {"message": f"unknown path {self.path}"}}, 404)
        except Exception as error:  # surface generation errors to the client
            self._json({"error": {"message": str(error), "type": type(error).__name__}}, 500)

    def _chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages") or []
        if not messages:
            raise ValueError("messages is required")
        prompt_ids = self.engine.chat_ids(messages)
        self._generate_and_reply(body, prompt_ids, chat=True)

    def _completions(self, body: dict[str, Any]) -> None:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        prompt_ids = self.engine.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        self._generate_and_reply(body, prompt_ids, chat=False)

    def _generate_and_reply(self, body: dict[str, Any], prompt_ids: list[int], *, chat: bool) -> None:
        started = time.time()
        text, tokens = self.engine.generate(
            prompt_ids,
            max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 128),
            temperature=float(body.get("temperature", 1.0)),
            top_p=float(body.get("top_p", 1.0)),
            top_k=int(body.get("top_k") or 0),
            stop=body.get("stop"),
            seed=body.get("seed"),
        )
        ident = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(started)
        model_name = body.get("model") or "minideepseek-base"
        finish = "stop" if len(tokens) < int(body.get("max_tokens") or 128) else "length"

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if chat:
                for piece in _chunks(text, 8):
                    self._sse(
                        {
                            "id": ident,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                            ],
                        }
                    )
                self._sse(
                    {
                        "id": ident,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    }
                )
            else:
                for piece in _chunks(text, 8):
                    self._sse(
                        {
                            "id": ident,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                        }
                    )
                self._sse(
                    {
                        "id": ident,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "text": "", "finish_reason": finish}],
                    }
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        elapsed = max(time.time() - started, 1e-6)
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt_ids) + len(tokens),
        }
        if chat:
            payload = {
                "id": ident,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish,
                    }
                ],
                "usage": usage,
            }
        else:
            payload = {
                "id": ident,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                "usage": usage,
            }
        self.log_message(
            "%d prompt + %d new tokens in %.2fs (%.1f tok/s)",
            usage["prompt_tokens"],
            usage["completion_tokens"],
            elapsed,
            usage["completion_tokens"] / elapsed,
        )
        self._json(payload)


def _chunks(text: str, size: int):
    for start in range(0, len(text), size):
        yield text[start : start + size]


_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>miniDeepSeek-base playground</title>
<style>
  :root { color-scheme: light dark; }
  body { margin:0; font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;
         background:#0f172a; color:#e2e8f0; }
  .wrap { max-width: 880px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color:#94a3b8; font-size:13px; margin-bottom:18px; }
  label { display:block; font-size:12px; color:#94a3b8; margin:14px 0 6px; }
  textarea, input { width:100%; box-sizing:border-box; background:#1e293b; color:#e2e8f0;
    border:1px solid #334155; border-radius:8px; padding:10px 12px; font:inherit; }
  textarea { min-height:96px; resize:vertical; }
  .row { display:flex; gap:12px; }
  .row > div { flex:1; }
  button { margin-top:16px; background:#2563eb; color:#fff; border:0; border-radius:8px;
    padding:10px 18px; font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.55; cursor:progress; }
  pre { white-space:pre-wrap; word-break:break-word; background:#111c31; border:1px solid #1e293b;
    border-radius:10px; padding:14px; min-height:80px; margin-top:14px; }
  .meta { color:#64748b; font-size:12px; margin-top:8px; }
  a { color:#60a5fa; }
</style></head><body><div class="wrap">
<h1>miniDeepSeek-base</h1>
<div class="sub">145M-A60M · 2.5B token 预训练 · <b>base 模型，续写效果最好</b> ·
  <a href="/v1/models">/v1/models</a> · <a href="/healthz">/healthz</a></div>

<label for="prompt">Prompt（续写）</label>
<textarea id="prompt">人工智能的未来是</textarea>
<div class="row">
  <div><label for="max">最多生成 token</label><input id="max" type="number" value="64" min="1" max="1024"></div>
  <div><label for="temp">temperature（0=贪心）</label><input id="temp" type="number" value="0.7" min="0" max="2" step="0.1"></div>
</div>
<div class="row">
  <div><label>模式</label>
    <select id="mode" style="width:100%;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:10px">
      <option value="completion">续写 /v1/completions</option>
      <option value="chat">对话 /v1/chat/completions（未 SFT，效果差）</option>
    </select>
  </div>
  <div><label for="stream">流式</label>
    <select id="stream" style="width:100%;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:10px">
      <option value="0">关闭</option><option value="1">开启</option>
    </select>
  </div>
</div>
<button id="go">生成</button>
<pre id="out"></pre>
<div class="meta" id="meta"></div>

<script>
const $ = (id) => document.getElementById(id);
$('go').onclick = async () => {
  const prompt = $('prompt').value;
  const body = $('mode').value === 'chat'
    ? { model: 'minideepseek-base', messages: [{ role: 'user', content: prompt }] }
    : { model: 'minideepseek-base', prompt };
  body.max_tokens = Number($('max').value);
  body.temperature = Number($('temp').value);
  body.stream = $('stream').value === '1';
  $('out').textContent = ''; $('meta').textContent = '生成中…'; $('go').disabled = true;
  const t0 = performance.now();
  try {
    const path = $('mode').value === 'chat' ? '/v1/chat/completions' : '/v1/completions';
    if (body.stream) {
      const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const reader = res.body.getReader(); const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split('\\n\\n'); buf = parts.pop();
        for (const part of parts) {
          const line = part.trim(); if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim(); if (data === '[DONE]') continue;
          const obj = JSON.parse(data);
          const piece = $('mode').value === 'chat'
            ? (obj.choices[0].delta.content || '')
            : (obj.choices[0].text || '');
          $('out').textContent += piece;
        }
      }
    } else {
      const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await res.json();
      if (data.error) throw new Error(data.error.message);
      $('out').textContent = $('mode').value === 'chat'
        ? data.choices[0].message.content : data.choices[0].text;
      const u = data.usage;
      $('meta').textContent = `${u.completion_tokens} tokens / ${((performance.now() - t0) / 1000).toFixed(1)}s`;
    }
  } catch (err) {
    $('out').textContent = 'Error: ' + err.message;
  } finally {
    $('go').disabled = false;
    if (!body.stream) return;
    $('meta').textContent = `完成 / ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }
};
</script>
</div></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Alita0429/miniDeepSeek", help="local dir or ModelScope model id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--max-context", type=int, default=None)
    args = parser.parse_args()

    model_dir = _resolve_model(args.model)
    Handler.engine = Engine(
        model_dir, device=args.device, dtype=args.dtype, max_context=args.max_context
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}  (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
