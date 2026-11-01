"""Signed inference proxy for OpenAI-compatible servers (vLLM / SGLang).

It does not make a remote model deterministic. It records the envelope, calls the model,
builds a receipt, and signs it. GTL verifies the signature before accepting EXACT/BATCH_INVARIANT.

    python3 -m dsc.inference_service --upstream http://127.0.0.1:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ --sign-key /secure/infer.pem --port 8094
    curl -s localhost:8094/infer -d '{"prompt":"2+2","max_tokens":32}'
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .nondeterminism import local_vllm_env, receipt, sign_inference
from .sigil import load_private


def env_for(model: str, batch_invariant: bool, temperature: float = 0.0,
            max_tokens: int = 0):
    env = local_vllm_env(model, batch_invariant=bool(batch_invariant), max_tokens=max_tokens)
    return replace(env, temperature=float(temperature))


def call_upstream(upstream: str, model: str, prompt: str, api_key: str = "",
                  timeout: float = 120.0, temperature: float = 0.0,
                  max_tokens: int = 512) -> tuple[str, dict]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature), "max_tokens": int(max_tokens)}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(upstream.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"], data


def signed_result(prompt: str, output: str, model: str, private_key=None,
                  key_id: str | None = None, batch_invariant: bool = False,
                  temperature: float = 0.0, max_tokens: int = 0,
                  logprobs: list | None = None, meta: dict | None = None) -> dict:
    env = env_for(model, batch_invariant, temperature, max_tokens)
    rec = receipt(prompt, output, env, logprobs=logprobs)
    if private_key is not None:
        rec = sign_inference(rec, private_key, key_id)
    return {"ok": True, "output": output, "envelope": env.to_dict(),
            "receipt": rec, "trust": rec["level"], "meta": meta or {}}


def _handler(cfg: dict):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, obj: dict):
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                self._send(200, {"ok": True, "upstream": cfg["upstream"],
                                 "model": cfg["model"],
                                 "batch_invariant": cfg["batch_invariant"],
                                 "signed": cfg["sign_key"] is not None})
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] not in ("/infer", "/v1/chat"):
                return self._send(404, {"ok": False, "error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                prompt = str(body.get("prompt") or "")
                if not prompt:
                    return self._send(400, {"ok": False, "error": "prompt required"})
                temp = float(body.get("temperature", 0.0))
                max_tokens = int(body.get("max_tokens", 512))
                output, meta = call_upstream(
                    cfg["upstream"], cfg["model"], prompt, cfg["api_key"],
                    cfg["timeout"], temp, max_tokens)
                res = signed_result(prompt, output, cfg["model"], cfg["sign_key"],
                                    cfg["key_id"], cfg["batch_invariant"], temp,
                                    max_tokens, meta=meta)
                self._send(200, res)
            except Exception as e:
                self._send(502, {"ok": False, "error": type(e).__name__ + ": " + str(e)})

    return H


def build_server(host: str = "127.0.0.1", port: int = 8094, upstream: str = "",
                 model: str = "unknown", api_key: str = "", timeout: float = 120.0,
                 sign_key: str | None = None, key_id: str | None = None,
                 batch_invariant: bool | None = None) -> ThreadingHTTPServer:
    if batch_invariant is None:
        batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    cfg = {"upstream": upstream, "model": model, "api_key": api_key,
           "timeout": timeout, "key_id": key_id, "batch_invariant": bool(batch_invariant),
           "sign_key": load_private(sign_key) if sign_key else None}
    return ThreadingHTTPServer((host, port), _handler(cfg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.inference_service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--upstream", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--sign-key", default="")
    ap.add_argument("--key-id", default="")
    ap.add_argument("--batch-invariant", action="store_true")
    a = ap.parse_args(argv)
    srv = build_server(a.host, a.port, a.upstream, a.model, a.api_key, a.timeout,
                       a.sign_key or None, a.key_id or None,
                       True if a.batch_invariant else None)
    print(json.dumps({"ok": True, "listen": "%s:%d" % srv.server_address[:2],
                      "model": a.model, "batch_invariant": a.batch_invariant,
                      "signed": bool(a.sign_key)}, ensure_ascii=False), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
