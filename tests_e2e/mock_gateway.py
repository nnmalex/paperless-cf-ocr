"""Minimal stand-in for Cloudflare AI Gateway's chat-completions routes (stdlib only).

POST <prefix>/workers-ai/v1/chat/completions and <prefix>/compat/chat/completions answer in
the OpenAI shape the plugin expects (with logprobs for Workers AI models). The transcription
is "E2E MOCK OCR <first 8 hex of the image's sha256>", so tests can tell pages apart.
GET /stats returns request counts and what auth headers were seen.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATS = {"requests": 0, "workers_ai": 0, "compat": 0, "authorization": 0, "aig": 0,
         "models": [], "user_agents": []}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # quiet
        pass

    def _json(self, code: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/stats":
            with LOCK:
                return self._json(200, STATS)
        if self.path == "/health":
            return self._json(200, {"ok": True})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        route = ("workers_ai" if self.path.endswith("/workers-ai/v1/chat/completions")
                 else "compat" if self.path.endswith("/compat/chat/completions") else None)
        if route is None:
            return self._json(404, {"error": {"message": f"unknown route {self.path}"}})
        if not (self.headers.get("Authorization") or self.headers.get("cf-aig-authorization")):
            return self._json(401, {"success": False, "errors": [{"code": 2009,
                                                                   "message": "Unauthorized"}]})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        url = body["messages"][0]["content"][1]["image_url"]["url"]
        digest = hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()[:8]
        text = f"E2E MOCK OCR {digest}"
        with LOCK:
            STATS["requests"] += 1
            STATS[route] += 1
            STATS["authorization"] += bool(self.headers.get("Authorization"))
            STATS["aig"] += bool(self.headers.get("cf-aig-authorization"))
            STATS["models"].append(body.get("model"))
            STATS["user_agents"].append(self.headers.get("User-Agent", ""))
        choice = {"index": 0, "message": {"role": "assistant", "content": text},
                  "finish_reason": "stop"}
        if body.get("logprobs"):
            choice["logprobs"] = {"content": [
                {"token": (" " if i else "") + w, "logprob": math.log(0.99)}
                for i, w in enumerate(text.split())]}
        return self._json(200, {"choices": [choice], "usage": {"prompt_tokens": 1000,
                                                               "completion_tokens": 5}})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
