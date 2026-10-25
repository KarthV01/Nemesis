"""A minimal DEFENSIVE monitoring sidecar for the lab. Isolated use only.

This is the "one defensive service" of the lab: it watches the vulnerable target
over the internal network and reports two things the sandbox verifier needs,
printing them to stdout as markers so the Nemesis ``collect_logs`` probe can read
them with ``docker logs`` (no ports are published anywhere):

* ``SERVICE-UP`` / ``SERVICE-DOWN`` — did the target's ``/health`` respond?
* ``INTRUSION-DETECTED`` — has the target seen a (simulated) attack, i.e. its
  ``/metrics`` ``suspicious_requests`` counter is above zero?

It also serves its own ``/health`` so the defender container has a HEALTHCHECK,
and ``/status`` (JSON) for convenience. It performs **no offensive action** — it
only observes and reports. Everything is stdlib; no dependencies.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET_URL = os.environ.get("TARGET_URL", "http://vulnerable-web:8080")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "5"))

_state = {"service_up": False, "intrusion_detected": False}
_lock = threading.Lock()


def _get(path: str, timeout: float = 2.0) -> str:
    with urllib.request.urlopen(TARGET_URL + path, timeout=timeout) as resp:
        if resp.status != 200:
            raise OSError(f"status {resp.status}")
        return resp.read().decode("utf-8", "replace")


def _poll_once() -> None:
    try:
        _get("/health")
        service_up = True
    except Exception:
        service_up = False

    intrusion = False
    if service_up:
        try:
            body = _get("/metrics")
            for line in body.splitlines():
                if line.startswith("suspicious_requests"):
                    intrusion = int(line.split()[1]) > 0
        except Exception:
            pass

    with _lock:
        _state["service_up"] = service_up
        _state["intrusion_detected"] = _state["intrusion_detected"] or intrusion

    print("SERVICE-UP" if service_up else "SERVICE-DOWN", flush=True)
    if intrusion:
        print("INTRUSION-DETECTED suspicious traffic observed at target", flush=True)


def _poll_loop() -> None:
    while True:
        _poll_once()
        time.sleep(POLL_SECONDS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep stdout clean for log probes
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, "ok", "text/plain")
            return
        if self.path == "/status":
            with _lock:
                body = json.dumps(dict(_state))
            self._send(200, body, "application/json")
            return
        self._send(404, "not found", "text/plain")

    def _send(self, code: int, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    threading.Thread(target=_poll_loop, daemon=True).start()
    port = int(os.environ.get("PORT", "9090"))
    print(f"[defender] monitoring {TARGET_URL}, serving :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
