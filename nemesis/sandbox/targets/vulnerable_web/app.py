"""An INTENTIONALLY VULNERABLE lab web app. Isolated, authorized use only.

DO NOT deploy this anywhere reachable. It exists solely as a target for sandbox
evaluation on an internal, no-internet Docker network. The vulnerabilities are
deliberate and self-contained (no destructive primitive): a reflected-input
endpoint and a debug endpoint that leaks a hard-coded weak secret.

Endpoints (stdlib only, no dependencies, listens on :8080):

* ``/``          — info page.
* ``/health``    — liveness/health check -> 200 "ok". Used by the container
                   HEALTHCHECK and the sandbox ``service_health`` probe.
* ``/greet?name=`` — reflects ``name`` into HTML **unescaped** (reflected XSS).
                   A request whose ``name`` contains an attack marker like
                   ``<script>`` is the lab's *simulated* attack trigger: it bumps
                   a ``suspicious_requests`` counter. No code is executed.
* ``/metrics``   — exposes ``suspicious_requests`` so the defender sidecar can
                   detect the (simulated) intrusion over the internal network.
* ``/debug``     — leaks a hard-coded admin token (intentional info disclosure).
"""

from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# An intentionally weak, hard-coded secret — the lab "finding". Not real.
WEAK_ADMIN_TOKEN = "admin:admin123"  # noqa: S105 (deliberate lab vuln)

# Substrings that mark a request as the lab's simulated attack.
ATTACK_MARKERS = ("<script", "onerror=", "../", "';--", "<img")

_suspicious_requests = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter, still to stdout for log probes
        print("[vuln-web] " + (fmt % args), flush=True)

    def _send(self, code: int, body: str, content_type: str = "text/html") -> None:
        payload = body.encode("utf-8", "replace")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        global _suspicious_requests
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/health":
            self._send(200, "ok", "text/plain")
            return

        if path == "/metrics":
            self._send(
                200,
                f"suspicious_requests {_suspicious_requests}\n",
                "text/plain",
            )
            return

        if path == "/debug":
            # Intentional information disclosure of a weak secret.
            self._send(200, f"admin_token={WEAK_ADMIN_TOKEN}\n", "text/plain")
            return

        if path == "/greet":
            name = (query.get("name") or ["world"])[0]
            if any(m in name.lower() for m in ATTACK_MARKERS):
                _suspicious_requests += 1
                print(f"[vuln-web] SUSPICIOUS request on /greet: {name!r}", flush=True)
            # Reflected, UNESCAPED -> deliberate reflected-XSS vulnerability.
            self._send(200, f"<h1>Hello, {name}!</h1>")
            return

        self._send(
            200,
            "<h1>Vulnerable lab app</h1><p>Isolated target. "
            "Try /health, /greet?name=, /metrics, /debug.</p>",
        )


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[vuln-web] listening on :{port} (ISOLATED LAB TARGET)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
