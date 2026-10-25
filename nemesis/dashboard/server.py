"""The dashboard server: tail an events JSONL to SSE, serve metric time-series.

``http.server`` only (no web framework). The two data sources are the platform
JSONL contracts, so a live run and a replay are interchangeable:

* **Events.** :class:`EventTailer` reads an events JSONL incrementally (by byte
  offset, never re-reading, and never emitting a half-written final line).
  :class:`FileFeed` runs one tailer thread and fans the events out to every
  connected browser as SSE, mirroring the id/keepalive bookkeeping of
  ``nemesis.live.Feed`` so ``web/dashboard`` consumes them unchanged.
* **Metrics.** :func:`metrics_timeseries` reads a metrics JSONL fresh on each
  request and computes per-episode series plus rolling and cumulative views.

``--follow`` tails a file that is still being written (live run); without it the
server replays a finished file (drains it once, optionally paced by ``--delay``).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nemesis.platform.events import Event
from nemesis.platform.metrics import EpisodeMetrics, MetricsLog, summarize

WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web" / "dashboard"

#: Numeric per-episode metrics we build time-series for. ``blue_win`` is handled
#: separately (it is a boolean, surfaced as a running/rolling win *rate*).
SERIES_FIELDS = (
    "blue_reward",
    "red_reward",
    "attack_success_rate",
    "defender_availability",
    "false_positives",
)

DEFAULT_WINDOW = 20


# --------------------------------------------------------------------------- #
# Metrics time-series
# --------------------------------------------------------------------------- #
def read_metrics(path: Optional[str]) -> List[EpisodeMetrics]:
    """Read every :class:`EpisodeMetrics` from a JSONL file.

    Returns ``[]`` for a missing path (a live run may not have written the file
    yet) and tolerates a half-written final line (a row mid-flush is skipped
    until it is complete).
    """
    if not path or not os.path.exists(path):
        return []
    out: List[EpisodeMetrics] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(EpisodeMetrics.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                # Partial final line during a live flush, or a corrupt row.
                continue
    return out


def rolling_mean(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Trailing-window mean, skipping ``None`` gaps.

    Point ``i`` is the mean of the present values in ``values[i-window+1 .. i]``;
    ``None`` where the window holds no present value.
    """
    out: List[Optional[float]] = []
    buf: deque = deque(maxlen=window)
    for v in values:
        buf.append(v)
        present = [x for x in buf if x is not None]
        out.append(sum(present) / len(present) if present else None)
    return out


def _points(episodes: List[int], values: List[Optional[float]]) -> List[Dict[str, float]]:
    """Zip episodes+values into chart points, dropping ``None`` gaps."""
    return [
        {"x": e, "y": v}
        for e, v in zip(episodes, values)
        if v is not None
    ]


def metrics_timeseries(
    metrics: List[EpisodeMetrics], window: int = DEFAULT_WINDOW
) -> Dict[str, object]:
    """Compute the chart-ready time-series payload for ``/metrics``.

    Returns per-episode ``raw`` series and a trailing ``rolling`` mean for each
    numeric metric, plus the blue win series as a per-episode 0/1, a *cumulative*
    win rate (wins so far / episodes so far) and a rolling win rate — because a
    single headline win rate hides whether blue is actually improving or just
    riding early luck while both policies are still moving.
    """
    episodes = [m.episode for m in metrics]
    raw: Dict[str, List[Dict[str, float]]] = {}
    rolling: Dict[str, List[Dict[str, float]]] = {}

    for field in SERIES_FIELDS:
        vals = [getattr(m, field) for m in metrics]
        raw[field] = _points(episodes, vals)
        rolling[field] = _points(episodes, rolling_mean(vals, window))

    # Blue win: boolean per episode -> 0/1 raw, cumulative rate, rolling rate.
    win_vals: List[Optional[float]] = [
        None if m.blue_win is None else (1.0 if m.blue_win else 0.0) for m in metrics
    ]
    cumulative_win: List[Optional[float]] = []
    wins = 0
    seen = 0
    for v in win_vals:
        if v is not None:
            seen += 1
            wins += int(v)
        cumulative_win.append(wins / seen if seen else None)

    raw["blue_win"] = _points(episodes, win_vals)
    rolling["blue_win_rate"] = _points(episodes, rolling_mean(win_vals, window))
    cumulative = {"blue_win_rate": _points(episodes, cumulative_win)}

    summary = summarize(metrics).as_dict() if metrics else None

    return {
        "count": len(metrics),
        "window": window,
        "summary": summary,
        "raw": raw,
        "rolling": rolling,
        "cumulative": cumulative,
    }


# --------------------------------------------------------------------------- #
# Event tailing + fan-out
# --------------------------------------------------------------------------- #
class EventTailer:
    """Incremental, partial-line-safe reader of an events JSONL file.

    Tracks a byte offset so each :meth:`read_new` returns only events appended
    since the last call. A half-written final line (no trailing newline yet) is
    left unconsumed until the writer finishes it. If the file shrinks — a new run
    truncated or replaced it — the offset resets so the new run is picked up.
    """

    def __init__(self, path: str):
        self.path = path
        self._offset = 0

    def read_new(self) -> List[Event]:
        if not os.path.exists(self.path):
            return []
        size = os.path.getsize(self.path)
        if size < self._offset:  # file truncated / replaced
            self._offset = 0
        if size == self._offset:
            return []
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        last_nl = data.rfind(b"\n")
        if last_nl == -1:  # only a partial line so far
            return []
        complete = data[: last_nl + 1]
        self._offset += len(complete)
        events: List[Event] = []
        for raw in complete.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(Event.from_dict(json.loads(raw.decode("utf-8"))))
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                continue
        return events


class FileFeed:
    """Fan events read from a JSONL file out to every connected browser.

    Mirrors ``nemesis.live.Feed``: a backlog deque so a page connecting
    mid-stream still sees recent lines, and a :meth:`since` that blocks for new
    events. One background thread tails the file; ``follow`` keeps it polling for
    appended lines (live run), otherwise it drains the file once (replay).
    """

    def __init__(
        self,
        path: str,
        follow: bool = False,
        poll: float = 0.25,
        delay: float = 0.0,
        backlog: int = 1000,
    ):
        self._tailer = EventTailer(path)
        self._follow = follow
        self._poll = poll
        self._delay = delay
        self._cond = threading.Condition()
        self._events: deque = deque(maxlen=backlog)
        self._max_id = -1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _publish(self, event: Event) -> None:
        d = event.to_dict()
        # Events from EventLog carry their own id; fall back to a running id so
        # SSE Last-Event-ID reconnects still work for ad-hoc logs.
        if d.get("id") is None:
            d["id"] = self._max_id + 1
        if d.get("t") is None:
            d["t"] = time.time()
        with self._cond:
            self._events.append(d)
            self._max_id = max(self._max_id, int(d["id"]))
            self._cond.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            new = self._tailer.read_new()
            for event in new:
                if self._stop.is_set():
                    return
                self._publish(event)
                if self._delay:
                    self._stop.wait(self._delay)
            if not self._follow and not new:
                # Replay: one empty read after draining means we are done.
                return
            self._stop.wait(self._poll)

    def start(self) -> "FileFeed":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def since(self, last_id: int, timeout: float = 15.0) -> List[dict]:
        """Events with id greater than ``last_id``; blocks up to ``timeout``."""
        with self._cond:
            self._cond.wait_for(lambda: self._max_id > last_id, timeout=timeout)
            return [e for e in self._events if int(e["id"]) > last_id]


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
def make_handler(feed: FileFeed, metrics_path: Optional[str], window: int):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def log_message(self, *args):  # keep the console quiet
            pass

        # -- helpers ---------------------------------------------------------- #
        def _send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # -- routes ----------------------------------------------------------- #
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/events":
                return self._stream_events()
            if path == "/metrics":
                return self._serve_metrics()
            if path == "/health":
                return self._send_json({"ok": True, "metrics_count": len(read_metrics(metrics_path))})
            return super().do_GET()  # static files out of web/dashboard

        def _serve_metrics(self):
            try:
                payload = metrics_timeseries(read_metrics(metrics_path), window=window)
            except Exception as exc:  # pragma: no cover - defensive
                return self._send_json({"error": str(exc)}, status=500)
            return self._send_json(payload)

        def _stream_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_id = int(self.headers.get("Last-Event-ID", -1))
            try:
                while True:
                    events = feed.since(last_id)
                    if not events:
                        self.wfile.write(b": keepalive\n\n")
                    for event in events:
                        self.wfile.write(
                            f"id: {event['id']}\ndata: {json.dumps(event)}\n\n".encode("utf-8")
                        )
                        last_id = int(event["id"])
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # browser closed the tab

    return Handler


# --------------------------------------------------------------------------- #
# Server wiring
# --------------------------------------------------------------------------- #
def build_server(
    events_path: str,
    metrics_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8001,
    follow: bool = False,
    poll: float = 0.25,
    delay: float = 0.0,
    window: int = DEFAULT_WINDOW,
) -> Tuple[ThreadingHTTPServer, FileFeed]:
    """Build (but do not block on) the server and its started feed.

    Pass ``port=0`` for an ephemeral port (tests). The caller runs
    ``server.serve_forever()`` and calls ``feed.stop()`` / ``server.shutdown()``.
    """
    feed = FileFeed(events_path, follow=follow, poll=poll, delay=delay).start()
    server = ThreadingHTTPServer((host, port), make_handler(feed, metrics_path, window))
    server.daemon_threads = True
    return server, feed


def serve(
    events_path: str,
    metrics_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8001,
    follow: bool = False,
    poll: float = 0.25,
    delay: float = 0.0,
    window: int = DEFAULT_WINDOW,
) -> None:
    """Build, announce the URL, and serve until Ctrl+C."""
    server, feed = build_server(
        events_path, metrics_path, host, port, follow, poll, delay, window
    )
    actual_port = server.server_address[1]
    mode = "following live" if follow else "replaying"
    print(f"Nemesis dashboard on http://localhost:{actual_port}  ({mode})  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemesis observability dashboard server.")
    parser.add_argument("--events", required=True, help="events JSONL file to stream")
    parser.add_argument("--metrics", default=None, help="metrics JSONL file for the charts")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--follow", action="store_true", help="tail a live file instead of replaying")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between replayed events")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="rolling-mean window (episodes)")
    args = parser.parse_args()
    serve(
        args.events,
        args.metrics,
        host=args.host,
        port=args.port,
        follow=args.follow,
        delay=args.delay,
        window=args.window,
    )


if __name__ == "__main__":
    main()
