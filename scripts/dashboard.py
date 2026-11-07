"""Launch the Nemesis observability dashboard.

Serves a terminal-style event feed plus evaluation metric charts, reading the
platform JSONL logs so it works for a live run (``--follow``) or a replay of a
finished run (the default). Needs no simulator — it only reads the logs.

Usage::

    # replay a finished run
    .venv\\Scripts\\python.exe scripts\\dashboard.py \\
        --events runs/demo/events.jsonl --metrics runs/demo/metrics.jsonl

    # follow a run that is still being written, pick a port
    .venv\\Scripts\\python.exe scripts\\dashboard.py \\
        --events runs/live/events.jsonl --metrics runs/live/metrics.jsonl \\
        --follow --port 8001

Then open the printed URL in a browser.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from a clean checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemesis.arena.controller import MatchManager
from nemesis.dashboard.server import DEFAULT_WINDOW, serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", default=None, help="legacy events JSONL file to stream")
    parser.add_argument("--metrics", default=None, help="metrics JSONL file for the charts")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--follow", action="store_true",
        help="tail a file that is still being written (live run); default replays a finished file",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="seconds between events when replaying (0 = stream instantly)",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help="rolling-mean window in episodes for the metric charts",
    )
    parser.add_argument("--arena", action="store_true", help="enable match selection and Docker job API")
    parser.add_argument("--arena-root", default="runs/arena", help="saved match/replay directory")
    parser.add_argument("--arena-image", default="nemesis-runtime:local", help="Docker image for matches")
    args = parser.parse_args()

    if not args.arena and not args.events:
        parser.error("--events is required unless --arena is enabled")
    events = args.events or str(Path(args.arena_root) / ".idle-events.jsonl")
    arena = MatchManager(args.arena_root, image=args.arena_image) if args.arena else None

    serve(
        events,
        args.metrics,
        host=args.host,
        port=args.port,
        follow=args.follow,
        delay=args.delay,
        window=args.window,
        arena=arena,
    )


if __name__ == "__main__":
    main()
