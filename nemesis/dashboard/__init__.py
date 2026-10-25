"""Nemesis observability dashboard — a richer, parallel view to ``nemesis.live``.

The live feed (:mod:`nemesis.live`) streams a running self-play game to a
terminal-style page. This dashboard consumes the *logged* contracts instead
(:mod:`nemesis.platform.events` and :mod:`nemesis.platform.metrics`), so the
same page works for a **live run it is following** and for a **replay of a
finished run** — anything that writes the JSONL logs, laptop or cloud job.

Two data sources, two endpoints:

* ``/events`` — Server-Sent Events, exactly the dict shape ``nemesis.live``
  sends the browser, sourced by tailing an events JSONL file
  (:class:`EventTailer` does the incremental, partial-line-safe read).
* ``/metrics`` — JSON time-series computed from a metrics JSONL file
  (:func:`metrics_timeseries`): reward per side, attack success rate, defender
  availability, false positives, and blue win rate — because win rate alone is
  misleading while both sides are still learning.

Nothing here imports CybORG; it only reads logs, so it unit-tests without the
simulator installed.
"""

from nemesis.dashboard.server import (
    EventTailer,
    build_server,
    metrics_timeseries,
    read_metrics,
    serve,
)

__all__ = [
    "EventTailer",
    "build_server",
    "metrics_timeseries",
    "read_metrics",
    "serve",
]
