"""Nemesis platform layer — the shared contracts the training stages agree on.

The architecture has several moving parts built in parallel (red training, blue
training, self-play, Modal cloud jobs, sandbox evaluation, LLM agents, a live
dashboard). They stay compatible by depending on this layer instead of on each
other:

* :mod:`nemesis.platform.envs` — one place that builds CybORG environments, so
  every trainer and evaluator sees the same scenarios and wrappers.
* :mod:`nemesis.platform.storage` — the on-disk layout for checkpoints and
  experiment logs (``red/``, ``blue/``, ``selfplay/``, ``experiments/``), with a
  backend seam so the same code writes to a local disk or a Modal Volume.
* :mod:`nemesis.platform.metrics` — the evaluation metrics we track beyond raw
  reward (attack success rate, defender availability, false positives, ...).
* :mod:`nemesis.platform.events` — the structured episode-event stream the live
  dashboard and commentator render.

Nothing here imports CybORG at module load, so the storage/metrics/events
contracts can be used (and unit-tested) without the simulator installed.
"""

from nemesis.platform.metrics import EpisodeMetrics, MetricsLog
from nemesis.platform.events import Event, EventKind, EventLog
from nemesis.platform.storage import ArtifactStore, Side

__all__ = [
    "ArtifactStore",
    "Side",
    "EpisodeMetrics",
    "MetricsLog",
    "Event",
    "EventKind",
    "EventLog",
]
