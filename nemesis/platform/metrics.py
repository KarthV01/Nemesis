"""Evaluation metrics for Nemesis episodes.

Win rate alone is misleading when both sides are still learning, so an
evaluation reports a small vector of outcomes per episode. These are the
quantities every evaluator (baseline harness, self-play league, sandbox runs)
fills in, so results from different stages are directly comparable.

The metrics are deliberately simulator-agnostic names; each evaluator documents
how it derives them from CybORG's reward and state. Reasonable defaults:

* ``blue_reward`` / ``red_reward`` — summed per-episode reward for each side.
* ``attack_success_rate`` — fraction of red objectives/hosts compromised.
* ``defender_availability`` — fraction of steps green (user) services stayed up.
* ``false_positives`` — blue actions that disrupted a host with no red presence.
* ``blue_win`` — whether blue met its success criterion this episode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import IO, Dict, List, Optional


@dataclass
class EpisodeMetrics:
    """Outcome of one evaluated episode. Unknown fields stay ``None``."""

    episode: int
    steps: int
    blue_reward: float
    red_reward: Optional[float] = None
    attack_success_rate: Optional[float] = None
    defender_availability: Optional[float] = None
    false_positives: Optional[int] = None
    blue_win: Optional[bool] = None
    #: Free-form extras an evaluator wants to keep (scenario id, opponent, ...).
    extra: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict) -> "EpisodeMetrics":
        known = {f.name for f in fields(cls)}
        core = {k: v for k, v in d.items() if k in known and k != "extra"}
        extra = {**d.get("extra", {}), **{k: v for k, v in d.items() if k not in known}}
        return cls(extra=extra, **core)


@dataclass
class MetricsSummary:
    """Mean of a batch of episodes; ``None`` where no episode supplied a value."""

    episodes: int
    blue_reward: float
    red_reward: Optional[float] = None
    attack_success_rate: Optional[float] = None
    defender_availability: Optional[float] = None
    false_positives: Optional[float] = None
    blue_win_rate: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def summarize(batch: List[EpisodeMetrics]) -> MetricsSummary:
    """Average a batch, skipping ``None`` entries field-by-field."""
    if not batch:
        raise ValueError("cannot summarize an empty batch")

    def mean(vals):
        present = [v for v in vals if v is not None]
        return (sum(present) / len(present)) if present else None

    win_vals = [1.0 if m.blue_win else 0.0 for m in batch if m.blue_win is not None]
    return MetricsSummary(
        episodes=len(batch),
        blue_reward=mean([m.blue_reward for m in batch]),
        red_reward=mean([m.red_reward for m in batch]),
        attack_success_rate=mean([m.attack_success_rate for m in batch]),
        defender_availability=mean([m.defender_availability for m in batch]),
        false_positives=mean([m.false_positives for m in batch]),
        blue_win_rate=(sum(win_vals) / len(win_vals)) if win_vals else None,
    )


class MetricsLog:
    """Append :class:`EpisodeMetrics` to a JSONL file and read them back.

    One JSON object per line, so a run can be tailed live and a crashed run
    still yields every episode written before it died.
    """

    def __init__(self, fh: IO):
        self._fh = fh

    def write(self, metrics: EpisodeMetrics) -> None:
        self._fh.write(metrics.to_json() + "\n")
        self._fh.flush()

    @staticmethod
    def read(fh: IO) -> List[EpisodeMetrics]:
        out = []
        for line in fh:
            line = line.strip()
            if line:
                out.append(EpisodeMetrics.from_dict(json.loads(line)))
        return out


__all__ = ["EpisodeMetrics", "MetricsSummary", "MetricsLog", "summarize"]
