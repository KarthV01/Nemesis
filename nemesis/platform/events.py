"""Structured episode events — the stream the dashboard and commentator render.

The existing live feed (:mod:`nemesis.live`) already pushes dict events to the
browser as ``{"id", "t", "kind", "text", ...}``. This module makes that shape a
typed contract so cloud training jobs can emit the *same* events to a log file,
and the dashboard can replay a finished run or follow a live one without caring
which produced it.

An :class:`Event` serializes to exactly the dict ``nemesis.live`` already sends,
so the web page needs no changes to consume logged events.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import IO, Dict, List, Optional


class EventKind(str, Enum):
    """The ``kind`` tag the dashboard styles each line by."""

    SYSTEM = "system"      # episode start/stop, topology notes
    RED = "red"            # an attacker move
    BLUE = "blue"          # a defender move
    REWARD = "reward"      # a reward readout
    TRAIN = "train"        # a learning update / checkpoint
    METRIC = "metric"      # an evaluation metric readout

    def __str__(self) -> str:
        return self.value


@dataclass
class Event:
    """One line of the feed.

    ``id`` and ``t`` are assigned by :class:`EventLog` when the event is emitted
    if left unset, so callers usually build ``Event(kind=..., text=...)``.
    """

    kind: EventKind
    text: str
    id: Optional[int] = None
    t: Optional[float] = None
    #: episode this belongs to, when applicable
    episode: Optional[int] = None
    #: any extra keys the web page or analysis wants (reward values, host, ...)
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        d = {"id": self.id, "t": self.t, "kind": str(self.kind), "text": self.text}
        if self.episode is not None:
            d["episode"] = self.episode
        d.update(self.extra)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict) -> "Event":
        reserved = {"id", "t", "kind", "text", "episode"}
        return cls(
            kind=EventKind(d["kind"]),
            text=d["text"],
            id=d.get("id"),
            t=d.get("t"),
            episode=d.get("episode"),
            extra={k: v for k, v in d.items() if k not in reserved},
        )


class EventLog:
    """Assign ids/timestamps and append events to a JSONL file.

    Mirrors the id/timestamp bookkeeping in ``nemesis.live.Feed`` so a logged
    run and a live run are interchangeable to the dashboard.
    """

    def __init__(self, fh: Optional[IO] = None):
        self._fh = fh
        self._next_id = 0

    def emit(self, event: Event) -> Event:
        if event.id is None:
            event.id = self._next_id
            self._next_id += 1
        if event.t is None:
            event.t = time.time()
        if self._fh is not None:
            self._fh.write(event.to_json() + "\n")
            self._fh.flush()
        return event

    def log(self, kind: EventKind, text: str, episode: Optional[int] = None, **extra) -> Event:
        return self.emit(Event(kind=kind, text=text, episode=episode, extra=extra))

    @staticmethod
    def read(fh: IO) -> List[Event]:
        out = []
        for line in fh:
            line = line.strip()
            if line:
                out.append(Event.from_dict(json.loads(line)))
        return out


__all__ = ["Event", "EventKind", "EventLog"]
