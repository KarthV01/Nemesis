"""The seam between the self-play loop and a side's trainer.

The self-play loop (:mod:`nemesis.league.selfplay_loop`) does not know how blue
or red actually train. It knows only how to:

1. ask a :class:`MatchFactory` for one side to ``build`` a :class:`Match` that
   pits that side's learner against a frozen ``Opponent`` from the league, and
2. tell the match to ``play_episode`` (learning or just evaluating), getting
   back :class:`~nemesis.platform.metrics.EpisodeMetrics`, and to ``save`` the
   learner's weights into a checkpoint directory.

Blue is implemented fully in :mod:`nemesis.league.selfplay_loop`
(``BlueMatchFactory`` / ``BlueTeamMatch``) by reusing :class:`PPOLearner`.

The RED learner is being built in parallel on branch ``stage/red-training``
(``nemesis/red_env.py`` with a *make-red* interface) and is not importable here
yet, so RED is expressed purely as this protocol:

    ┌─────────────────────── THE RED-LEARNER HOOK ───────────────────────┐
    │  The integrator provides an object implementing `MatchFactory` with │
    │  `side == Side.RED`. Its `.build(blue_opponent)` must:              │
    │    * construct stage/red-training's red-controllable env            │
    │      (`nemesis.red_env`) with BLUE loaded *frozen* from             │
    │      `blue_opponent.ref` (a learned checkpoint dir) or, for a       │
    │      scripted blue, `blue_opponent.ref` as a blue agent class name; │
    │    * wrap a red `PPOLearner` (or any learner with act/record/       │
    │      predict/save) driving the red agent(s) on that env;            │
    │    * return a `Match` whose `play_episode(learn=True)` updates red   │
    │      and whose `save(checkpoint)` writes red weights into the        │
    │      checkpoint dir so the league can freeze them next generation.  │
    │  Set `can_learn = True`. The loop then runs the red-update phase    │
    │  (red learner vs sampled frozen blue) symmetrically with the blue   │
    │  phase, and registers each red snapshot as a learned RED opponent.  │
    │                                                                      │
    │  Pass the factory to `run_selfplay(..., red_factory=<yours>)`.      │
    │  Until then the loop uses the scripted-red fallback                  │
    │  (`ScriptedRedMatchFactory`, `can_learn = False`): red never        │
    │  updates, but blue still trains against the frozen scripted red     │
    │  roster, so the whole loop runs end-to-end today.                   │
    └──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from nemesis.platform.events import EventLog
from nemesis.platform.metrics import EpisodeMetrics
from nemesis.platform.storage import Checkpoint, Side
from nemesis.league.league import Opponent


@runtime_checkable
class Match(Protocol):
    """One side's learner playing a shared CybORG episode vs a frozen opponent."""

    #: The side being trained in this match (``Side.BLUE`` / ``Side.RED``).
    side: Side
    #: The trainable policy (a ``PPOLearner`` or team thereof), or ``None`` for a
    #: non-learning fallback. The loop never pokes inside it.
    learner: object

    def play_episode(
        self,
        *,
        learn: bool,
        deterministic: bool = True,
        events: Optional[EventLog] = None,
        episode: int = 0,
    ) -> EpisodeMetrics:
        """Run one episode. ``learn`` updates the policy; otherwise act greedily.

        ``deterministic`` only applies when ``learn`` is ``False`` (evaluation).
        """

    def save(self, checkpoint: Checkpoint) -> None:
        """Persist the learner's weights into ``checkpoint``'s directory."""

    def close(self) -> None:
        """Release the underlying environment."""


@runtime_checkable
class MatchFactory(Protocol):
    """Builds :class:`Match` objects for one side against league opponents."""

    #: The side this factory trains.
    side: Side
    #: Whether this factory's matches actually update weights. The scripted-red
    #: fallback sets this ``False`` so the loop skips the red-update phase.
    can_learn: bool

    def build(self, opponent: Opponent) -> Match:
        """A match where this side's learner faces the frozen ``opponent``."""


__all__ = ["Match", "MatchFactory"]
