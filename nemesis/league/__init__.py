"""Frozen-opponent league for Nemesis adversarial self-play.

``nemesis.league`` keeps red and blue policies improving against each other
without overfitting to the latest rival, by playing a mixture of the newest and
older *frozen* snapshots (a league).

Import layout
-------------
* :mod:`nemesis.league.league` — the league bookkeeping (``League``,
  ``Opponent``, ``Standing``). Pure logic: no CybORG, SB3 or torch, so it
  imports instantly and is unit-testable with a fake policy.
* :mod:`nemesis.league.protocol` — the ``Match`` / ``MatchFactory`` seam the
  self-play loop drives, and the documented RED-learner hook for the integrator.
* :mod:`nemesis.league.selfplay_loop` — the alternating-update training loop and
  the blue PPO implementation. Imported explicitly (it pulls in SB3/torch), so
  it is *not* re-exported here.
"""

from nemesis.league.league import League, Opponent, Standing
from nemesis.league.protocol import Match, MatchFactory

__all__ = ["League", "Opponent", "Standing", "Match", "MatchFactory"]
