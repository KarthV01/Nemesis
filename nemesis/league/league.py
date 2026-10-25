"""A frozen-opponent league for adversarial self-play.

In self-play, a policy that only ever trains against the *latest* version of its
rival tends to overfit: it learns to beat one specific opponent and forgets how
to handle the strategies it already defeated. The fix is a **league** — we keep
a snapshot of each side's policy at every generation, freeze it, and keep making
the current learner play a mixture of the latest snapshot and random older ones.

This module owns the bookkeeping of that league, and nothing else: it does not
touch CybORG, SB3 or torch, so it imports instantly and is unit-testable with a
fake policy. A *snapshot* is just a small manifest (:class:`Opponent`) recorded
through :class:`~nemesis.platform.storage.ArtifactStore`:

* ``kind="learned"`` — a policy whose weights live in a checkpoint directory
  (``ref`` is the store-relative path to that directory).
* ``kind="scripted"`` — a rule-based agent with no weights (``ref`` is the agent
  *class name* resolvable by :mod:`nemesis.platform.envs`, e.g.
  ``"FiniteStateRedAgent"``). Scripted opponents are the ready-made,
  already-frozen red roster the self-play loop uses before a learned red exists.

Each registration allocates a ``selfplay/generation_NNNNNN`` directory via
``ArtifactStore.new_generation`` and drops an ``opponent.json`` manifest in it.
The league also keeps a tiny win/score table (:class:`Standing`) built from
:class:`~nemesis.platform.metrics.EpisodeMetrics`, persisted alongside the
generations as ``selfplay/<name>.json`` so a resumed run sees the whole history.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Union

from nemesis.platform.metrics import EpisodeMetrics
from nemesis.platform.storage import ArtifactStore, Checkpoint, Side

OPPONENT_MANIFEST = "opponent.json"

SideLike = Union[Side, str]


def _posix_join(*parts: str) -> str:
    return PurePosixPath(*[str(p) for p in parts]).as_posix()


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Opponent:
    """One frozen policy snapshot in the league.

    Attributes
    ----------
    side : str
        The side this policy *plays* — ``"red"`` or ``"blue"``. A red opponent
        is something blue trains against, and vice versa.
    generation : int
        The ``generation_NNNNNN`` number allocated for this snapshot.
    name : str
        Stable identifier, e.g. ``"red_gen_000003"``.
    kind : str
        ``"learned"`` (weights in a checkpoint dir) or ``"scripted"`` (rule-based
        agent, no weights).
    ref : str
        For ``learned`` snapshots, the store-relative path to the checkpoint
        directory. For ``scripted`` snapshots, the agent class name.
    gen_dir : str
        Store-relative path of the generation directory holding the manifest.
    created : float
        ``time.time()`` at registration.
    meta : dict
        Free-form extras (episode length, the opponent it was trained against,
        eval score at snapshot time, ...).
    """

    side: str
    generation: int
    name: str
    kind: str
    ref: str
    gen_dir: str
    created: float = field(default_factory=time.time)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def is_learned(self) -> bool:
        return self.kind == "learned"

    def checkpoint(self, store: ArtifactStore) -> Checkpoint:
        """Handle to this snapshot's checkpoint directory (learned snapshots)."""
        if not self.is_learned:
            raise ValueError(f"{self.name} is {self.kind}, not a learned checkpoint")
        return Checkpoint(store, self.ref)

    def local_weights_dir(self, store: ArtifactStore) -> Optional[str]:
        """Local filesystem path of the weights dir, or ``None`` on a remote backend."""
        return store.backend.local_path(self.ref) if self.is_learned else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "side": self.side,
            "generation": self.generation,
            "name": self.name,
            "kind": self.kind,
            "ref": self.ref,
            "gen_dir": self.gen_dir,
            "created": self.created,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Opponent":
        return cls(
            side=str(d["side"]),
            generation=int(d["generation"]),
            name=d["name"],
            kind=d["kind"],
            ref=d["ref"],
            gen_dir=d["gen_dir"],
            created=float(d.get("created", 0.0)),
            meta=dict(d.get("meta", {})),
        )


# --------------------------------------------------------------------------- #
# Win / score table
# --------------------------------------------------------------------------- #
@dataclass
class Standing:
    """Running record of how the *learner* fared against one frozen opponent.

    "Learner" is whichever side is training; against a red opponent that is
    blue, so ``wins`` counts episodes blue won. Rewards are accumulated so means
    can be reported without keeping every episode.
    """

    opponent: str
    side: str  # the side the opponent plays
    games: int = 0
    win_labeled_games: int = 0
    wins: int = 0
    blue_reward_sum: float = 0.0
    red_reward_sum: float = 0.0
    red_reward_games: int = 0

    def update(self, m: EpisodeMetrics) -> None:
        self.games += 1
        self.blue_reward_sum += float(m.blue_reward)
        if m.red_reward is not None:
            self.red_reward_sum += float(m.red_reward)
            self.red_reward_games += 1
        if m.blue_win is not None:
            self.win_labeled_games += 1
            if m.blue_win:
                self.wins += 1

    @property
    def mean_blue_reward(self) -> Optional[float]:
        return self.blue_reward_sum / self.games if self.games else None

    @property
    def mean_red_reward(self) -> Optional[float]:
        return self.red_reward_sum / self.red_reward_games if self.red_reward_games else None

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.win_labeled_games if self.win_labeled_games else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "opponent": self.opponent,
            "side": self.side,
            "games": self.games,
            "win_labeled_games": self.win_labeled_games,
            "wins": self.wins,
            "blue_reward_sum": self.blue_reward_sum,
            "red_reward_sum": self.red_reward_sum,
            "red_reward_games": self.red_reward_games,
            "mean_blue_reward": self.mean_blue_reward,
            "mean_red_reward": self.mean_red_reward,
            "win_rate": self.win_rate,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Standing":
        return cls(
            opponent=d["opponent"],
            side=str(d["side"]),
            games=int(d.get("games", 0)),
            win_labeled_games=int(d.get("win_labeled_games", 0)),
            wins=int(d.get("wins", 0)),
            blue_reward_sum=float(d.get("blue_reward_sum", 0.0)),
            red_reward_sum=float(d.get("red_reward_sum", 0.0)),
            red_reward_games=int(d.get("red_reward_games", 0)),
        )


# --------------------------------------------------------------------------- #
# League
# --------------------------------------------------------------------------- #
class League:
    """Register frozen snapshots, sample opponents, and keep a score table.

    Parameters
    ----------
    store : ArtifactStore
        Where generation directories and the league state file live.
    name : str
        State-file basename under ``selfplay/`` (``selfplay/<name>.json``).
    seed : int, optional
        Seeds the default RNG used by :meth:`sample`.
    """

    def __init__(self, store: ArtifactStore, *, name: str = "league", seed: Optional[int] = None):
        self.store = store
        self.name = name
        self._rng = random.Random(seed)
        self._opponents: List[Opponent] = []
        self._standings: Dict[str, Standing] = {}
        self._load()

    # -- persistence -------------------------------------------------------- #
    @property
    def state_path(self) -> str:
        return _posix_join(str(Side.SELFPLAY), f"{self.name}.json")

    def _load(self) -> None:
        if not self.store.backend.exists(self.state_path):
            return
        state = self.store.read_json(self.state_path)
        self._opponents = [Opponent.from_dict(d) for d in state.get("opponents", [])]
        self._standings = {
            k: Standing.from_dict(v) for k, v in state.get("standings", {}).items()
        }

    def _save(self) -> None:
        self.store.write_json(
            self.state_path,
            {
                "name": self.name,
                "opponents": [o.to_dict() for o in self._opponents],
                "standings": {k: v.to_dict() for k, v in self._standings.items()},
            },
        )

    # -- registration ------------------------------------------------------- #
    def register(
        self,
        side: SideLike,
        kind: str,
        ref: Optional[str] = None,
        *,
        meta: Optional[Dict[str, object]] = None,
    ) -> Opponent:
        """Allocate a generation dir and record a frozen snapshot.

        A fresh ``selfplay/generation_NNNNNN`` dir is always allocated. For a
        ``learned`` snapshot with no explicit ``ref``, the checkpoint directory
        defaults to ``<generation>/<side>`` and is returned on the
        :class:`Opponent` (via :meth:`Opponent.local_weights_dir`) so the caller
        can write weights there. For a ``scripted`` snapshot, ``ref`` is the
        required agent class name.
        """
        if kind not in ("learned", "scripted"):
            raise ValueError(f"kind must be 'learned' or 'scripted', got {kind!r}")
        side = str(side)

        gen_ckpt = self.store.new_generation()
        generation = int(gen_ckpt.name.rsplit("_", 1)[-1])
        gen_dir = gen_ckpt.rel_path

        if kind == "scripted":
            if not ref:
                raise ValueError("a scripted opponent needs an agent class name as ref")
        else:
            ref = ref or _posix_join(gen_dir, side)
            self.store.backend.makedirs(ref)

        opp = Opponent(
            side=side,
            generation=generation,
            name=f"{side}_gen_{generation:06d}",
            kind=kind,
            ref=ref,
            gen_dir=gen_dir,
            created=time.time(),
            meta=dict(meta or {}),
        )
        self.store.write_json(_posix_join(gen_dir, OPPONENT_MANIFEST), opp.to_dict())
        self._opponents.append(opp)
        self._save()
        return opp

    def register_scripted_roster(self, side: SideLike, class_names) -> List[Opponent]:
        """Register any roster class names not already present for ``side``.

        Idempotent: re-running a loop does not duplicate the scripted roster.
        """
        side = str(side)
        have = {o.ref for o in self._opponents if o.side == side and o.kind == "scripted"}
        added = []
        for cls_name in class_names:
            if cls_name not in have:
                added.append(self.register(side, "scripted", cls_name))
        return added

    # -- queries ------------------------------------------------------------ #
    def opponents(self, side: Optional[SideLike] = None) -> List[Opponent]:
        if side is None:
            return list(self._opponents)
        side = str(side)
        return [o for o in self._opponents if o.side == side]

    def latest(self, side: SideLike) -> Optional[Opponent]:
        pool = self.opponents(side)
        return max(pool, key=lambda o: o.generation) if pool else None

    def __len__(self) -> int:
        return len(self._opponents)

    # -- sampling ----------------------------------------------------------- #
    def sample(
        self,
        side: SideLike,
        *,
        latest_prob: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> Optional[Opponent]:
        """Sample a frozen opponent for ``side``.

        With probability ``latest_prob`` return the newest snapshot; otherwise
        return one drawn uniformly from the whole historical pool (the newest
        included). This is the prioritized-fictitious-self-play mixture: most
        games are against the current best rival, but a steady fraction revisit
        older strategies so the learner cannot forget how to beat them.
        """
        pool = self.opponents(side)
        if not pool:
            return None
        r = rng or self._rng
        if len(pool) == 1 or r.random() < latest_prob:
            return max(pool, key=lambda o: o.generation)
        return r.choice(pool)

    def sample_eval_set(
        self,
        side: SideLike,
        k: int,
        *,
        rng: Optional[random.Random] = None,
    ) -> List[Opponent]:
        """Up to ``k`` distinct frozen opponents for evaluation.

        Always includes the latest snapshot, then fills the rest uniformly from
        the historical pool, so evaluation never collapses to only the current
        training opponent.
        """
        pool = self.opponents(side)
        if not pool:
            return []
        r = rng or self._rng
        latest = max(pool, key=lambda o: o.generation)
        chosen = [latest]
        rest = [o for o in pool if o.name != latest.name]
        r.shuffle(rest)
        chosen.extend(rest[: max(0, k - 1)])
        return chosen

    # -- scoring ------------------------------------------------------------ #
    def record_result(self, opponent: Opponent, metrics: EpisodeMetrics) -> Standing:
        """Fold one evaluated episode into ``opponent``'s standing."""
        st = self._standings.get(opponent.name)
        if st is None:
            st = Standing(opponent=opponent.name, side=opponent.side)
            self._standings[opponent.name] = st
        st.update(metrics)
        self._save()
        return st

    def standing(self, opponent: Union[Opponent, str]) -> Optional[Standing]:
        key = opponent.name if isinstance(opponent, Opponent) else opponent
        return self._standings.get(key)

    def standings(self, side: Optional[SideLike] = None) -> List[Standing]:
        """Score table, worst mean-blue-reward first (hardest opponents on top)."""
        rows = list(self._standings.values())
        if side is not None:
            side = str(side)
            rows = [s for s in rows if s.side == side]
        return sorted(rows, key=lambda s: (s.mean_blue_reward is None, s.mean_blue_reward or 0.0))


__all__ = ["League", "Opponent", "Standing", "OPPONENT_MANIFEST"]
