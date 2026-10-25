"""Alternating-update self-play on one shared CybORG episode.

Each *generation* does two phases:

1. **Blue update** — the blue PPO team (five :class:`PPOLearner` policies, one
   per CC4 blue agent, reusing the pattern in ``scripts/train_ppo_team.py``)
   trains for a while against a frozen red opponent *sampled from the league*
   (newest with probability ``latest_prob``, otherwise a uniformly-drawn past
   one). The frozen red is baked into the scenario as its ``red_agent_class``.
2. **Red update** — symmetric: a red learner trains against a frozen blue
   opponent sampled from the league. The red learner lives on branch
   ``stage/red-training`` and is injected through :class:`MatchFactory` (see
   :mod:`nemesis.league.protocol`). Until it is wired, the loop uses the
   scripted-red fallback and *skips* the red update, logging where it plugs in.

After each generation the current blue weights are frozen into a new league
snapshot, and periodically blue is evaluated against a *set* of frozen red
opponents (not just the one it is currently training on) — the league-wide
evaluation that tells us whether blue is getting broadly better or merely
beating its latest sparring partner. All episodes are logged as
:class:`EpisodeMetrics` to ``experiments/<name>/`` through the
:class:`ArtifactStore`, and ``TRAIN``/``METRIC`` events are emitted for the
dashboard.

Everything runs inside CybORG's **simulated** CC4 network; agents emit only the
simulator's abstract action objects.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nemesis.platform.events import EventKind, EventLog
from nemesis.platform.metrics import EpisodeMetrics, MetricsLog, summarize
from nemesis.platform.storage import ArtifactStore, Checkpoint, Side
from nemesis.league.league import League, Opponent

DEFAULT_ARTIFACTS_ROOT = os.path.join("runs", "selfplay")  # under a .gitignore'd dir


def _posix_join(*parts: str) -> str:
    return PurePosixPath(*[str(p) for p in parts]).as_posix()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SelfPlayConfig:
    """Knobs for a self-play run. :meth:`smoke` returns a fast, tiny variant."""

    name: str = "selfplay"
    generations: int = 5
    #: env steps of blue training per generation (summed across the team)
    blue_steps_per_gen: int = 4000
    #: env steps of red training per generation (used by the red hook when wired)
    red_steps_per_gen: int = 4000
    episode_steps: int = 500
    #: transitions per PPO rollout / update, per blue learner
    n_steps: int = 2048
    batch_size: int = 256
    #: evaluate every this many generations (0 disables evaluation)
    eval_every: int = 1
    #: episodes per frozen opponent during evaluation
    eval_episodes: int = 3
    #: how many distinct frozen red opponents to evaluate blue against
    eval_opponents: int = 3
    #: opponent sampling: probability of drawing the newest snapshot
    latest_prob: float = 0.5
    #: scripted red agents forming the initial frozen red roster (class names
    #: resolvable by nemesis.platform.envs: CybORG or nemesis.agents classes)
    scripted_red_roster: Tuple[str, ...] = (
        "FiniteStateRedAgent",
        "RedCaptain",
        "NemesisRedAgent",
    )
    #: blue "wins" an episode if its team reward >= this; ``None`` leaves
    #: ``blue_win`` unlabeled (so the score table reports mean reward only).
    blue_win_threshold: Optional[float] = None
    seed: Optional[int] = None
    artifacts_root: str = DEFAULT_ARTIFACTS_ROOT
    verbose: int = 0

    @classmethod
    def smoke(cls, **overrides) -> "SelfPlayConfig":
        """A tiny configuration that exercises the whole loop in seconds."""
        base = dict(
            name="selfplay_smoke",
            generations=1,
            blue_steps_per_gen=40,
            red_steps_per_gen=40,
            episode_steps=20,
            n_steps=32,
            batch_size=16,
            eval_every=1,
            eval_episodes=1,
            eval_opponents=2,
            scripted_red_roster=("FiniteStateRedAgent", "RedCaptain"),
        )
        base.update(overrides)
        return cls(**base)

    def total_blue_timesteps(self) -> int:
        return self.generations * self.blue_steps_per_gen


# --------------------------------------------------------------------------- #
# Blue side — a team of PPOLearners trained via BlueFlatWrapper
# --------------------------------------------------------------------------- #
class BlueTeamMatch:
    """Blue PPO team playing one shared CybORG episode vs a frozen red opponent.

    The red opponent is a scripted ``red_agent_class`` baked into the scenario
    (CybORG steps it internally), so a frozen red here needs no external driver.
    Learners are owned by the factory and shared across matches; this object
    just drives one env with them.
    """

    def __init__(self, env, learners: Dict[str, "object"], opponent: Opponent, config: SelfPlayConfig, *, seed: Optional[int]):
        self.side = Side.BLUE
        self.env = env
        self.learners = learners  # {agent_name: PPOLearner}
        self.learner = learners   # protocol alias
        self.opponent = opponent
        self.config = config
        self._seed = seed
        self._episodes = 0

    def _next_seed(self) -> Optional[int]:
        if self._seed is None:
            return None
        s = self._seed + self._episodes
        self._episodes += 1
        return s

    def play_episode(self, *, learn: bool, deterministic: bool = True, events: Optional[EventLog] = None, episode: int = 0) -> EpisodeMetrics:
        env = self.env
        obs, _ = env.reset(seed=self._next_seed())
        team_reward = 0.0
        steps = 0
        updates = 0
        while True:
            agents = list(env.agents)
            if not agents:
                break
            if learn:
                actions = {a: self.learners[a].act(obs[a]) for a in agents}
            else:
                actions = {a: self.learners[a].predict(obs[a], deterministic=deterministic) for a in agents}
            obs, rewards, terminated, truncated, _ = env.step(actions)
            # CC4 blue reward is team-shared; mean over the acting agents is the
            # team's per-step reward (equal across agents when fully shared).
            team_reward += sum(float(rewards[a]) for a in agents) / len(agents)
            steps += 1
            if learn:
                for a in agents:
                    if self.learners[a].record(rewards[a], obs[a], terminated[a], truncated[a]):
                        updates += 1
            if all(terminated[a] or truncated[a] for a in agents):
                break
            if steps >= self.config.episode_steps:
                break

        win = None
        if self.config.blue_win_threshold is not None:
            win = team_reward >= self.config.blue_win_threshold
        if events is not None and learn and updates:
            events.log(EventKind.TRAIN, f"blue PPO update(s): {updates} (vs {self.opponent.name})", episode=episode, updates=updates)
        return EpisodeMetrics(
            episode=episode,
            steps=steps,
            blue_reward=team_reward,
            blue_win=win,
            extra={"opponent": self.opponent.name, "phase": "train" if learn else "eval"},
        )

    def save(self, checkpoint: Checkpoint) -> None:
        dst = checkpoint.local_dir()
        if dst is None:
            raise NotImplementedError("BlueTeamMatch.save needs a local backend; stage the copy for remote stores")
        for agent, learner in self.learners.items():
            learner.save(os.path.join(dst, agent), extra={"agent": agent, "opponent": self.opponent.name})

    def close(self) -> None:
        pass


class BlueMatchFactory:
    """Builds :class:`BlueTeamMatch` objects and owns the persistent blue team.

    Implements :class:`nemesis.league.protocol.MatchFactory` for ``Side.BLUE``.
    The learners are created lazily from the env's spaces on the first build and
    reused across every opponent and generation (they are the thing being
    trained); only the environment is rebuilt per opponent.
    """

    side = Side.BLUE
    can_learn = True

    def __init__(self, store: ArtifactStore, config: SelfPlayConfig):
        self.store = store
        self.config = config
        self._learners: Optional[Dict[str, object]] = None
        self._agents: Optional[List[str]] = None

    def _build_env(self, opponent: Opponent):
        from nemesis.platform.envs import EnvSpec, build

        if opponent.side != str(Side.RED):
            raise ValueError(f"blue trains against a RED opponent, got {opponent.side}")
        if opponent.is_learned:
            # A learned red opponent must be driven externally (red_env). The
            # BlueFlatWrapper path only supports scripted reds baked into the
            # scenario; wiring learned-red here is part of the red hook.
            raise NotImplementedError(
                f"learned red opponent {opponent.name!r} needs stage/red-training's red_env; "
                "the scripted-red fallback only consumes scripted red opponents"
            )
        spec = EnvSpec(
            steps=self.config.episode_steps,
            red_agent=opponent.ref,  # scripted red class name
            blue_agent="SleepAgent",
            wrapper="blue_flat",
            seed=self.config.seed,
        )
        return build(spec)

    def _ensure_learners(self, env) -> Dict[str, object]:
        if self._learners is not None:
            return self._learners
        from nemesis.ppo_learner import PPOLearner

        self._agents = list(env.possible_agents)
        learners: Dict[str, object] = {}
        for i, agent in enumerate(self._agents):
            learners[agent] = PPOLearner(
                env.observation_space(agent),
                env.action_space(agent),
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                total_timesteps=self.config.total_blue_timesteps(),
                seed=(self.config.seed + i) if self.config.seed is not None else None,
                verbose=self.config.verbose,
            )
        self._learners = learners
        return learners

    def build(self, opponent: Opponent) -> BlueTeamMatch:
        env = self._build_env(opponent)
        learners = self._ensure_learners(env)
        return BlueTeamMatch(env, learners, opponent, self.config, seed=self.config.seed)

    def save(self, checkpoint: Checkpoint) -> None:
        """Save every blue learner into ``checkpoint``'s dir, one subdir each."""
        if self._learners is None:
            return
        dst = checkpoint.local_dir()
        if dst is None:
            raise NotImplementedError("saving needs a local backend; stage the copy for remote stores")
        for agent, learner in self._learners.items():
            learner.save(os.path.join(dst, agent), extra={"agent": agent})


# --------------------------------------------------------------------------- #
# Red side — the integrator hook (scripted fallback does not learn)
# --------------------------------------------------------------------------- #
class ScriptedRedMatchFactory:
    """Placeholder red factory for before ``stage/red-training`` is wired.

    ``can_learn`` is ``False``, so :func:`run_selfplay` skips the red-update
    phase entirely and blue simply keeps training against the frozen scripted
    red roster. The real red learner replaces this object — see the big hook
    comment in :mod:`nemesis.league.protocol`.
    """

    side = Side.RED
    can_learn = False

    def build(self, opponent: Opponent):  # pragma: no cover - intentionally unused
        raise NotImplementedError(
            "No red learner wired. Provide a MatchFactory(side=Side.RED, can_learn=True) "
            "backed by stage/red-training's nemesis.red_env to run the red-update phase. "
            "See nemesis.league.protocol for the exact contract."
        )


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def run_selfplay(
    config: SelfPlayConfig,
    *,
    store: Optional[ArtifactStore] = None,
    league: Optional[League] = None,
    blue_factory: Optional["BlueMatchFactory"] = None,
    red_factory: Optional[object] = None,
    events: Optional[EventLog] = None,
) -> League:
    """Run alternating-update self-play and return the populated :class:`League`.

    Parameters mirror the module docstring. ``red_factory`` is the integrator
    hook: pass a learning red :class:`MatchFactory` to enable the red-update
    phase; the default scripted fallback skips it.
    """
    store = store or ArtifactStore(config.artifacts_root)
    league = league or League(store, name=config.name, seed=config.seed)
    rng = Random(config.seed)

    exp = config.name
    store.experiment_dir(exp)
    _write_run_config(store, config)

    ev_fh = None
    owns_events = events is None
    if owns_events:
        ev_fh = store.open_experiment_file(exp, "events.jsonl", "a")
        events = EventLog(ev_fh)

    blue_factory = blue_factory or BlueMatchFactory(store, config)
    red_factory = red_factory or ScriptedRedMatchFactory()
    red_can_learn = bool(getattr(red_factory, "can_learn", False))

    try:
        # Seed the frozen red roster (idempotent across resumed runs).
        added = league.register_scripted_roster(Side.RED, config.scripted_red_roster)
        events.log(
            EventKind.SYSTEM,
            f"self-play '{config.name}': {config.generations} generations, "
            f"red roster={list(config.scripted_red_roster)} ({len(added)} new), "
            f"red_learner={'wired' if red_can_learn else 'scripted-fallback'}",
        )

        best: Optional[Tuple[float, Opponent]] = None
        for g in range(1, config.generations + 1):
            events.log(EventKind.TRAIN, f"generation {g}/{config.generations} start", episode=g)

            # --- blue update phase -------------------------------------- #
            red_opp = league.sample(Side.RED, latest_prob=config.latest_prob, rng=rng)
            events.log(EventKind.TRAIN, f"gen {g}: blue trains vs frozen red {red_opp.name} ({red_opp.ref})", episode=g, opponent=red_opp.name)
            _train_phase(blue_factory, red_opp, config.blue_steps_per_gen, config, events, g)

            # --- freeze blue into a league snapshot --------------------- #
            blue_snap = league.register(
                Side.BLUE,
                "learned",
                meta={"generation": g, "trained_vs": red_opp.name, "episode_steps": config.episode_steps},
            )
            blue_factory.save(blue_snap.checkpoint(store))
            events.log(EventKind.TRAIN, f"gen {g}: froze blue -> {blue_snap.name}", episode=g, snapshot=blue_snap.name)

            # --- red update phase (hook) -------------------------------- #
            if red_can_learn:
                blue_opp = league.sample(Side.BLUE, latest_prob=config.latest_prob, rng=rng)
                if blue_opp is not None:
                    events.log(EventKind.TRAIN, f"gen {g}: red trains vs frozen blue {blue_opp.name}", episode=g, opponent=blue_opp.name)
                    red_snap = _red_update_phase(red_factory, blue_opp, config, events, g, store, league)
                    events.log(EventKind.TRAIN, f"gen {g}: froze red -> {red_snap.name}", episode=g, snapshot=red_snap.name)
            else:
                events.log(
                    EventKind.TRAIN,
                    f"gen {g}: red-update phase skipped (scripted-red fallback). "
                    "Wire stage/red-training's red_env via run_selfplay(red_factory=...) to enable it.",
                    episode=g,
                    hook="stage/red-training:nemesis.red_env",
                )

            # --- periodic league-wide evaluation ------------------------ #
            if config.eval_every and g % config.eval_every == 0:
                overall = _evaluate(blue_factory, league, config, events, store, generation=g, rng=rng)
                if overall is not None and (best is None or overall.blue_reward > best[0]):
                    best = (overall.blue_reward, blue_snap)

        # Promote the best (by eval) blue snapshot, else the latest.
        chosen = best[1] if best is not None else league.latest(Side.BLUE)
        if chosen is not None:
            store.promote_best(chosen.checkpoint(store), Side.BLUE)
            events.log(EventKind.TRAIN, f"promoted {chosen.name} -> blue/best_policy", snapshot=chosen.name)

        _write_standings(store, config, league)
        events.log(EventKind.SYSTEM, f"self-play '{config.name}' complete: {len(league)} snapshots")
    finally:
        if owns_events and ev_fh is not None:
            ev_fh.close()
    return league


def _train_phase(blue_factory, red_opp: Opponent, budget_steps: int, config: SelfPlayConfig, events: EventLog, generation: int) -> None:
    match = blue_factory.build(red_opp)
    steps = 0
    episode = 0
    try:
        while steps < budget_steps:
            m = match.play_episode(learn=True, events=events, episode=episode)
            steps += m.steps
            episode += 1
            events.log(EventKind.REWARD, f"gen {generation} train ep {episode}: blue_reward={m.blue_reward:.2f}", episode=episode, blue_reward=m.blue_reward, phase="train")
    finally:
        match.close()


def _red_update_phase(red_factory, blue_opp: Opponent, config: SelfPlayConfig, events: EventLog, generation: int, store: ArtifactStore, league: League) -> Opponent:
    """Run the red learner vs a frozen blue, then freeze red into the league.

    Only reached when the integrator supplies a learning red factory. The red
    learner, env construction and metrics come entirely from that factory's
    Match; this function just drives the budget and records the snapshot.
    """
    match = red_factory.build(blue_opp)
    steps = 0
    episode = 0
    try:
        while steps < config.red_steps_per_gen:
            m = match.play_episode(learn=True, events=events, episode=episode)
            steps += m.steps
            episode += 1
        red_snap = league.register(
            Side.RED,
            "learned",
            meta={"generation": generation, "trained_vs": blue_opp.name},
        )
        match.save(red_snap.checkpoint(store))
        return red_snap
    finally:
        match.close()


def _evaluate(blue_factory, league: League, config: SelfPlayConfig, events: EventLog, store: ArtifactStore, *, generation: int, rng: Random) -> Optional[EpisodeMetrics]:
    opponents = league.sample_eval_set(Side.RED, config.eval_opponents, rng=rng)
    if not opponents:
        return None
    metrics_fh = store.open_experiment_file(config.name, "evaluations.jsonl", "a")
    mlog = MetricsLog(metrics_fh)
    all_m: List[EpisodeMetrics] = []
    try:
        for opp in opponents:
            match = blue_factory.build(opp)
            batch: List[EpisodeMetrics] = []
            try:
                for e in range(config.eval_episodes):
                    m = match.play_episode(learn=False, deterministic=True, episode=e)
                    m.extra["generation"] = generation
                    mlog.write(m)
                    league.record_result(opp, m)
                    batch.append(m)
                    all_m.append(m)
            finally:
                match.close()
            s = summarize(batch)
            events.log(
                EventKind.METRIC,
                f"gen {generation} eval vs {opp.name}: blue_reward={s.blue_reward:.2f} over {s.episodes} ep",
                episode=generation,
                opponent=opp.name,
                blue_reward=s.blue_reward,
            )
    finally:
        metrics_fh.close()
    overall = summarize(all_m)
    events.log(
        EventKind.METRIC,
        f"gen {generation} eval summary: blue_reward={overall.blue_reward:.2f} over {overall.episodes} ep, {len(opponents)} opponents",
        episode=generation,
        **{k: v for k, v in overall.as_dict().items() if v is not None},
    )
    return overall


# --------------------------------------------------------------------------- #
# Reproducibility artifacts
# --------------------------------------------------------------------------- #
def _write_run_config(store: ArtifactStore, config: SelfPlayConfig) -> None:
    from nemesis.platform.envs import EnvSpec

    base_red = config.scripted_red_roster[0] if config.scripted_red_roster else "FiniteStateRedAgent"
    spec = EnvSpec(steps=config.episode_steps, red_agent=base_red, blue_agent="SleepAgent", wrapper="blue_flat", seed=config.seed)
    rel = _posix_join(store.experiment_dir(config.name), "config.json")
    store.write_json(rel, {"config": asdict(config), "env_spec": spec.as_dict()})


def _write_standings(store: ArtifactStore, config: SelfPlayConfig, league: League) -> None:
    rel = _posix_join(store.experiment_dir(config.name), "standings.json")
    store.write_json(rel, {s.opponent: s.to_dict() for s in league.standings()})


__all__ = [
    "SelfPlayConfig",
    "BlueTeamMatch",
    "BlueMatchFactory",
    "ScriptedRedMatchFactory",
    "run_selfplay",
]
