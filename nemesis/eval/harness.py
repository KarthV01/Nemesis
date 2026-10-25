"""Run episodes of a CC4 :class:`EnvSpec` with a blue policy and score them.

The harness is the evaluation seam the whole platform shares: give it a blue
policy (random / scripted / trained — see :mod:`nemesis.eval.policies`) and an
:class:`~nemesis.platform.envs.EnvSpec`, it plays N episodes against the scripted
red and green opponents and returns one :class:`EpisodeMetrics` per episode plus
the :func:`~nemesis.platform.metrics.summarize` summary.

Metric derivation (``MetricProbe``) reads the *unwrapped* CybORG controller after
every step, so it is identical whether the policy drove the flat wrapper or the
raw simulator. See ``docs/evaluation.md`` for exactly how each metric is derived
from CC4 state and reward.

Everything here runs inside CybORG's simulated CC4 network; no real hosts or
traffic are involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Set

from nemesis.platform import envs as _envs
from nemesis.platform.envs import EnvSpec
from nemesis.platform.metrics import EpisodeMetrics, MetricsSummary, summarize


@dataclass
class EvalConfig:
    """How to run an evaluation.

    Attributes
    ----------
    spec : EnvSpec
        The scenario to evaluate on. Its ``wrapper`` is overridden to match the
        policy, and ``steps`` bounds the episode length.
    episodes : int
        Number of episodes to play.
    seed : int, optional
        Base seed. Episode ``i`` runs on ``seed + i`` (so episodes differ but the
        whole evaluation reproduces). ``None`` leaves the env to seed itself.
    win_reward_threshold : float, optional
        If set, ``blue_win = blue_reward >= win_reward_threshold`` for each
        episode. If ``None`` (default), ``blue_win`` stays ``None`` because CC4
        defines no intrinsic win condition (see docs).
    """

    spec: EnvSpec
    episodes: int = 10
    seed: Optional[int] = None
    win_reward_threshold: Optional[float] = None


# --------------------------------------------------------------------------- #
# Target-host classification
# --------------------------------------------------------------------------- #
def _is_target_host(hostname: str) -> bool:
    """True for attackable hosts: user/server hosts, not routers or the internet."""
    return "router" not in hostname and not hostname.startswith("root_internet_host")


# --------------------------------------------------------------------------- #
# Metric probe: reads ground truth from the unwrapped controller each step
# --------------------------------------------------------------------------- #
class MetricProbe:
    """Accumulates the CC4 metric vector across one episode.

    Call :meth:`observe` after each ``step`` (the environment having just
    advanced), then :meth:`finish` to get an :class:`EpisodeMetrics`.
    """

    def __init__(self, env):
        from CybORG.Shared.Enums import TernaryEnum

        self._TRUE = TernaryEnum.TRUE
        self._FALSE = TernaryEnum.FALSE

        self._ec = env.unwrapped.environment_controller
        self._red_agents = list(self._ec.team["Red"])
        self._green_agents = list(self._ec.team["Green"])
        self._targets = {h for h in self._ec.state.hosts if _is_target_host(h)}
        # Each green agent's own host (fixed for the episode).
        self._green_host: Dict[str, Optional[str]] = {}
        for g in self._green_agents:
            sessions = list(self._ec.state.sessions.get(g, {}).values())
            self._green_host[g] = sessions[0].hostname if sessions else None

        self.steps = 0
        self.blue_reward = 0.0
        self.green_attempts = 0
        self.green_successes = 0
        self.false_positives = 0
        self._ever_root: Set[str] = set()
        self._root_peak = 0
        self._last_root: Set[str] = set()
        self._last_foothold: Set[str] = set()

    def observe(self) -> None:
        ec = self._ec
        self.steps += 1

        # --- blue reward (canonical CC4 team reward for this step) ---
        self.blue_reward += float(ec.reward.get("Blue", {}).get("BlueRewardMachine", 0.0))

        # --- red footprint ---
        root_now: Set[str] = set()
        foothold_now: Set[str] = set()
        for r in self._red_agents:
            for s in ec.state.sessions.get(r, {}).values():
                foothold_now.add(s.hostname)
                if s.has_privileged_access():
                    root_now.add(s.hostname)
        root_targets = root_now & self._targets
        self._ever_root |= root_targets
        self._root_peak = max(self._root_peak, len(root_targets))
        self._last_root = root_targets
        self._last_foothold = foothold_now & self._targets

        # --- green availability + blue false positives ---
        for g in self._green_agents:
            acts = ec.action.get(g, [])
            if not acts or all(type(a).__name__ == "Sleep" for a in acts):
                continue  # not an availability attempt
            success = self._green_success(ec, g)
            if success is None:
                continue
            self.green_attempts += 1
            if success == self._TRUE:
                self.green_successes += 1
            elif success == self._FALSE:
                # Failed with no attacker on the user's host => blue caused it.
                if self._green_host.get(g) not in foothold_now:
                    self.false_positives += 1

    @staticmethod
    def _green_success(ec, agent):
        oset = ec.observation.get(agent)
        try:
            return oset.observations[0].data.get("success")
        except (AttributeError, IndexError, KeyError):
            return None

    def finish(self, episode: int, *, win_reward_threshold: Optional[float],
               extra: Optional[Dict] = None) -> EpisodeMetrics:
        n_targets = len(self._targets) or 1
        attack_success_rate = len(self._ever_root) / n_targets
        availability = (self.green_successes / self.green_attempts) if self.green_attempts else None
        blue_win = None if win_reward_threshold is None else (self.blue_reward >= win_reward_threshold)

        merged_extra = {
            "target_hosts": len(self._targets),
            "red_root_hosts_final": sorted(self._last_root),
            "red_foothold_hosts": sorted(self._last_foothold),
            "red_root_peak": self._root_peak / n_targets,
            "green_attempts": self.green_attempts,
            "green_failures": self.green_attempts - self.green_successes,
        }
        if extra:
            merged_extra.update(extra)

        return EpisodeMetrics(
            episode=episode,
            steps=self.steps,
            blue_reward=round(self.blue_reward, 3),
            red_reward=round(-self.blue_reward, 3),  # derived zero-sum proxy; see docs
            attack_success_rate=round(attack_success_rate, 4),
            defender_availability=round(availability, 4) if availability is not None else None,
            false_positives=self.false_positives,
            blue_win=blue_win,
            extra=merged_extra,
        )


# --------------------------------------------------------------------------- #
# Episode driver
# --------------------------------------------------------------------------- #
def run_episode(policy, spec: EnvSpec, episode: int, *,
                seed: Optional[int] = None,
                win_reward_threshold: Optional[float] = None) -> EpisodeMetrics:
    """Play one episode of ``spec`` with ``policy`` and return its metrics.

    ``spec.wrapper`` is forced to ``policy.wrapper`` so the policy gets a
    compatible environment. The episode runs until CybORG truncates (its step
    limit) or ``spec.steps`` iterations, whichever comes first.
    """
    spec = replace(spec, wrapper=policy.wrapper, seed=seed)
    env = _envs.build(spec)
    try:
        observations = _reset(env, seed)
        policy.reset(env)
        probe = MetricProbe(env)
        for _ in range(spec.steps):
            actions = policy.act(env, observations)
            observations, done = _step(env, actions)
            probe.observe()
            if done:
                break
        return probe.finish(
            episode,
            win_reward_threshold=win_reward_threshold,
            extra={"wrapper": policy.wrapper, "red_agent": spec.red_agent,
                   "blue_policy": getattr(policy, "name", type(policy).__name__),
                   "seed": seed},
        )
    finally:
        _close(env)


def evaluate(policy, config: EvalConfig,
             on_episode: Optional[Callable[[EpisodeMetrics], None]] = None) -> List[EpisodeMetrics]:
    """Run ``config.episodes`` episodes and return their metrics.

    ``on_episode`` (if given) is called with each :class:`EpisodeMetrics` as it is
    produced, e.g. to stream it to a :class:`~nemesis.platform.metrics.MetricsLog`.
    """
    results: List[EpisodeMetrics] = []
    for i in range(config.episodes):
        seed = None if config.seed is None else config.seed + i
        metrics = run_episode(
            policy, config.spec, i,
            seed=seed, win_reward_threshold=config.win_reward_threshold,
        )
        results.append(metrics)
        if on_episode is not None:
            on_episode(metrics)
    return results


def evaluate_summary(policy, config: EvalConfig) -> MetricsSummary:
    """Convenience: :func:`evaluate` then :func:`summarize`."""
    return summarize(evaluate(policy, config))


# --------------------------------------------------------------------------- #
# Reset/step/close normalised across the flat wrapper and raw CybORG
# --------------------------------------------------------------------------- #
def _is_raw(env) -> bool:
    return env is env.unwrapped


def _reset(env, seed):
    if _is_raw(env):
        env.reset(seed=seed)
        return None  # scripted policy reads obs from env directly
    observations, _info = env.reset(seed=seed)
    return observations


def _step(env, actions):
    """Return (observations, done). ``observations`` is None for the raw driver."""
    if _is_raw(env):
        env.parallel_step(actions)
        return None, bool(env.unwrapped.environment_controller.done)
    observations, _rew, _term, truncated, _info = env.step(actions)
    done = bool(truncated) if isinstance(truncated, bool) else any(truncated.values())
    return observations, done


def _close(env) -> None:
    try:
        env.close()
    except Exception:
        pass


__all__ = [
    "EvalConfig",
    "MetricProbe",
    "run_episode",
    "evaluate",
    "evaluate_summary",
]
