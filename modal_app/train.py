"""PPO training on Modal (and locally) that persists through ``ArtifactStore``.

The real work lives in :func:`run_training` — a *plain* callable with no Modal
dependency. It reuses the existing team-training logic (one
:class:`nemesis.ppo_learner.PPOLearner` per blue agent, all acting every step in
one shared CybORG episode) and writes checkpoints through
``ArtifactStore(backend)``. The Modal functions :func:`train_blue` /
:func:`train_red` are thin wrappers that build a :class:`VolumeBackend` over the
mounted Volume and call :func:`run_training`; a local ``--dry-run`` calls the
very same function with a :class:`LocalBackend`. The logic is therefore fully
tested off-Modal.

``modal`` is imported only inside the guarded block at the bottom, so this
module imports and its functions run with ``modal`` absent.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np

from nemesis.platform import ArtifactStore, Side
from nemesis.platform.metrics import EpisodeMetrics, summarize
from nemesis.ppo_learner import PPOLearner

from modal_app.images import (
    APP_NAME,  # noqa: F401  (re-exported for callers)
    ARTIFACTS_PATH,
    MODAL_AVAILABLE,
    VOLUME_MOUNT_PATH,
    app,
    image,
    volume,
)
from modal_app.volume_backend import VolumeBackend, maybe_commit


# --------------------------------------------------------------------------- #
# A tiny multi-agent env for local --dry-run / tests (CybORG build is ~30 s).
# --------------------------------------------------------------------------- #
class DummyTeamEnv:
    """A fast stand-in with CC4's ``BlueFlatWrapper`` parallel interface.

    Same shape as the real env: ``possible_agents`` / ``agents``, per-agent
    ``observation_space`` / ``action_space``, ``reset() -> (obs, info)`` and
    ``step(actions) -> (obs, rewards, terminated, truncated, infos)`` as dicts,
    with episodes ending only by **truncation** at a step limit (as every CC4
    episode does). Reward is team-shared and mildly learnable (favouring action
    0), so a PPO update has a real signal. No CybORG needed.
    """

    def __init__(self, n_agents: int = 2, obs_dim: int = 4, n_actions: int = 2,
                 episode_len: int = 10, seed: Optional[int] = None):
        self.possible_agents = [f"blue_agent_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)
        self._obs_dim = obs_dim
        self._spaces_obs = gym.spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self._spaces_act = gym.spaces.Discrete(n_actions)
        self._episode_len = episode_len
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def observation_space(self, agent: str) -> gym.Space:
        return self._spaces_obs

    def action_space(self, agent: str) -> gym.Space:
        return self._spaces_act

    def _obs(self) -> Dict[str, np.ndarray]:
        return {
            a: self._rng.uniform(-1.0, 1.0, size=self._obs_dim).astype(np.float32)
            for a in self.possible_agents
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self.agents = list(self.possible_agents)
        return self._obs(), {a: {} for a in self.possible_agents}

    def step(self, actions: Dict[str, int]):
        self._t += 1
        # Team-shared reward: +1 for every agent that picked action 0.
        team = float(sum(1 for a in actions.values() if int(a) == 0))
        rewards = {a: team for a in self.possible_agents}
        trunc = self._t >= self._episode_len
        terminated = {a: False for a in self.possible_agents}
        truncated = {a: trunc for a in self.possible_agents}
        return self._obs(), rewards, terminated, truncated, {a: {} for a in self.possible_agents}


def make_dummy_team_env(**kwargs) -> DummyTeamEnv:
    """Factory for :class:`DummyTeamEnv` (the local/test env)."""
    return DummyTeamEnv(**kwargs)


# --------------------------------------------------------------------------- #
# Store helpers
# --------------------------------------------------------------------------- #
def _posix_join(*parts: str) -> str:
    return PurePosixPath(*[str(p) for p in parts]).as_posix()


def _agent_subdirs(store: ArtifactStore, ckpt_rel: str) -> List[str]:
    """Agent names inside a team checkpoint (dirs carrying a learner)."""
    names = []
    for name in store.backend.listdir(ckpt_rel):
        if store.backend.exists(_posix_join(ckpt_rel, name, "latest.json")):
            names.append(name)
    return names


def save_team_checkpoint(
    store: ArtifactStore,
    side,
    run_id: str,
    learners: Dict[str, PPOLearner],
    *,
    extra: Optional[Dict[str, Any]] = None,
    keep_last: Optional[int] = None,
):
    """Allocate the next ``checkpoint_NNNNNN`` and save every learner into it.

    Each agent is a ``nemesis.checkpoints`` run under
    ``<side>/<run_id>/checkpoint_NNNNNN/<agent>/``. Commits the Volume (if any)
    afterwards so the snapshot is durable the moment it is written.
    """
    ckpt = store.new_checkpoint(side, run_id)
    base = ckpt.local_dir()
    if base is None:
        raise NotImplementedError(
            "run_training needs a backend with a real local_path (LocalBackend "
            "or VolumeBackend); a fully-remote backend would need ArtifactStore."
            "put_dir (see docs/modal.md, NOTE FOR INTEGRATOR)."
        )
    for agent, learner in learners.items():
        learner.save(
            os.path.join(base, agent),
            extra={**(extra or {}), "agent": agent},
            keep_last=keep_last,
        )
    maybe_commit(store.backend)
    return ckpt


def load_team(store: ArtifactStore, side, run_id: str, *, checkpoint=None, device: str = "auto"):
    """Load every agent's :class:`PPOLearner` from a team checkpoint.

    Returns ``(learners_by_agent, checkpoint_handle)``. ``checkpoint`` defaults
    to the latest checkpoint for the run.
    """
    ckpt = checkpoint or store.latest_checkpoint(side, run_id)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint for {side}/{run_id} in the store")
    base = ckpt.local_dir()
    if base is None:
        raise NotImplementedError("load_team needs a backend with a real local_path")
    learners = {
        agent: PPOLearner.load(os.path.join(base, agent), device=device)
        for agent in _agent_subdirs(store, ckpt.rel_path)
    }
    if not learners:
        raise FileNotFoundError(f"checkpoint {ckpt.rel_path} holds no agent learners")
    return learners, ckpt


# --------------------------------------------------------------------------- #
# The plain, Modal-free training entrypoint.
# --------------------------------------------------------------------------- #
def run_training(
    *,
    store: ArtifactStore,
    run_id: str,
    timesteps: int,
    side=Side.BLUE,
    env: Optional[Any] = None,
    env_factory: Optional[Callable[[], Any]] = None,
    spec: Optional[Any] = None,
    n_steps: int = 2048,
    batch_size: int = 256,
    seed: Optional[int] = None,
    verbose: int = 0,
    save_every: Optional[int] = None,
    keep_last: Optional[int] = 3,
    fresh: bool = False,
    experiment: Optional[str] = None,
    time_budget_min: Optional[float] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Train one PPO learner per agent and checkpoint through ``store``.

    This is the single code path the Modal functions and the local ``--dry-run``
    both use. The environment is resolved as, in order: an explicit ``env``, an
    ``env_factory()``, or a CybORG env built from ``spec`` via
    ``nemesis.platform.envs.build`` (imported lazily, so no CybORG is needed for
    the dummy-env path).

    Resumes every agent from the run's latest checkpoint if one exists (unless
    ``fresh``). Saves every ``save_every`` timesteps (if set) and always once at
    the end. Returns a JSON-serialisable summary.
    """
    side = Side(side) if not isinstance(side, Side) else side
    run_dir = store.run_dir(side, run_id)
    if fresh:
        store.backend.remove_tree(run_dir)

    if env is None:
        if env_factory is not None:
            env = env_factory()
        else:
            if spec is None:
                raise ValueError("run_training needs one of env, env_factory, or spec")
            from nemesis.platform import envs as _envs  # lazy: imports CybORG

            env = _envs.build(spec)

    # Record the EnvSpec so the checkpoint can be re-evaluated on its scenario.
    if spec is not None and hasattr(spec, "as_dict"):
        store.write_json(_posix_join(run_dir, "config.json"), spec.as_dict())

    observations, _ = env.reset(seed=seed) if _accepts_seed(env.reset) else env.reset()

    # One learner per agent: resume from latest checkpoint, else fresh.
    learners: Dict[str, PPOLearner] = {}
    latest = None if fresh else store.latest_checkpoint(side, run_id)
    resume_dir = latest.local_dir() if latest is not None else None
    resume_agents = set(_agent_subdirs(store, latest.rel_path)) if latest is not None else set()
    for agent in env.possible_agents:
        if resume_dir is not None and agent in resume_agents:
            learners[agent] = PPOLearner.load(
                os.path.join(resume_dir, agent), total_timesteps=timesteps, verbose=verbose
            )
        else:
            learners[agent] = PPOLearner(
                env.observation_space(agent),
                env.action_space(agent),
                n_steps=n_steps,
                batch_size=batch_size,
                total_timesteps=timesteps,
                seed=seed,
                verbose=verbose,
            )

    start = min(l.num_timesteps for l in learners.values())
    extra = {
        "trained_with": "modal_app.train.run_training",
        "side": str(side),
        "run_id": run_id,
    }
    if spec is not None and hasattr(spec, "as_dict"):
        extra["env_spec"] = spec.as_dict()

    reward_fh = store.open_experiment_file(experiment, "reward_history.jsonl", "a") if experiment else None
    deadline = time.time() + time_budget_min * 60 if time_budget_min is not None else None

    episode_return, episode, updates, step = 0.0, 0, 0, 0
    last_checkpoint = None
    anchor = env.possible_agents[0]
    try:
        while start + step < timesteps and (deadline is None or time.time() < deadline):
            actions = {a: learners[a].act(observations[a]) for a in env.agents}
            observations, rewards, terminated, truncated, _ = env.step(actions)
            for a in actions:
                updates += int(learners[a].record(rewards[a], observations[a], terminated[a], truncated[a]))
            step += 1
            episode_return += float(rewards[anchor])  # blue reward is team-shared

            if all(terminated[a] or truncated[a] for a in actions):
                episode += 1
                if reward_fh is not None:
                    reward_fh.write(json.dumps(
                        {"episode": episode, "timesteps": start + step, "blue_return": episode_return}
                    ) + "\n")
                    reward_fh.flush()
                if progress:
                    print(f"episode {episode}: blue return {episode_return:.1f} ({start + step} ts)")
                episode_return = 0.0
                observations, _ = env.reset(seed=None) if _accepts_seed(env.reset) else env.reset()

            if save_every and step % save_every == 0:
                last_checkpoint = save_team_checkpoint(
                    store, side, run_id, learners, extra=extra, keep_last=keep_last
                )
    finally:
        if reward_fh is not None:
            reward_fh.close()

    last_checkpoint = save_team_checkpoint(store, side, run_id, learners, extra=extra, keep_last=keep_last)

    final_ts = min(l.num_timesteps for l in learners.values())
    return {
        "side": str(side),
        "run_id": run_id,
        "run_dir": run_dir,
        "agents": list(learners),
        "timesteps": final_ts,
        "updates": updates,
        "episodes": episode,
        "checkpoint": last_checkpoint.rel_path,
        "checkpoint_name": last_checkpoint.name,
    }


def run_evaluation(env, learners: Dict[str, PPOLearner], *, episodes: int = 1,
                   deterministic: bool = True) -> List[EpisodeMetrics]:
    """Run ``episodes`` deterministic episodes and return per-episode metrics.

    Each agent acts with ``PPOLearner.predict`` (no learning). Blue reward is
    team-shared, so the team return is any one agent's summed reward.
    """
    out: List[EpisodeMetrics] = []
    anchor = env.possible_agents[0]
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=None) if _accepts_seed(env.reset) else env.reset()
        team_return, steps, done = 0.0, 0, False
        while not done:
            actions = {a: learners[a].predict(obs[a], deterministic=deterministic) for a in env.agents}
            obs, rewards, terminated, truncated, _ = env.step(actions)
            team_return += float(rewards[anchor])
            steps += 1
            done = all(terminated[a] or truncated[a] for a in actions)
        out.append(EpisodeMetrics(
            episode=ep, steps=steps, blue_reward=team_return, blue_win=team_return > 0
        ))
    return out


def _accepts_seed(fn) -> bool:
    import inspect

    try:
        return "seed" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def default_run_id(prefix: str) -> str:
    """A unique, worker-safe run id (UTC timestamp) — see docs/modal.md."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"


# --------------------------------------------------------------------------- #
# Modal functions — defined only when modal is installed.
# --------------------------------------------------------------------------- #
def _train_on_volume(side, run_id, timesteps, *, steps, n_steps, batch_size,
                     save_every, seed, red_learns=False):
    """Body shared by train_blue/train_red: build a VolumeBackend + CybORG env."""
    from nemesis.platform.envs import blue_training_spec

    backend = VolumeBackend(ARTIFACTS_PATH, volume=volume)
    store = ArtifactStore(backend)
    run_id = run_id or default_run_id(str(side))
    spec = blue_training_spec(steps=steps, seed=seed)
    result = run_training(
        store=store, side=side, run_id=run_id, timesteps=timesteps, spec=spec,
        n_steps=n_steps, batch_size=batch_size, save_every=save_every, seed=seed,
        experiment=f"{side}/{run_id}", progress=True,
    )
    backend.commit()
    print("training complete:", json.dumps(result, indent=2))
    return result


if MODAL_AVAILABLE:  # pragma: no cover - needs modal installed + authed
    _FN_KW = dict(
        image=image,
        volumes={VOLUME_MOUNT_PATH: volume},
        cpu=8.0,                 # CybORG/PPO is CPU-bound; no GPU (see docs/modal.md)
        timeout=6 * 60 * 60,     # raise for longer runs (Modal max is 24 h)
    )

    @app.function(**_FN_KW)
    def train_blue(timesteps: int = 200_000, run_id: Optional[str] = None, steps: int = 500,
                   n_steps: int = 2048, batch_size: int = 256, save_every: int = 20_000,
                   seed: Optional[int] = None):
        """Train the blue CC4 team on Modal, persisting to the Volume."""
        return _train_on_volume(
            Side.BLUE, run_id, timesteps, steps=steps, n_steps=n_steps,
            batch_size=batch_size, save_every=save_every, seed=seed,
        )

    @app.function(**_FN_KW)
    def train_red(timesteps: int = 200_000, run_id: Optional[str] = None, steps: int = 500,
                  n_steps: int = 2048, batch_size: int = 256, save_every: int = 20_000,
                  seed: Optional[int] = None):
        """Train a red policy on Modal, persisting to the Volume.

        NOTE: the learning-red environment (``nemesis/red_env.py``) is owned by
        ``stage/red-training``. Until it lands this uses the shared blue_flat
        build as a structural placeholder; the integrator should point this at
        the red env factory. The storage/Volume path is identical.
        """
        return _train_on_volume(
            Side.RED, run_id, timesteps, steps=steps, n_steps=n_steps,
            batch_size=batch_size, save_every=save_every, seed=seed,
        )


__all__ = [
    "run_training",
    "run_evaluation",
    "save_team_checkpoint",
    "load_team",
    "make_dummy_team_env",
    "DummyTeamEnv",
    "default_run_id",
]
