"""Save and restore everything a Stable-Baselines3 model has learned.

``model.save()`` on its own keeps the policy weights, optimizer state and
hyperparameters, but quietly drops two things training needs to pick up where
it left off: an off-policy algorithm's replay buffer (DQN, SAC, ...) and the
running observation/reward statistics of a ``VecNormalize`` wrapper. Lose the
latter and a restored policy sees differently-scaled inputs than it was trained
on. This module bundles all of it into one checkpoint directory::

    checkpoints/sb3_blue/
        step_000020000/
            model.zip           # weights + optimizer + hyperparameters
            replay_buffer.pkl   # off-policy algorithms only
            vec_normalize.pkl   # only when training under VecNormalize
            meta.json           # algorithm, timesteps, versions, caller extras
        step_000040000/
        latest.json             # {"checkpoint": "step_000040000"}

Typical use::

    model = PPO("MlpPolicy", env)
    model.learn(100_000, callback=LearnedCheckpointCallback("checkpoints/sb3_blue", save_freq=20_000))

    # later, possibly in a new process
    model = load_learned("checkpoints/sb3_blue", env=env)   # resolves latest.json
    model.learn(100_000, reset_num_timesteps=False)          # keeps counting from where it stopped
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

PathLike = Union[str, os.PathLike]

MODEL_FILE = "model.zip"
REPLAY_BUFFER_FILE = "replay_buffer.pkl"
VEC_NORMALIZE_FILE = "vec_normalize.pkl"
META_FILE = "meta.json"
LATEST_FILE = "latest.json"
CHECKPOINT_PREFIX = "step_"


def save_learned(
    model: BaseAlgorithm,
    root: PathLike,
    *,
    extra: Optional[Dict[str, Any]] = None,
    keep_last: Optional[int] = None,
    include_replay_buffer: bool = True,
) -> Path:
    """Write a full checkpoint of ``model`` under ``root`` and mark it latest.

    The checkpoint is named after ``model.num_timesteps`` and written to a
    temporary directory first, so an interrupted save never leaves a
    half-written checkpoint behind for ``load_learned`` to trip over.

    Parameters
    ----------
    model : BaseAlgorithm
        Any SB3 algorithm (PPO, A2C, DQN, SAC, ...).
    root : path
        Directory holding this run's checkpoints; created if missing.
    extra : dict, optional
        JSON-serialisable values stored in ``meta.json`` (agent name, scenario
        settings, ...), so a checkpoint documents what it was trained against.
    keep_last : int, optional
        After saving, delete all but the newest ``keep_last`` checkpoints.
    include_replay_buffer : bool
        Off-policy buffers can run to gigabytes; set False to skip them.

    Returns
    -------
    Path
        The checkpoint directory that was written.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"{CHECKPOINT_PREFIX}{model.num_timesteps:09d}"
    final = root / name
    staging = root / f".{name}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()

    model.save(staging / MODEL_FILE)

    has_buffer = (
        include_replay_buffer
        and isinstance(model, OffPolicyAlgorithm)
        and model.replay_buffer is not None
    )
    if has_buffer:
        model.save_replay_buffer(staging / REPLAY_BUFFER_FILE)

    vec_normalize = model.get_vec_normalize_env()
    if vec_normalize is not None:
        vec_normalize.save(str(staging / VEC_NORMALIZE_FILE))

    meta = {
        "algorithm": f"{type(model).__module__}.{type(model).__qualname__}",
        "num_timesteps": int(model.num_timesteps),
        # Mean return over SB3's rolling window of recent episodes (up to 100).
        "recent_episode_reward_mean": (
            float(sum(ep["r"] for ep in model.ep_info_buffer) / len(model.ep_info_buffer))
            if model.ep_info_buffer
            else None
        ),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sb3_version": stable_baselines3.__version__,
        "has_replay_buffer": has_buffer,
        "has_vec_normalize": vec_normalize is not None,
        "extra": extra or {},
    }
    (staging / META_FILE).write_text(json.dumps(meta, indent=2))

    # Saving twice at the same timestep (e.g. a periodic save landing on the
    # final step) simply replaces the older copy.
    shutil.rmtree(final, ignore_errors=True)
    os.replace(staging, final)
    _write_latest(root, name)

    if keep_last is not None:
        prune_checkpoints(root, keep_last)
    return final


def load_learned(
    path: PathLike,
    env: Optional[Any] = None,
    *,
    device: str = "auto",
    training: bool = True,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> BaseAlgorithm:
    """Restore a model saved by ``save_learned``, ready to keep training or act.

    Parameters
    ----------
    path : path
        Either a single checkpoint directory or a run's root directory, in
        which case the checkpoint named by ``latest.json`` is loaded.
    env : gym.Env or VecEnv, optional
        Environment to attach. When the checkpoint carries VecNormalize stats,
        ``env`` is wrapped in a restored ``VecNormalize`` so the policy sees
        observations scaled exactly as during training. Omit it for
        inference-only use on raw observations.
    device : str
        Torch device to load onto.
    training : bool
        Whether a restored VecNormalize keeps updating its statistics. Pass
        False when evaluating so evaluation doesn't drift the stats.
    custom_objects : dict, optional
        Forwarded to ``BaseAlgorithm.load`` to override saved attributes.
    """
    checkpoint = resolve_checkpoint(path)
    meta = read_meta(checkpoint)
    algorithm_cls = _import_algorithm(meta["algorithm"])

    if env is not None and meta["has_vec_normalize"]:
        venv = env if isinstance(env, VecEnv) else DummyVecEnv([lambda: env])
        env = VecNormalize.load(str(checkpoint / VEC_NORMALIZE_FILE), venv)
        env.training = training

    model = algorithm_cls.load(
        checkpoint / MODEL_FILE, env=env, device=device, custom_objects=custom_objects
    )
    if meta["has_replay_buffer"]:
        model.load_replay_buffer(checkpoint / REPLAY_BUFFER_FILE)
    return model


def resolve_checkpoint(path: PathLike) -> Path:
    """Turn a run root or checkpoint directory into a concrete checkpoint path."""
    path = Path(path)
    if (path / META_FILE).is_file():
        return path
    latest = path / LATEST_FILE
    if latest.is_file():
        return path / json.loads(latest.read_text())["checkpoint"]
    # No pointer file (e.g. it was deleted by hand): fall back to the newest.
    checkpoints = list_checkpoints(path)
    if not checkpoints:
        raise FileNotFoundError(f"No SB3 checkpoint found in {path}")
    return checkpoints[-1]


def read_meta(path: PathLike) -> Dict[str, Any]:
    """Return the ``meta.json`` of a checkpoint (or of a run's latest one)."""
    return json.loads((resolve_checkpoint(path) / META_FILE).read_text())


def list_checkpoints(root: PathLike) -> list:
    """Complete checkpoints under ``root``, oldest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(CHECKPOINT_PREFIX) and (p / META_FILE).is_file()
    )


def prune_checkpoints(root: PathLike, keep_last: int) -> None:
    """Delete all but the newest ``keep_last`` checkpoints (never the latest)."""
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    latest = resolve_checkpoint(root)
    for old in list_checkpoints(root)[:-keep_last]:
        if old != latest:
            shutil.rmtree(old)


class LearnedCheckpointCallback(BaseCallback):
    """Periodically ``save_learned`` during ``model.learn`` and once at the end.

    Parameters
    ----------
    root : path
        Run directory passed to ``save_learned``.
    save_freq : int
        Save every this many environment timesteps (summed across vectorised
        envs, i.e. measured in ``model.num_timesteps``).
    keep_last : int, optional
        Rolling window of checkpoints to keep; ``None`` keeps them all.
    extra : dict, optional
        Stored in every checkpoint's ``meta.json``.
    include_replay_buffer : bool
        See ``save_learned``.
    on_save : callable, optional
        Called as ``on_save(model, checkpoint_path)`` after every save, e.g. to
        export weights or push them to git.
    """

    def __init__(
        self,
        root: PathLike,
        save_freq: int,
        *,
        keep_last: Optional[int] = 3,
        extra: Optional[Dict[str, Any]] = None,
        include_replay_buffer: bool = True,
        on_save: Optional[Callable[[BaseAlgorithm, Path], None]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if save_freq < 1:
            raise ValueError("save_freq must be at least 1")
        self.root = Path(root)
        self.save_freq = save_freq
        self.keep_last = keep_last
        self.extra = extra
        self.include_replay_buffer = include_replay_buffer
        self.on_save = on_save
        self._last_saved = 0

    def _on_training_start(self) -> None:
        # On a resumed run num_timesteps starts above zero; count from there.
        self._last_saved = self.model.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved >= self.save_freq:
            self._save()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last_saved:
            self._save()

    def _save(self) -> None:
        path = save_learned(
            self.model,
            self.root,
            extra=self.extra,
            keep_last=self.keep_last,
            include_replay_buffer=self.include_replay_buffer,
        )
        self._last_saved = self.num_timesteps
        if self.verbose:
            print(f"Saved SB3 checkpoint -> {path}")
        if self.on_save is not None:
            self.on_save(self.model, path)


def _write_latest(root: Path, name: str) -> None:
    tmp = root / f".{LATEST_FILE}.tmp"
    tmp.write_text(json.dumps({"checkpoint": name}))
    os.replace(tmp, root / LATEST_FILE)


def _import_algorithm(qualified_name: str) -> type:
    module_name, _, cls_name = qualified_name.rpartition(".")
    return getattr(importlib.import_module(module_name), cls_name)
