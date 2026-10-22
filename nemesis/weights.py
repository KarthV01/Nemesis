"""Plain PyTorch weight files for every trained agent, small enough to keep in git.

``nemesis.checkpoints`` keeps *everything* needed to resume training (optimizer
state, replay buffers, ...) under the git-ignored ``checkpoints/``. This module
keeps just what a trained agent *is* — its policy network's weights — as one
small ``.pt`` file per agent, meant to be committed::

    weights/
        sb3_ppo/
            blue_agent_0/
                policy.pt     # torch.save({"state_dict": ..., "obs_rms": ...})
                meta.json     # timesteps, reward, spaces, sha256 of policy.pt
        rllib_ppo/
            blue_agent_0/ ...

Each export overwrites the agent's previous file, so the repo holds one copy
per agent and git history holds the progression.

``policy.pt`` contains only tensors and plain Python values, so it loads with
``torch.load(path, weights_only=True)`` — no pickled classes, safe to load
from a pulled branch.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

PathLike = Union[str, os.PathLike]

WEIGHTS_ROOT = Path("weights")
POLICY_FILE = "policy.pt"
META_FILE = "meta.json"
FORMAT_VERSION = 1


def agent_weights_dir(trainer: str, agent: str, root: PathLike = WEIGHTS_ROOT) -> Path:
    """``weights/<trainer>/<agent>``, e.g. ``weights/sb3_ppo/blue_agent_0``."""
    return Path(root) / trainer / agent


def save_policy_weights(
    state_dict: Mapping[str, Any],
    out_dir: PathLike,
    *,
    meta: Optional[Dict[str, Any]] = None,
    obs_rms: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``state_dict`` to ``out_dir/policy.pt`` with a ``meta.json`` beside it.

    Parameters
    ----------
    state_dict : mapping
        Parameter name -> tensor (or numpy array, as RLlib returns them).
        Everything is moved to CPU so the file loads on any machine.
    out_dir : path
        The agent's weights directory; created if missing.
    meta : dict, optional
        JSON-serialisable description stored in both files.
    obs_rms : dict, optional
        Observation-normalisation statistics the policy was trained under
        (``mean``, ``var``, ``clip_obs``, ``epsilon``); without them a policy
        trained on normalised inputs would act on wrongly scaled observations.

    Returns
    -------
    Path
        The ``policy.pt`` written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": FORMAT_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # str(): TorchVersion is a str subclass weights_only loading rejects.
        "torch_version": str(torch.__version__),
        **(meta or {}),
    }
    payload = {
        "format": FORMAT_VERSION,
        "state_dict": {k: _to_cpu_tensor(v) for k, v in state_dict.items()},
        "obs_rms": {k: _to_cpu_tensor(v) for k, v in obs_rms.items()} if obs_rms else None,
        "meta": meta,
    }

    # Write-then-rename so a crash mid-save never leaves a truncated policy.pt
    # for the git sync to commit.
    policy_path = out_dir / POLICY_FILE
    tmp = out_dir / f".{POLICY_FILE}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, policy_path)

    meta_on_disk = {
        **meta,
        "parameters": int(sum(t.numel() for t in payload["state_dict"].values())),
        "has_obs_rms": obs_rms is not None,
        "sha256": _sha256(policy_path),
    }
    tmp = out_dir / f".{META_FILE}.tmp"
    tmp.write_text(json.dumps(meta_on_disk, indent=2, sort_keys=True))
    os.replace(tmp, out_dir / META_FILE)
    return policy_path


def load_policy_weights(path: PathLike, *, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a ``policy.pt`` (or the one inside an agent's weights directory).

    Returns the payload dict: ``state_dict``, ``obs_rms`` (or None), ``meta``.
    """
    path = Path(path)
    if path.is_dir():
        path = path / POLICY_FILE
    return torch.load(path, map_location=map_location, weights_only=True)


# ---------------------------------------------------------------------- #
# Stable-Baselines3
# ---------------------------------------------------------------------- #
def export_sb3_weights(
    model,
    agent: str,
    *,
    trainer: Optional[str] = None,
    root: PathLike = WEIGHTS_ROOT,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export an SB3 model's policy network to ``weights/<trainer>/<agent>``.

    ``trainer`` defaults to ``sb3_<algorithm>``, e.g. ``sb3_ppo``.
    """
    trainer = trainer or f"sb3_{type(model).__name__.lower()}"
    meta = {
        "agent": agent,
        "trainer": trainer,
        "algorithm": f"{type(model).__module__}.{type(model).__qualname__}",
        "policy_class": f"{type(model.policy).__module__}.{type(model.policy).__qualname__}",
        "num_timesteps": int(model.num_timesteps),
        "observation_space": _space_repr(model.observation_space),
        "action_space": _space_repr(model.action_space),
        "mean_episode_reward": _sb3_mean_reward(model),
        "extra": extra or {},
    }

    obs_rms = None
    vec_normalize = model.get_vec_normalize_env()
    if vec_normalize is not None and vec_normalize.norm_obs:
        obs_rms = {
            "mean": vec_normalize.obs_rms.mean,
            "var": vec_normalize.obs_rms.var,
            "clip_obs": float(vec_normalize.clip_obs),
            "epsilon": float(vec_normalize.epsilon),
        }

    return save_policy_weights(
        model.policy.state_dict(),
        agent_weights_dir(trainer, agent, root),
        meta=meta,
        obs_rms=obs_rms,
    )


def load_sb3_weights(model, path: PathLike, *, strict: bool = True) -> Dict[str, Any]:
    """Load exported weights into an existing SB3 model's policy (same architecture).

    Returns the loaded payload so callers can inspect ``meta`` / ``obs_rms``.
    """
    payload = load_policy_weights(path, map_location=str(model.device))
    model.policy.load_state_dict(payload["state_dict"], strict=strict)
    return payload


# ---------------------------------------------------------------------- #
# RLlib
# ---------------------------------------------------------------------- #
def export_rllib_weights(
    algo,
    policy_map: Mapping[str, str],
    *,
    trainer: str = "rllib_ppo",
    root: PathLike = WEIGHTS_ROOT,
    iteration: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> list:
    """Export each RLlib policy's weights, one directory per agent.

    ``policy_map`` maps agent name -> policy id (e.g. ``"blue_agent_0" -> "Agent0"``).
    Returns the list of ``policy.pt`` paths written.
    """
    written = []
    for agent, policy_id in policy_map.items():
        policy = algo.get_policy(policy_id)
        meta = {
            "agent": agent,
            "trainer": trainer,
            "policy_id": policy_id,
            "model_class": f"{type(policy.model).__module__}.{type(policy.model).__qualname__}",
            "iteration": iteration,
            "observation_space": _space_repr(policy.observation_space),
            "action_space": _space_repr(policy.action_space),
            "extra": extra or {},
        }
        written.append(
            save_policy_weights(
                policy.get_weights(), agent_weights_dir(trainer, agent, root), meta=meta
            )
        )
    return written


def _space_repr(space) -> str:
    # numpy wraps long arrays across lines; keep meta.json one value per line.
    return " ".join(repr(space).split())


def _sb3_mean_reward(model) -> Optional[float]:
    buffer = model.ep_info_buffer
    if not buffer:
        return None
    return float(np.mean([info["r"] for info in buffer]))


def _to_cpu_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value)).clone()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
