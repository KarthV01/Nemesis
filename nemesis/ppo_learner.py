"""PPO learner that updates a policy one environment step at a time.

SB3's ``PPO.learn`` insists on driving the environment itself: it calls
``env.step`` and owns the loop. Nemesis needs the opposite in self-play — the
multi-agent loop steps CybORG for every agent at once, and each learner is just
told what its agent saw, did and earned. ``PPOLearner`` turns PPO inside out:

    learner = PPOLearner(observation_space, action_space)
    obs = ...
    while training:
        action = learner.act(obs)
        obs, reward, terminated, truncated = <step the shared env>
        learner.record(reward, obs, terminated, truncated)   # updates weights when a rollout fills

Under the hood the policy, rollout buffer and clipped-surrogate update are
SB3's own (``learner.model`` is a real ``stable_baselines3.PPO``), so the
weights save and restore with ``nemesis.checkpoints`` exactly like a model
trained by ``PPO.learn``, and a checkpoint from either path loads into the other.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

from nemesis.checkpoints import PathLike, load_learned, save_learned


class _SpacesOnlyEnv(gym.Env):
    """Placeholder that only carries spaces: SB3 builds a model from an env,
    but the learner never lets PPO step one."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("PPOLearner is fed experience via act/record; it never steps an env")

    def step(self, action):
        raise RuntimeError("PPOLearner is fed experience via act/record; it never steps an env")


class PPOLearner:
    """Collect experience step by step and run a PPO update whenever a rollout fills.

    Parameters
    ----------
    observation_space, action_space : gym.Space
        Spaces of the single agent this learner controls.
    n_steps : int
        Transitions collected per rollout; one PPO update runs per full rollout.
    total_timesteps : int, optional
        Planned training length. Only used to anneal schedule-valued
        hyperparameters (e.g. a callable ``learning_rate``); leave unset for
        constant ones.
    verbose : int
        1 prints SB3's training stats after every update.
    **ppo_kwargs
        Any other ``stable_baselines3.PPO`` argument (``batch_size``,
        ``n_epochs``, ``gamma``, ``clip_range``, ``policy_kwargs``, ``seed``, ...).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        n_steps: int = 2048,
        total_timesteps: Optional[int] = None,
        verbose: int = 0,
        policy: str = "MlpPolicy",
        **ppo_kwargs: Any,
    ):
        env = DummyVecEnv([lambda: _SpacesOnlyEnv(observation_space, action_space)])
        model = PPO(policy, env, n_steps=n_steps, verbose=verbose, **ppo_kwargs)
        self._attach(model, total_timesteps, verbose)

    @classmethod
    def from_model(
        cls, model: PPO, *, total_timesteps: Optional[int] = None, verbose: int = 0
    ) -> "PPOLearner":
        """Wrap an existing SB3 PPO model (e.g. one restored by ``load_learned``)."""
        if not isinstance(model, PPO):
            raise TypeError(f"PPOLearner needs a stable_baselines3.PPO model, got {type(model).__name__}")
        if model.n_envs != 1:
            raise ValueError(f"PPOLearner feeds a single experience stream; model has n_envs={model.n_envs}")
        learner = cls.__new__(cls)
        learner._attach(model, total_timesteps, verbose)
        return learner

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        device: str = "auto",
        total_timesteps: Optional[int] = None,
        verbose: int = 0,
    ) -> "PPOLearner":
        """Resume from a ``nemesis.checkpoints`` directory (a run root or one checkpoint)."""
        return cls.from_model(load_learned(path, device=device), total_timesteps=total_timesteps, verbose=verbose)

    def _attach(self, model: PPO, total_timesteps: Optional[int], verbose: int) -> None:
        self.model = model
        self.total_timesteps = total_timesteps
        # PPO.learn normally sets these up in _setup_learn; we never call learn.
        model.set_logger(configure(None, ["stdout"] if verbose else []))
        # Create both buffers together: a checkpoint resumed under PPO.learn
        # skips creating them if ep_info_buffer already exists.
        if model.ep_info_buffer is None:
            model.ep_info_buffer = deque(maxlen=model._stats_window_size)
            model.ep_success_buffer = deque(maxlen=model._stats_window_size)
        model.policy.set_training_mode(False)
        model.rollout_buffer.reset()

        self.num_updates = 0
        self._pending: Optional[Dict[str, Any]] = None
        self._episode_start = True
        self._episode_reward = 0.0
        self._episode_length = 0

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    @property
    def num_timesteps(self) -> int:
        return self.model.num_timesteps

    def act(self, observation: np.ndarray) -> Any:
        """Sample an action for ``observation`` and remember it for ``record``.

        Every call must be followed by exactly one ``record`` before the next
        ``act``, since the update needs the value and log-probability of the
        action that was actually taken.
        """
        if self._pending is not None:
            raise RuntimeError("act() called twice without record(); report the last step's outcome first")
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with th.no_grad():
            action, value, log_prob = self.model.policy(obs_tensor)
        action = action.cpu().numpy()
        self._pending = {
            "obs": np.asarray(observation)[None],
            "action": action,
            "value": value,
            "log_prob": log_prob,
        }
        return action[0].item() if isinstance(self.model.action_space, gym.spaces.Discrete) else action[0]

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        """Act without recording anything — for evaluation or a frozen opponent."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action.item() if isinstance(self.model.action_space, gym.spaces.Discrete) else action

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def record(
        self,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> bool:
        """Store the outcome of the last ``act`` and update the policy if a rollout is full.

        ``next_observation`` is the observation after the step. On a
        truncation (time limit, which is how every CC4 episode ends) it is used
        to bootstrap the value of the cut-off future, as ``PPO.learn`` does.
        After an episode ends, start the next one with the env's reset
        observation as usual.

        Returns True when this step triggered a weight update.
        """
        if self._pending is None:
            raise RuntimeError("record() called without a preceding act()")
        pending, self._pending = self._pending, None
        done = bool(terminated or truncated)

        self._episode_reward += float(reward)
        self._episode_length += 1
        bootstrapped = float(reward)
        if truncated and not terminated:
            bootstrapped += self.model.gamma * self._value(next_observation).item()

        action = pending["action"]
        if isinstance(self.model.action_space, gym.spaces.Discrete):
            action = action.reshape(-1, 1)
        self.model.rollout_buffer.add(
            pending["obs"],
            action,
            np.array([bootstrapped], dtype=np.float32),
            np.array([self._episode_start]),
            pending["value"],
            pending["log_prob"],
        )
        self.model.num_timesteps += 1
        self._episode_start = done

        if done:
            self.model.ep_info_buffer.append({"r": self._episode_reward, "l": self._episode_length})
            self._episode_reward = 0.0
            self._episode_length = 0

        if not self.model.rollout_buffer.full:
            return False
        self._update(next_observation, done)
        return True

    def _value(self, observation: np.ndarray) -> th.Tensor:
        with th.no_grad():
            return self.model.policy.predict_values(obs_as_tensor(np.asarray(observation)[None], self.model.device))

    def _update(self, last_observation: np.ndarray, done: bool) -> None:
        """One PPO update on the full rollout, then start collecting the next."""
        model = self.model
        model.rollout_buffer.compute_returns_and_advantage(
            last_values=self._value(last_observation), dones=np.array([done])
        )
        if self.total_timesteps:
            model._update_current_progress_remaining(model.num_timesteps, self.total_timesteps)
        if model.ep_info_buffer:
            model.logger.record("rollout/ep_rew_mean", float(np.mean([e["r"] for e in model.ep_info_buffer])))
            model.logger.record("rollout/ep_len_mean", float(np.mean([e["l"] for e in model.ep_info_buffer])))
        model.logger.record("time/total_timesteps", model.num_timesteps)
        model.train()  # SB3's clipped-surrogate update
        model.logger.dump(step=model.num_timesteps)
        model.policy.set_training_mode(False)
        model.rollout_buffer.reset()
        self.num_updates += 1

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, root: PathLike, *, extra: Optional[Dict[str, Any]] = None, keep_last: Optional[int] = None):
        """Checkpoint the weights and optimizer with ``nemesis.checkpoints.save_learned``.

        Experience collected since the last update is not saved; a resumed
        learner starts a fresh rollout.
        """
        return save_learned(self.model, root, extra=extra, keep_last=keep_last)
