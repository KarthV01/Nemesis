"""A small, dependency-light PPO agent implemented directly in JAX.

The agent follows Nemesis' step-at-a-time learner contract: call :meth:`act`,
step the shared multi-agent environment, then call :meth:`record`.  It supports
flat Box observations and Discrete actions, which are the spaces exposed by the
CC4 fixed-action wrappers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np


ArrayTree = Dict[str, jax.Array]


@dataclass(frozen=True)
class JAXPPOConfig:
    hidden_size: int = 128
    rollout_steps: int = 256
    epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 0


def _dense(key: jax.Array, inputs: int, outputs: int, scale: float = np.sqrt(2.0)) -> Tuple[jax.Array, jax.Array]:
    weight = jax.random.normal(key, (inputs, outputs)) * (scale / np.sqrt(inputs))
    return weight, jnp.zeros(outputs)


def _init_params(key: jax.Array, obs_dim: int, actions: int, hidden: int) -> ArrayTree:
    keys = jax.random.split(key, 4)
    w1, b1 = _dense(keys[0], obs_dim, hidden)
    w2, b2 = _dense(keys[1], hidden, hidden)
    wp, bp = _dense(keys[2], hidden, actions, 0.01)
    wv, bv = _dense(keys[3], hidden, 1, 1.0)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "wp": wp, "bp": bp, "wv": wv, "bv": bv}


@jax.jit
def _forward(params: ArrayTree, observations: jax.Array) -> Tuple[jax.Array, jax.Array]:
    hidden = jnp.tanh(observations @ params["w1"] + params["b1"])
    hidden = jnp.tanh(hidden @ params["w2"] + params["b2"])
    return hidden @ params["wp"] + params["bp"], (hidden @ params["wv"] + params["bv"]).squeeze(-1)


def _loss(params: ArrayTree, batch: Dict[str, jax.Array], config: JAXPPOConfig) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    logits, values = _forward(params, batch["observations"])
    log_probs = jax.nn.log_softmax(logits)
    selected = jnp.take_along_axis(log_probs, batch["actions"][:, None], axis=1).squeeze(1)
    ratio = jnp.exp(selected - batch["log_probs"])
    unclipped = ratio * batch["advantages"]
    clipped = jnp.clip(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * batch["advantages"]
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = jnp.mean(jnp.square(values - batch["returns"]))
    entropy = -jnp.mean(jnp.sum(jnp.exp(log_probs) * log_probs, axis=1))
    total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


class JAXPPOAgent:
    """Independent PPO policy for one CC4 agent slot."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Optional[JAXPPOConfig] = None):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("JAXPPOAgent requires a one-dimensional Box observation space")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("JAXPPOAgent requires a Discrete action space")
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or JAXPPOConfig()
        self._key = jax.random.PRNGKey(self.config.seed)
        self._key, init_key = jax.random.split(self._key)
        self.params = _init_params(init_key, int(observation_space.shape[0]), int(action_space.n), self.config.hidden_size)
        self._adam_m = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        self._adam_v = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        self._adam_step = 0
        self.num_timesteps = 0
        self.num_updates = 0
        self.last_metrics: Dict[str, float] = {}
        self._pending: Optional[Dict[str, Any]] = None
        self._rollout: List[Dict[str, Any]] = []

    def _observation(self, observation: np.ndarray) -> jax.Array:
        value = jnp.asarray(observation, dtype=jnp.float32)
        if value.shape != self.observation_space.shape:
            raise ValueError(f"expected observation shape {self.observation_space.shape}, got {value.shape}")
        return value

    def act(self, observation: np.ndarray, deterministic: bool = False) -> int:
        if self._pending is not None:
            raise RuntimeError("act() called twice without record()")
        obs = self._observation(observation)
        logits, value = _forward(self.params, obs[None])
        self._key, action_key = jax.random.split(self._key)
        action = jnp.argmax(logits[0]) if deterministic else jax.random.categorical(action_key, logits[0])
        log_prob = jax.nn.log_softmax(logits[0])[action]
        self._pending = {"observation": np.asarray(obs), "action": int(action), "value": float(value[0]), "log_prob": float(log_prob)}
        return int(action)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        obs = self._observation(observation)
        logits, _ = _forward(self.params, obs[None])
        if deterministic:
            return int(jnp.argmax(logits[0]))
        self._key, action_key = jax.random.split(self._key)
        return int(jax.random.categorical(action_key, logits[0]))

    def record(self, reward: float, next_observation: np.ndarray, terminated: bool, truncated: bool = False) -> bool:
        if self._pending is None:
            raise RuntimeError("record() called without a preceding act()")
        transition, self._pending = self._pending, None
        transition.update(reward=float(reward), done=bool(terminated or truncated))
        self._rollout.append(transition)
        self.num_timesteps += 1
        if len(self._rollout) < self.config.rollout_steps:
            return False
        bootstrap = 0.0
        if not terminated:
            _, value = _forward(self.params, self._observation(next_observation)[None])
            bootstrap = float(value[0])
        self._update(bootstrap)
        return True

    def _update(self, bootstrap: float) -> None:
        rewards = np.asarray([step["reward"] for step in self._rollout], dtype=np.float32)
        values = np.asarray([step["value"] for step in self._rollout] + [bootstrap], dtype=np.float32)
        dones = np.asarray([step["done"] for step in self._rollout], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            continuation = 1.0 - dones[index]
            delta = rewards[index] + self.config.gamma * values[index + 1] * continuation - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * continuation * gae
            advantages[index] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        data = {
            "observations": np.asarray([step["observation"] for step in self._rollout], dtype=np.float32),
            "actions": np.asarray([step["action"] for step in self._rollout], dtype=np.int32),
            "log_probs": np.asarray([step["log_prob"] for step in self._rollout], dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }
        size = len(rewards)
        batch_size = min(self.config.minibatch_size, size)
        metrics: Dict[str, jax.Array] = {}
        for _ in range(self.config.epochs):
            self._key, permutation_key = jax.random.split(self._key)
            order = np.asarray(jax.random.permutation(permutation_key, size))
            for start in range(0, size, batch_size):
                indices = order[start : start + batch_size]
                batch = {name: jnp.asarray(value[indices]) for name, value in data.items()}
                (loss, metrics), gradients = jax.value_and_grad(_loss, has_aux=True)(self.params, batch, self.config)
                self._apply_gradients(gradients)
                metrics = {**metrics, "loss": loss}
        self.last_metrics = {name: float(value) for name, value in metrics.items()}
        self.num_updates += 1
        self._rollout.clear()

    def _apply_gradients(self, gradients: ArrayTree) -> None:
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(gradients)))
        scale = jnp.minimum(1.0, self.config.max_grad_norm / (norm + 1e-6))
        gradients = jax.tree_util.tree_map(lambda value: value * scale, gradients)
        self._adam_step += 1
        self._adam_m = jax.tree_util.tree_map(lambda old, grad: 0.9 * old + 0.1 * grad, self._adam_m, gradients)
        self._adam_v = jax.tree_util.tree_map(lambda old, grad: 0.999 * old + 0.001 * jnp.square(grad), self._adam_v, gradients)
        correction_m = 1.0 - 0.9**self._adam_step
        correction_v = 1.0 - 0.999**self._adam_step
        self.params = jax.tree_util.tree_map(
            lambda param, mean, variance: param - self.config.learning_rate * (mean / correction_m) / (jnp.sqrt(variance / correction_v) + 1e-8),
            self.params, self._adam_m, self._adam_v,
        )

    def save(self, path: Any) -> Path:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez(destination / "jax_ppo.npz", **{name: np.asarray(value) for name, value in self.params.items()})
        (destination / "metadata.json").write_text(json.dumps({
            "format": "nemesis-jax-ppo-v1", "config": asdict(self.config),
            "observation_shape": list(self.observation_space.shape), "actions": int(self.action_space.n),
            "num_timesteps": self.num_timesteps, "num_updates": self.num_updates,
        }, indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: Any) -> "JAXPPOAgent":
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "nemesis-jax-ppo-v1":
            raise ValueError(f"unsupported JAX agent format: {metadata.get('format')!r}")
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=tuple(metadata["observation_shape"]), dtype=np.float32)
        agent = cls(observation_space, gym.spaces.Discrete(metadata["actions"]), JAXPPOConfig(**metadata["config"]))
        with np.load(source / "jax_ppo.npz") as archive:
            agent.params = {name: jnp.asarray(archive[name]) for name in archive.files}
        agent.num_timesteps = int(metadata["num_timesteps"])
        agent.num_updates = int(metadata["num_updates"])
        return agent

