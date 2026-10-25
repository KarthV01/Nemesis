"""Blue policy adapters: one interface over the three things we evaluate.

The evaluation harness scores *a blue policy*, but the three kinds of blue policy
we care about speak different interfaces:

* a **random** baseline and a **trained SB3** policy act on the flat, vectorised
  view of the environment (``BlueFlatWrapper``): a ``Discrete`` action index per
  blue agent;
* a **scripted Nemesis** heuristic (e.g. :class:`~nemesis.agents.BlueCaptain`)
  acts on CybORG's raw dict observation and dict action space, returning an
  instantiated action object.

:class:`BluePolicy` hides that difference behind ``reset`` + ``act``. Each policy
declares the ``wrapper`` (``EnvSpec.wrapper``) it needs; the harness builds a
matching environment and drives it uniformly. Metric derivation never looks at
the policy or the wrapper — it reads the unwrapped simulator — so the three are
directly comparable.

Everything here runs inside CybORG's simulated CC4 network.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BluePolicy(Protocol):
    """A blue policy the harness can evaluate.

    Attributes
    ----------
    name : str
        Short label recorded in the metrics (``extra["blue_policy"]``).
    wrapper : str
        The ``EnvSpec.wrapper`` this policy requires (``"blue_flat"`` or
        ``"none"``). The harness forces the environment to this wrapper.
    """

    name: str
    wrapper: str

    def reset(self, env) -> None:
        """Start a fresh episode against ``env`` (already ``reset``)."""

    def act(self, env, observations: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return ``{blue_agent: action}`` for this step.

        ``observations`` is the dict the wrapped env's ``step``/``reset``
        returned (flat vectors), or ``None`` for the raw-CybORG driver, in which
        case the policy reads what it needs from ``env`` directly.
        """


def _blue_agents(env) -> List[str]:
    """Blue agent names, read from the unwrapped controller (wrapper-agnostic)."""
    return list(env.unwrapped.environment_controller.team["Blue"])


# --------------------------------------------------------------------------- #
# Random baseline (flat wrapper)
# --------------------------------------------------------------------------- #
class RandomBluePolicy:
    """Every blue agent samples its own ``Discrete`` action space. The floor."""

    wrapper = "blue_flat"

    def __init__(self, seed: Optional[int] = None, name: str = "random"):
        self.name = name
        self._rng = np.random.default_rng(seed)
        self._agents: List[str] = []

    def reset(self, env) -> None:
        self._agents = list(env.possible_agents)

    def act(self, env, observations) -> Dict[str, int]:
        return {
            a: int(self._rng.integers(env.action_space(a).n)) for a in self._agents
        }


# --------------------------------------------------------------------------- #
# Scripted Nemesis heuristic (raw CybORG)
# --------------------------------------------------------------------------- #
class ScriptedBluePolicy:
    """Drive the raw CybORG dict interface with a Nemesis heuristic per blue slot.

    ``factory(name, np_random)`` builds one agent for each blue agent name (so a
    full team of, e.g., :class:`BlueCaptain` defends its own zone). The agent
    reads ``env.get_observation(agent)`` and ``env.get_action_space(agent)`` and
    returns an instantiated CybORG action.
    """

    wrapper = "none"

    def __init__(
        self,
        factory: Callable[..., Any],
        *,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self._factory = factory
        self._seed = seed
        self.name = name or getattr(factory, "__name__", "scripted")
        self._agents: Dict[str, Any] = {}

    def reset(self, env) -> None:
        rng = np.random.default_rng(self._seed)
        self._agents = {}
        for agent_name in _blue_agents(env):
            try:
                self._agents[agent_name] = self._factory(agent_name, np_random=rng)
            except TypeError:
                # Factory without an np_random kwarg.
                self._agents[agent_name] = self._factory(agent_name)

    def act(self, env, observations) -> Dict[str, Any]:
        actions = {}
        for agent_name, agent in self._agents.items():
            obs = env.get_observation(agent_name)
            action_space = env.get_action_space(agent_name)
            actions[agent_name] = agent.get_action(obs, action_space)
        return actions


# --------------------------------------------------------------------------- #
# Trained SB3 checkpoint (flat wrapper)
# --------------------------------------------------------------------------- #
class TrainedBluePolicy:
    """A loaded SB3 policy acting greedily; the other blue agents ``Sleep``.

    One trained agent (``agent``, default ``blue_agent_0``) is controlled by the
    model; the rest idle so the policy is scored in isolation, mirroring how it
    was trained (``nemesis.blue_env.SingleBlueEnv``). If the checkpoint carries
    ``VecNormalize`` observation statistics they are applied before predicting,
    so the policy sees inputs scaled as during training.
    """

    wrapper = "blue_flat"

    def __init__(
        self,
        checkpoint: str,
        *,
        agent: str = "blue_agent_0",
        device: str = "cpu",
        name: Optional[str] = None,
    ):
        from nemesis.checkpoints import load_learned, read_meta, resolve_checkpoint

        self.agent = agent
        self.name = name or f"trained:{agent}"
        self._model = load_learned(checkpoint, env=None, device=device, training=False)
        self._normalize = self._load_obs_normalizer(resolve_checkpoint(checkpoint), read_meta(checkpoint))
        self._sleep_idx: Dict[str, int] = {}
        self._agents: List[str] = []

    @staticmethod
    def _load_obs_normalizer(checkpoint_dir, meta):
        """Return a function obs->normalized obs from the checkpoint, or identity."""
        if not meta.get("has_vec_normalize"):
            return lambda obs: obs
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from nemesis.checkpoints import VEC_NORMALIZE_FILE

            # VecNormalize.load needs a venv only to carry the spaces; it is
            # never stepped. normalize_obs() uses the restored obs_rms + clip.
            venv = DummyVecEnv([lambda: _SpaceCarrier()])
            vec = VecNormalize.load(str(checkpoint_dir / VEC_NORMALIZE_FILE), venv)
            vec.training = False
            if not vec.norm_obs:
                return lambda obs: obs
            return lambda obs: vec.normalize_obs(np.asarray(obs))
        except Exception:
            # Normalisation is a refinement; fall back to raw observations.
            return lambda obs: obs

    def reset(self, env) -> None:
        self._agents = list(env.possible_agents)
        self._sleep_idx = {}
        for a in self._agents:
            labels = env.action_labels(a)
            self._sleep_idx[a] = labels.index("Sleep") if "Sleep" in labels else 0

    def act(self, env, observations) -> Dict[str, int]:
        actions = {a: self._sleep_idx[a] for a in self._agents}
        if self.agent in self._agents and observations is not None:
            obs = self._normalize(np.asarray(observations[self.agent]))
            action, _ = self._model.predict(obs, deterministic=True)
            actions[self.agent] = int(action)
        return actions


class _SpaceCarrier:
    """A minimal gym.Env that only carries spaces, for VecNormalize.load."""

    def __init__(self):
        import gymnasium as gym

        # Shapes are irrelevant to normalize_obs (it uses the saved obs_rms);
        # a 1-D Box is enough to construct the wrapper.
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *args, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


# --------------------------------------------------------------------------- #
# Factories (what callers use)
# --------------------------------------------------------------------------- #
def random_blue(seed: Optional[int] = None) -> RandomBluePolicy:
    """Random blue baseline."""
    return RandomBluePolicy(seed=seed)


def scripted_blue(factory: Optional[Callable[..., Any]] = None, *, seed: Optional[int] = None,
                  name: Optional[str] = None) -> ScriptedBluePolicy:
    """Scripted Nemesis blue team. Defaults to a :class:`BlueCaptain` per zone."""
    if factory is None:
        from nemesis.agents import BlueCaptain

        factory = BlueCaptain
        name = name or "BlueCaptain"
    return ScriptedBluePolicy(factory, seed=seed, name=name)


def trained_blue(checkpoint: str, *, agent: str = "blue_agent_0",
                 device: str = "cpu", name: Optional[str] = None) -> TrainedBluePolicy:
    """A trained SB3 checkpoint, acting greedily for ``agent``."""
    return TrainedBluePolicy(checkpoint, agent=agent, device=device, name=name)


__all__ = [
    "BluePolicy",
    "RandomBluePolicy",
    "ScriptedBluePolicy",
    "TrainedBluePolicy",
    "random_blue",
    "scripted_blue",
    "trained_blue",
]
