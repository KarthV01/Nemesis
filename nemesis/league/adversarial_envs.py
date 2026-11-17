"""Shared CybORG environments for alternating learned red/blue training."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from nemesis.ppo_learner import PPOLearner
from nemesis.red_env import SingleRedEnv


def _raw_env(steps: int, seed=None):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    scenario = EnterpriseScenarioGenerator(
        red_agent_class=SleepAgent,
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        steps=steps,
    )
    return CybORG(scenario_generator=scenario, seed=seed)


def _load_team(checkpoint: Path, prefix: str):
    learners = {}
    for child in Path(checkpoint).iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            learners[child.name] = PPOLearner.load(child)
    if not learners:
        raise FileNotFoundError(f"{checkpoint} contains no {prefix} policies")
    return learners


class BlueAgainstRedEnv:
    """BlueFlatWrapper-compatible team env with a frozen learned red team."""

    def __init__(self, red_checkpoint, *, steps: int, seed=None):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        self.raw = _raw_env(steps, seed)
        self.steps = steps
        self.wrapper = BlueFlatWrapper(self.raw)
        self.red_learners = _load_team(Path(red_checkpoint), "red_agent_")
        self.red_codecs = {}
        self.red_observations = {}
        self.possible_agents = self.wrapper.possible_agents
        self.agents = list(self.possible_agents)

    def observation_space(self, agent):
        return self.wrapper.observation_space(agent)

    def action_space(self, agent):
        return self.wrapper.action_space(agent)

    def reset(self, *, seed=None, options=None):
        observations, info = self.wrapper.reset(seed=seed)
        self.agents = list(self.possible_agents)
        red_agents = sorted(a for a in self.raw.agents if a.startswith("red_agent_"))[:5]
        fallback = self.red_learners.get("red_agent_0")
        self.red_codecs = {}
        for agent in red_agents:
            if agent not in self.red_learners:
                if fallback is None:
                    raise ValueError(f"red checkpoint is missing {agent}")
                self.red_learners[agent] = fallback
            codec = SingleRedEnv(agent=agent, steps=self.steps)
            codec.attach_cyborg(self.raw)
            self.red_codecs[agent] = codec
        self.red_observations = {
            agent: codec.current_observation() for agent, codec in self.red_codecs.items()
        }
        return observations, info

    def step(self, blue_actions):
        actions = {
            agent: self.wrapper.actions(agent)[int(action)]
            for agent, action in blue_actions.items()
        }
        for agent, codec in self.red_codecs.items():
            encoded = self.red_learners[agent].predict(
                self.red_observations[agent], deterministic=True
            )
            actions[agent] = codec.decode_action(encoded)
        observations, rewards, dones, _ = self.raw.parallel_step(actions)
        blue_obs = {
            agent: self.wrapper.observation_change(agent, observations[agent])
            for agent in self.agents
        }
        self.red_observations = {
            agent: codec.current_observation() for agent, codec in self.red_codecs.items()
        }
        blue_rewards = {
            agent: float(sum(rewards.get(agent, {}).values())) for agent in self.agents
        }
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: bool(dones.get(agent, False)) for agent in self.agents}
        return blue_obs, blue_rewards, terminated, truncated, {a: {} for a in self.agents}

    def close(self):
        close = getattr(self.raw, "close", None)
        if close is not None:
            try:
                close()
            except AttributeError:
                # Simulator-only CybORG instances in CC4 do not initialize
                # the emulator GUI flag that its close() method reads.
                pass


class RedAgainstBlueEnv(gym.Env):
    """Single learned red policy against a frozen five-policy blue team."""

    def __init__(self, blue_checkpoint, *, steps: int, seed=None):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        super().__init__()
        self.raw = _raw_env(steps, seed)
        self.wrapper = BlueFlatWrapper(self.raw)
        self.blue_learners = _load_team(Path(blue_checkpoint), "blue_agent_")
        self.codec = SingleRedEnv(agent="red_agent_0", steps=steps)
        self.observation_space = self.codec.observation_space
        self.action_space = self.codec.action_space
        self.blue_observations = {}

    def reset(self, *, seed=None, options=None):
        observations, _ = self.wrapper.reset(seed=seed)
        missing = set(self.wrapper.possible_agents) - set(self.blue_learners)
        if missing:
            raise ValueError(f"blue checkpoint is missing policies: {sorted(missing)}")
        self.blue_observations = observations
        self.codec.attach_cyborg(self.raw)
        return self.codec.current_observation(), {}

    def step(self, red_action):
        actions = {}
        for agent in self.wrapper.possible_agents:
            index = self.blue_learners[agent].predict(
                self.blue_observations[agent], deterministic=True
            )
            actions[agent] = self.wrapper.actions(agent)[int(index)]
        actions["red_agent_0"] = self.codec.decode_action(red_action)
        observations, rewards, dones, _ = self.raw.parallel_step(actions)
        self.blue_observations = {
            agent: self.wrapper.observation_change(agent, observations[agent])
            for agent in self.wrapper.possible_agents
        }
        blue_reward = sum(rewards.get("blue_agent_0", {}).values())
        truncated = bool(dones.get("red_agent_0", False))
        return self.codec.current_observation(), -float(blue_reward), False, truncated, {
            "blue_reward": float(blue_reward)
        }

    def close(self):
        close = getattr(self.raw, "close", None)
        if close is not None:
            try:
                close()
            except AttributeError:
                pass


__all__ = ["BlueAgainstRedEnv", "RedAgainstBlueEnv"]
