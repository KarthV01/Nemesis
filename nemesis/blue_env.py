"""Single-agent Gymnasium view of the CC4 network, for Stable-Baselines3.

SB3 trains one policy against one ``gym.Env``, while CybORG's ``BlueFlatWrapper``
steps all five blue agents at once. ``SingleBlueEnv`` hands SB3 the flat
observation and discrete action space of one chosen blue agent; the other
blue agents ``Sleep``. The reward is that agent's (team-shared) blue reward.
"""

from __future__ import annotations

import gymnasium as gym

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

EPISODE_LENGTH = 500


class SingleBlueEnv(gym.Env):
    """Expose one CC4 blue agent as a standard Gymnasium environment.

    Parameters
    ----------
    agent : str
        Which blue agent SB3 controls (``blue_agent_0`` .. ``blue_agent_4``).
    steps : int
        Episode length. CC4 episodes only end on this limit, so the episode
        end is reported as a truncation, not a termination.
    red_agent_class : type
        Attacker to train against.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent: str = "blue_agent_0",
        steps: int = EPISODE_LENGTH,
        red_agent_class: type = FiniteStateRedAgent,
    ):
        super().__init__()
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=red_agent_class,
            steps=steps,
        )
        self._cyborg = CybORG(scenario_generator=scenario)
        self._env = BlueFlatWrapper(self._cyborg)
        if agent not in self._env.possible_agents:
            raise ValueError(f"{agent!r} is not one of {self._env.possible_agents}")
        self.agent = agent
        self.observation_space = self._env.observation_space(agent)
        self.action_space = self._env.action_space(agent)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, info = self._env.reset(seed=seed)
        return observations[self.agent], info.get(self.agent, {})

    def step(self, action):
        actions = {a: Sleep() for a in self._env.agents if a != self.agent}
        actions[self.agent] = int(action)
        observations, rewards, _, truncated, info = self._env.step(actions)
        return (
            observations[self.agent],
            float(rewards[self.agent]),
            False,
            bool(truncated[self.agent]),
            info.get(self.agent, {}),
        )

    def action_labels(self) -> list:
        """Human-readable meaning of each action index, for inspecting a policy."""
        return self._env.action_labels(self.agent)
