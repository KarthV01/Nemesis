"""Checkpoint-backed PyTorch policies sharing one CybORG arena simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from nemesis.ppo_learner import PPOLearner
from nemesis.red_env import SingleRedEnv


class LearnedArena:
    """Drive learned red and/or blue policies in one simultaneous simulation.

    Blue observations/actions use CybORG's ``BlueFlatWrapper`` exactly as they
    do during training. Red policies reuse ``SingleRedEnv`` as a codec attached
    to the same raw CybORG instance. This prevents checkpoint evaluation from
    drifting away from either side's training representation.
    """

    def __init__(self, request, artifact_root: Path):
        from CybORG import CybORG
        from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
        from nemesis.arena.runner import _resolve_agent

        red_name, blue_name = request.agent_classes()
        red_class = SleepAgent if request.red_checkpoint else _resolve_agent(red_name)
        blue_class = SleepAgent if request.blue_checkpoint else _resolve_agent(blue_name)
        scenario = EnterpriseScenarioGenerator(
            red_agent_class=red_class,
            blue_agent_class=blue_class,
            green_agent_class=EnterpriseGreenAgent,
            steps=request.steps,
        )
        self.raw = CybORG(scenario_generator=scenario, seed=request.seed)
        self.blue_wrapper = BlueFlatWrapper(self.raw)
        self.request = request
        self.artifact_root = Path(artifact_root).resolve()
        self.blue_learners = self._load_blue(request.blue_checkpoint)
        self.red_learners = self._load_red(request.red_checkpoint)
        self.red_codecs: Dict[str, SingleRedEnv] = {}
        self.blue_agents = []
        self.red_agents = []
        self.blue_observations = {}
        self.red_observations = {}

    def _checkpoint_path(self, reference: str) -> Path:
        if reference == "ppo_team/default":
            reference = "ppo_team"
        path = (self.artifact_root / reference).resolve()
        if self.artifact_root != path and self.artifact_root not in path.parents:
            raise ValueError("checkpoint reference escapes artifact root")
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {reference}")
        return path

    def _load_blue(self, reference: Optional[str]) -> Dict[str, PPOLearner]:
        if not reference:
            return {}
        root = self._checkpoint_path(reference)
        learners = {}
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("blue_agent_"):
                learners[child.name] = PPOLearner.load(child)
        if not learners:
            raise FileNotFoundError(f"blue checkpoint has no agent policies: {reference}")
        return learners

    def _load_red(self, reference: Optional[str]) -> Dict[str, PPOLearner]:
        if not reference:
            return {}
        root = self._checkpoint_path(reference)
        learners = {}
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("red_agent_"):
                learners[child.name] = PPOLearner.load(child)
        if not learners:
            raise FileNotFoundError(f"red checkpoint has no agent policies: {reference}")
        return learners

    @staticmethod
    def _five(names: Iterable[str], side: str):
        return sorted(name for name in names if name.startswith(f"{side}_agent_"))[:5]

    def reset(self):
        blue_obs, _ = self.blue_wrapper.reset(seed=self.request.seed)
        all_agents = list(self.raw.agents)
        if self.request.format == "1v1":
            self.blue_agents = ["blue_agent_0"]
            self.red_agents = ["red_agent_0"]
        else:
            self.blue_agents = self._five(all_agents, "blue")
            self.red_agents = self._five(all_agents, "red")

        if self.blue_learners:
            missing = set(self.blue_agents) - set(self.blue_learners)
            if missing:
                raise ValueError(f"blue checkpoint is missing policies: {sorted(missing)}")
            self.blue_observations = {a: blue_obs[a] for a in self.blue_agents}

        if self.red_learners:
            fallback = self.red_learners.get("red_agent_0")
            for agent in self.red_agents:
                if agent not in self.red_learners:
                    if fallback is None:
                        raise ValueError(f"red checkpoint is missing policy {agent}")
                    self.red_learners[agent] = fallback
                codec = SingleRedEnv(agent=agent, steps=self.request.steps)
                codec.attach_cyborg(self.raw)
                self.red_codecs[agent] = codec
            self.red_observations = {
                a: self.red_codecs[a].current_observation() for a in self.red_agents
            }

    def step(self):
        actions = {}
        if self.blue_learners:
            for agent in self.blue_agents:
                index = self.blue_learners[agent].predict(
                    self.blue_observations[agent], deterministic=True
                )
                actions[agent] = self.blue_wrapper.actions(agent)[int(index)]
        if self.red_learners:
            for agent in self.red_agents:
                encoded = self.red_learners[agent].predict(
                    self.red_observations[agent], deterministic=True
                )
                actions[agent] = self.red_codecs[agent].decode_action(encoded)

        observations, rewards, dones, _ = self.raw.parallel_step(actions)
        if self.blue_learners:
            self.blue_observations = {
                agent: self.blue_wrapper.observation_change(agent, observations[agent])
                for agent in self.blue_agents
            }
        if self.red_learners:
            self.red_observations = {
                agent: self.red_codecs[agent].current_observation()
                for agent in self.red_agents
            }
        return actions, rewards, dones

    def close(self):
        close = getattr(self.raw, "close", None)
        if close is not None:
            close()


__all__ = ["LearnedArena"]
