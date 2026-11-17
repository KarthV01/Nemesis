"""End-to-end checkpoint-backed arena smoke test using real CybORG."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("CybORG")

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from nemesis.arena.models import MatchRequest
from nemesis.arena.runner import run_match
from nemesis.ppo_learner import PPOLearner
from nemesis.red_env import SingleRedEnv


def test_learned_red_and_blue_checkpoint_duel(tmp_path):
    artifacts = tmp_path / "artifacts"

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        steps=2,
    )
    blue_env = BlueFlatWrapper(CybORG(scenario_generator=scenario, seed=1))
    blue_env.reset(seed=1)
    blue = PPOLearner(
        blue_env.observation_space("blue_agent_0"),
        blue_env.action_space("blue_agent_0"),
        n_steps=4,
        batch_size=2,
        seed=1,
    )
    blue.save(
        artifacts / "blue" / "test" / "checkpoint_000001" / "blue_agent_0"
    )

    red_env = SingleRedEnv(agent="red_agent_0", steps=2)
    red = PPOLearner(
        red_env.observation_space,
        red_env.action_space,
        n_steps=4,
        batch_size=2,
        seed=1,
    )
    red.save(
        artifacts / "red" / "test" / "checkpoint_000001" / "red_agent_0"
    )

    request = MatchRequest(
        format="1v1",
        red_role="recon",
        blue_role="detect",
        red_checkpoint="red/test/checkpoint_000001",
        blue_checkpoint="blue/test/checkpoint_000001",
        steps=2,
        delay=0,
        seed=1,
    )
    result = run_match(request, tmp_path / "match", artifact_root=artifacts)

    assert result["status"] == "completed"
    assert result["steps"] == 2
    events = [json.loads(line) for line in (tmp_path / "match" / "events.jsonl").read_text().splitlines()]
    acting = {event.get("agent") for event in events if event["kind"] in {"red", "blue"}}
    assert acting <= {"red_agent_0", "blue_agent_0"}
