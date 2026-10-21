"""Focused regression tests for Nemesis specialists and captains."""

from CybORG import CybORG
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from nemesis.agents import BlueCaptain, NemesisAgent, RedCaptain


class PreferredAction:
    pass


class UnrelatedAction:
    pass


class TargetedAction:
    def __init__(self, hostname):
        self.hostname = hostname


def test_strict_agent_exploration_stays_within_its_role():
    agent = NemesisAgent(name="test_agent", epsilon=1.0)
    agent.strict_priority = True
    agent.action_priority = ["PreferredAction"]
    action_space = {
        "action": {PreferredAction: True, UnrelatedAction: True},
    }

    assert isinstance(agent.get_action({}, action_space), PreferredAction)


def test_strict_agent_sleeps_when_its_role_has_no_legal_action():
    agent = NemesisAgent(name="test_agent", epsilon=1.0)
    agent.strict_priority = True
    agent.action_priority = ["PreferredAction"]
    action_space = {
        "action": {PreferredAction: False, UnrelatedAction: True},
    }

    assert isinstance(agent.get_action({}, action_space), Sleep)


def test_preferred_target_is_used_when_legal():
    agent = NemesisAgent(name="test_agent")
    agent.action_priority = ["TargetedAction"]
    agent.preferred_targets = ["host_b"]
    action_space = {
        "action": {TargetedAction: True},
        "hostname": {"host_a": True, "host_b": True},
    }

    assert agent.get_action({}, action_space).hostname == "host_b"


def test_captains_complete_a_short_episode():
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=BlueCaptain,
        red_agent_class=RedCaptain,
        steps=3,
    )
    env = CybORG(scenario_generator=scenario)
    env.reset()

    for _ in range(3):
        env.step()
