"""Minimal self-play harness — watch the Nemesis red and blue agents spar.

This is a *readable demo*, not the training loop. It wires our agents into a
CybORG Enterprise scenario, runs a handful of steps, and prints what each side
did and the reward blue earned. Run it from the repo root:

    PYTHONPATH=external/cage-challenge-4 python -m nemesis.selfplay

(Once CybORG is pip-installed into the venv, the PYTHONPATH prefix is unneeded.)
"""

from __future__ import annotations

import warnings
from collections import Counter

warnings.filterwarnings("ignore")  # silence CybORG's gym-deprecation noise

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from nemesis.agents import NemesisBlueAgent, NemesisRedAgent


def run(steps: int = 30) -> None:
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=NemesisBlueAgent,
        red_agent_class=NemesisRedAgent,
        steps=steps,
    )
    env = CybORG(scenario_generator=scenario)
    env.reset()

    actions = Counter()
    for _ in range(steps):
        env.step()
        for agent_name in env.active_agents:
            action = env.get_last_action(agent_name)
            action = action[0] if isinstance(action, list) else action
            side = agent_name.split("_")[0]
            actions[(side, type(action).__name__)] += 1

    print(f"Ran {steps} steps of red-vs-blue self-play.\n")
    print("Actions taken (side, action, count):")
    for (side, name), count in sorted(actions.items()):
        print(f"  {side:5} {name:28} x{count}")


if __name__ == "__main__":
    run()
