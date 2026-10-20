"""Smoke test: run one CC4 episode with random blue agents against the default red agent.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_test.py [--steps 100] [--seed 0]
"""
import argparse
import time

import numpy as np

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


def make_env(steps: int, seed: int) -> BlueFlatWrapper:
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=steps,
    )
    return BlueFlatWrapper(CybORG(scenario_generator=sg, seed=seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(args.steps, args.seed)
    obs, _ = env.reset()
    rng = np.random.default_rng(args.seed)

    print("Blue agents:", env.agents)
    for a in env.agents:
        print(f"  {a}: obs={env.observation_space(a).shape} actions={env.action_space(a).n}")

    totals = {a: 0.0 for a in env.agents}
    start = time.time()
    for _ in range(args.steps):
        actions = {a: int(rng.integers(env.action_space(a).n)) for a in env.agents}
        obs, rewards, terminated, truncated, _ = env.step(actions)
        for a, r in rewards.items():
            totals[a] += r
        if all(terminated.values()) or all(truncated.values()):
            break

    print(f"Episode finished in {time.time() - start:.1f}s")
    print("Total reward per agent:", totals)


if __name__ == "__main__":
    main()
