"""Train independent PPO policies for the 5 CC4 blue agents with RLlib.

Adapted from external/cage-challenge-4/CybORG/Evaluation/training_example/TrainingRay.py.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_ppo_rllib.py --iterations 50
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

NUM_AGENTS = 5
EPISODE_LENGTH = 500
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}


def env_creator(env_config: dict) -> EnterpriseMAE:
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=env_config.get("steps", EPISODE_LENGTH),
    )
    return EnterpriseMAE(CybORG(scenario_generator=sg))


def policy_mapper(agent_id, episode, worker=None, **kwargs):
    return POLICY_MAP[agent_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH, help="episode length")
    parser.add_argument("--batch-size", type=int, default=4000, help="env steps per training iteration")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_cc4")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    register_env("CC4", env_creator)
    env = env_creator({"steps": args.steps})

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .framework("torch")
        .environment(env="CC4", env_config={"steps": args.steps})
        # CybORG steps are slow (~50ms), so give rollouts plenty of time.
        .env_runners(num_env_runners=args.workers, sample_timeout_s=1800)
        .training(train_batch_size=args.batch_size, minibatch_size=min(256, args.batch_size), num_epochs=10, gamma=0.99, lr=3e-4)
        .multi_agent(
            policies={
                policy_id: PolicySpec(
                    observation_space=env.observation_space(agent_id),
                    action_space=env.action_space(agent_id),
                )
                for agent_id, policy_id in POLICY_MAP.items()
            },
            policy_mapping_fn=policy_mapper,
        )
    )
    algo = config.build()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    for i in range(1, args.iterations + 1):
        result = algo.train()
        runners = result.get("env_runners", {})
        print(
            f"iter {i:4d} | reward_mean {runners.get('episode_reward_mean', float('nan')):9.2f}"
            f" | episodes {runners.get('episodes_this_iter', 0)}"
        )
        if i % args.checkpoint_every == 0 or i == args.iterations:
            print("  checkpoint ->", algo.save(checkpoint_dir).checkpoint.path)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
