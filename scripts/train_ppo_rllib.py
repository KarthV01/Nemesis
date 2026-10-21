"""Train independent PPO policies for the 5 CC4 blue agents with RLlib.

Adapted from external/cage-challenge-4/CybORG/Evaluation/training_example/TrainingRay.py.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_ppo_rllib.py --iterations 50
    .venv/bin/python scripts/train_ppo_rllib.py --iterations 1000 --time-budget-min 50
"""
import argparse
import csv
import os
import time
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


def default_workers() -> int:
    # Leave one core for the driver/learner.
    return max(1, (os.cpu_count() or 2) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None, help="rollout workers (default: CPU count - 1)")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH, help="episode length")
    parser.add_argument("--batch-size", type=int, default=4000, help="env steps per training iteration")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_cc4")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--time-budget-min",
        type=float,
        default=None,
        help="stop and save a final checkpoint once this many minutes have elapsed",
    )
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else default_workers()

    start = time.time()
    ray.init(ignore_reinit_error=True)
    register_env("CC4", env_creator)
    env = env_creator({"steps": args.steps})

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .framework("torch")
        .environment(env="CC4", env_config={"steps": args.steps})
        # CybORG steps are slow (~50ms), so give rollouts plenty of time.
        .env_runners(num_env_runners=workers, sample_timeout_s=1800)
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
    print(f"Training with {workers} rollout workers")

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["iteration", "elapsed_s", "episode_reward_mean", "episodes_this_iter"])

    # Track how long an iteration takes so we stop before overrunning the budget.
    iter_seconds = 0.0
    for i in range(1, args.iterations + 1):
        iter_start = time.time()
        result = algo.train()
        iter_seconds = max(iter_seconds, time.time() - iter_start)

        runners = result.get("env_runners", {})
        reward = runners.get("episode_reward_mean", float("nan"))
        episodes = runners.get("episodes_this_iter", 0)
        elapsed = time.time() - start
        print(f"iter {i:4d} | {elapsed / 60:6.1f} min | reward_mean {reward:9.2f} | episodes {episodes}", flush=True)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([i, round(elapsed, 1), reward, episodes])

        out_of_time = (
            args.time_budget_min is not None
            and elapsed + iter_seconds > args.time_budget_min * 60
        )
        if i % args.checkpoint_every == 0 or i == args.iterations or out_of_time:
            print("  checkpoint ->", algo.save(checkpoint_dir).checkpoint.path, flush=True)
        if out_of_time:
            print(f"Time budget of {args.time_budget_min} min reached after {i} iterations")
            break

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
