"""Train all five CC4 blue agents together, each with its own PPOLearner.

Every blue agent acts every step in one shared CybORG episode (no agent sits
out), and each one's policy weights update whenever its rollout fills. Each
agent checkpoints separately via nemesis.checkpoints under
<checkpoint-dir>/<agent>, every --save-every timesteps, on Ctrl+C, and at the
end. Re-running resumes every agent from its latest checkpoint; pass --fresh to
start over. A checkpoint is a plain SB3 PPO model, so load_learned works on it.

Usage:
    $env:PYTHONPATH="external\\cage-challenge-4"
    .venv\\Scripts\\python.exe scripts\\train_ppo_team.py --timesteps 100000
    .venv/bin/python scripts/train_ppo_team.py --timesteps 500000 --time-budget-min 50
"""
import argparse
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from nemesis.checkpoints import list_checkpoints
from nemesis.ppo_learner import PPOLearner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000, help="env steps to train to (each agent sees each step)")
    parser.add_argument("--steps", type=int, default=500, help="episode length")
    parser.add_argument("--n-steps", type=int, default=2048, help="transitions per rollout / PPO update")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-dir", default=os.path.join("checkpoints", "ppo_team"))
    parser.add_argument("--save-every", type=int, default=20_000, help="timesteps between checkpoints")
    parser.add_argument("--keep-last", type=int, default=3, help="checkpoints to keep per agent (0 keeps all)")
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="discard existing checkpoints and start over")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0, help="1 prints SB3 stats after every update")
    args = parser.parse_args()

    root = os.path.abspath(args.checkpoint_dir)
    if args.fresh:
        shutil.rmtree(root, ignore_errors=True)

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=args.steps,
    )
    env = BlueFlatWrapper(CybORG(scenario_generator=scenario, seed=args.seed))
    observations, _ = env.reset()

    learners = {}
    for agent in env.possible_agents:
        agent_dir = os.path.join(root, agent)
        if list_checkpoints(agent_dir):
            learners[agent] = PPOLearner.load(agent_dir, total_timesteps=args.timesteps, verbose=args.verbose)
            print(f"Resumed {agent} at {learners[agent].num_timesteps} timesteps")
        else:
            learners[agent] = PPOLearner(
                env.observation_space(agent),
                env.action_space(agent),
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                total_timesteps=args.timesteps,
                seed=args.seed,
                verbose=args.verbose,
            )
    start = min(learner.num_timesteps for learner in learners.values())
    print(f"Training {len(learners)} blue agents from {start} to {args.timesteps} timesteps -> {root}")

    extra = {"episode_length": args.steps, "red_agent": "FiniteStateRedAgent", "trained_with": "train_ppo_team"}

    def save_all(reason: str) -> None:
        for agent, learner in learners.items():
            learner.save(os.path.join(root, agent), extra={**extra, "agent": agent}, keep_last=args.keep_last or None)
        print(f"[{reason}] saved {len(learners)} agents at {start + step} timesteps")

    deadline = time.time() + args.time_budget_min * 60 if args.time_budget_min is not None else None
    episode_return, episode = 0.0, 0
    step = 0
    try:
        while start + step < args.timesteps and (deadline is None or time.time() < deadline):
            actions = {agent: learners[agent].act(observations[agent]) for agent in env.agents}
            observations, rewards, terminated, truncated, _ = env.step(actions)
            for agent in actions:
                learners[agent].record(rewards[agent], observations[agent], terminated[agent], truncated[agent])
            step += 1
            # Blue reward is team-shared, so any one agent's reward is the team's.
            episode_return += rewards[env.possible_agents[0]]

            if all(terminated[a] or truncated[a] for a in actions):
                episode += 1
                print(f"episode {episode}: blue return {episode_return:.1f} ({start + step} timesteps)")
                episode_return = 0.0
                observations, _ = env.reset()
            if step % args.save_every == 0:
                save_all("periodic")
    except KeyboardInterrupt:
        # A step interrupted between act() and record() has nothing pending
        # worth keeping; the saved weights are those of the last full update.
        save_all("interrupted")
        return
    save_all("final")


if __name__ == "__main__":
    main()
