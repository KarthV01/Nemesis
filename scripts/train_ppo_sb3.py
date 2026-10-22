"""Train one CC4 blue agent with Stable-Baselines3 PPO, checkpointing as it learns.

Everything the model has learned (weights, optimizer, reward-normalisation
stats) is saved every --save-every timesteps via nemesis.checkpoints, on
Ctrl+C, and at the end. Re-running the same command resumes from the latest
checkpoint and trains until --timesteps total, so a run can be stopped and
continued at will. Pass --fresh to start over.

Alongside each checkpoint, the policy network's weights are exported as a small
PyTorch file, weights/sb3_ppo/<agent>/policy.pt, meant to live in git. With
--push-branch they are committed to that branch and pushed every
--push-every-min minutes and at the end (without touching the checked-out
branch; see nemesis.git_sync).

Usage:
    $env:PYTHONPATH="external\\cage-challenge-4"
    .venv\\Scripts\\python.exe scripts\\train_ppo_sb3.py --agent blue_agent_0 --timesteps 200000
    .venv/bin/python scripts/train_ppo_sb3.py --agent blue_agent_4 --n-envs 4 --time-budget-min 50
    .venv/bin/python scripts/train_ppo_sb3.py --agent blue_agent_0 --push-branch training/weights
"""
import argparse
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from nemesis.blue_env import EPISODE_LENGTH, SingleBlueEnv
from nemesis.checkpoints import LearnedCheckpointCallback, list_checkpoints, load_learned, save_learned
from nemesis.git_sync import PeriodicGitSync
from nemesis.weights import WEIGHTS_ROOT, export_sb3_weights


class StopAfterMinutes(BaseCallback):
    """End ``learn`` once a wall-clock budget is spent (the final save still runs)."""

    def __init__(self, minutes: float):
        super().__init__()
        self.deadline = time.time() + minutes * 60

    def _on_step(self) -> bool:
        return time.time() < self.deadline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="blue_agent_0", help="blue agent SB3 controls")
    parser.add_argument("--timesteps", type=int, default=200_000, help="total timesteps to train to")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH, help="episode length")
    parser.add_argument("--n-envs", type=int, default=1, help="parallel envs (subprocesses when > 1)")
    parser.add_argument("--checkpoint-dir", default=None, help="default: checkpoints/sb3_<agent>")
    parser.add_argument("--save-every", type=int, default=20_000, help="timesteps between checkpoints")
    parser.add_argument("--keep-last", type=int, default=3, help="checkpoints to keep (0 keeps all)")
    parser.add_argument("--norm-reward", action="store_true", help="train under VecNormalize reward scaling")
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="discard existing checkpoints and start over")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weights-dir", default=str(WEIGHTS_ROOT), help="where policy.pt files are exported")
    parser.add_argument("--push-branch", default=None, help="commit + push weights to this branch (e.g. training/weights)")
    parser.add_argument("--push-every-min", type=float, default=30.0, help="minutes between weight pushes")
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir or os.path.join("checkpoints", f"sb3_{args.agent}"))
    if args.fresh:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    env = make_vec_env(
        SingleBlueEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"agent": args.agent, "steps": args.steps},
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )

    if list_checkpoints(checkpoint_dir):
        # VecNormalize stats, if any, come back from the checkpoint itself.
        model = load_learned(checkpoint_dir, env=env)
        print(f"Resumed {args.agent} from {checkpoint_dir} at {model.num_timesteps} timesteps")
    else:
        if args.norm_reward:
            # Observations are categorical (MultiDiscrete, one-hot encoded by the
            # policy), so only the reward is normalised.
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        model = PPO("MlpPolicy", env, n_steps=2048 // args.n_envs, batch_size=256, seed=args.seed, verbose=1)
        print(f"Training {args.agent} from scratch -> {checkpoint_dir}")

    remaining = args.timesteps - model.num_timesteps
    if remaining <= 0:
        print(f"Already trained to {model.num_timesteps} >= {args.timesteps} timesteps; nothing to do")
        env.close()
        return

    extra = {"agent": args.agent, "episode_length": args.steps, "red_agent": "FiniteStateRedAgent"}
    sync = PeriodicGitSync([args.weights_dir], args.push_branch, every_minutes=args.push_every_min) if args.push_branch else None

    def export_weights(model, _checkpoint=None, final=False):
        path = export_sb3_weights(model, args.agent, root=args.weights_dir, extra=extra)
        print(f"Exported weights -> {path}")
        if sync is not None:
            message = f"{args.agent}: PPO weights at {model.num_timesteps} timesteps"
            sync.push(message) if final else sync.maybe_push(message)

    callbacks = [
        LearnedCheckpointCallback(
            checkpoint_dir,
            save_freq=args.save_every,
            keep_last=args.keep_last or None,
            extra=extra,
            on_save=export_weights,
            verbose=1,
        )
    ]
    if args.time_budget_min is not None:
        callbacks.append(StopAfterMinutes(args.time_budget_min))

    try:
        model.learn(remaining, callback=CallbackList(callbacks), reset_num_timesteps=False)
    except KeyboardInterrupt:
        path = save_learned(model, checkpoint_dir, extra=extra, keep_last=args.keep_last or None)
        print(f"\nInterrupted at {model.num_timesteps} timesteps; saved -> {path}")
    finally:
        env.close()
    # Covers Ctrl+C too, and always pushes the final weights regardless of the interval.
    export_weights(model, final=True)


if __name__ == "__main__":
    main()
