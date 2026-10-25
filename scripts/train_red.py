"""PPO-train one CC4 RED (attacker) agent with Stable-Baselines3, resumable.

The red learner drives ``red_agent_0`` through ``nemesis.red_env.SingleRedEnv``
(see that module and docs/red-training.md for how red is flattened and why the
reward is the negated blue-team score). Everything the model learns (weights,
optimizer, reward-normalisation stats) is checkpointed via ``nemesis.checkpoints``
every --save-every timesteps, on Ctrl+C, and at the end. Checkpoints live under
the Nemesis artifact store at ``<store-root>/red/<run-id>/`` (Side.RED), so
self-play and evaluation can find them. Re-running the same command resumes from
the latest checkpoint and trains until --timesteps total; pass --fresh to start
over.

Alongside each checkpoint the policy network's weights are exported as a small
PyTorch file (``.../weights/sb3_ppo_red/<agent>/policy.pt``) via nemesis.weights.
Outputs stay under the git-ignored store root; nothing binary is committed.

Usage:
    .venv/Scripts/python.exe scripts/train_red.py --timesteps 50000
    .venv/Scripts/python.exe scripts/train_red.py --smoke          # fast end-to-end check
    .venv/Scripts/python.exe scripts/train_red.py --teammates fsm --timesteps 200000
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

from nemesis.checkpoints import LearnedCheckpointCallback, list_checkpoints, load_learned, save_learned
from nemesis.platform.storage import ArtifactStore, Side
from nemesis.red_env import EPISODE_LENGTH, SingleRedEnv
from nemesis.weights import export_sb3_weights

TRAINER = "sb3_ppo_red"


class StopAfterMinutes(BaseCallback):
    """End ``learn`` once a wall-clock budget is spent (the final save still runs)."""

    def __init__(self, minutes: float):
        super().__init__()
        self.deadline = time.time() + minutes * 60

    def _on_step(self) -> bool:
        return time.time() < self.deadline


def _teammate_class(name: str):
    from CybORG.Agents import FiniteStateRedAgent, SleepAgent

    return {"sleep": SleepAgent, "fsm": FiniteStateRedAgent}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="red_agent_0", help="red agent the policy controls")
    parser.add_argument("--timesteps", type=int, default=50_000, help="total timesteps to train to")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH, help="episode length")
    parser.add_argument("--n-envs", type=int, default=1, help="parallel envs (subprocesses when > 1)")
    parser.add_argument("--teammates", choices=["sleep", "fsm"], default="sleep",
                        help="scripted class for the OTHER red agents (sleep=sole learner/clean credit; "
                             "fsm=full scripted red campaign)")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="scales the (negated blue) red reward")
    parser.add_argument("--store-root", default="checkpoints", help="Nemesis artifact-store root (git-ignored)")
    parser.add_argument("--run-id", default=None, help="run id under <store-root>/red/ (default: sb3_<agent>)")
    parser.add_argument("--save-every", type=int, default=10_000, help="timesteps between checkpoints")
    parser.add_argument("--keep-last", type=int, default=3, help="checkpoints to keep (0 keeps all)")
    parser.add_argument("--norm-reward", action="store_true", help="train under VecNormalize reward scaling")
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="discard existing checkpoints and start over")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run for a quick sanity check")
    args = parser.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 20)
        args.timesteps = 512
        args.save_every = 256
        args.store_root = os.path.join("logs", "red_smoke")
        args.run_id = args.run_id or "smoke"

    run_id = args.run_id or f"sb3_{args.agent}"

    # ArtifactStore owns the Side.RED layout; nemesis.checkpoints owns the format
    # inside the resolved run directory. This keeps red checkpoints where the
    # rest of Nemesis (self-play, eval) looks for them.
    store = ArtifactStore(args.store_root)
    run_rel = store.run_dir(Side.RED, run_id)
    store.backend.makedirs(run_rel)
    run_local = store.backend.local_path(run_rel)
    checkpoint_dir = os.path.join(run_local, "sb3")
    weights_dir = os.path.join(run_local, "weights")
    if args.fresh:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(weights_dir, ignore_errors=True)

    env_kwargs = {
        "agent": args.agent,
        "steps": args.steps,
        "reward_scale": args.reward_scale,
        "red_agent_class": _teammate_class(args.teammates),
    }
    env = make_vec_env(
        SingleRedEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )

    if list_checkpoints(checkpoint_dir):
        model = load_learned(checkpoint_dir, env=env)
        print(f"Resumed {args.agent} from {checkpoint_dir} at {model.num_timesteps} timesteps")
    else:
        if args.norm_reward:
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        model = PPO("MlpPolicy", env, n_steps=2048 // args.n_envs, batch_size=256, seed=args.seed, verbose=1)
        print(f"Training {args.agent} from scratch -> {checkpoint_dir}")

    remaining = args.timesteps - model.num_timesteps
    if remaining <= 0:
        print(f"Already trained to {model.num_timesteps} >= {args.timesteps} timesteps; nothing to do")
        env.close()
        return

    extra = {
        "agent": args.agent,
        "side": str(Side.RED),
        "episode_length": args.steps,
        "teammates": args.teammates,
        "reward": "negated_blue_team_reward",
        "run_id": run_id,
    }

    def export_weights(model, _checkpoint=None):
        path = export_sb3_weights(model, args.agent, trainer=TRAINER, root=weights_dir, extra=extra)
        print(f"Exported weights -> {path}")

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
    export_weights(model)
    print(f"Done. Checkpoints under {checkpoint_dir}; weights under {weights_dir}")


if __name__ == "__main__":
    main()
