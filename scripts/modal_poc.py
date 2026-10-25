"""Proof of concept: train a blue policy, persist it, reload it, evaluate it.

The flow the user asked for — *train a blue policy on Modal, save the checkpoint
to the Volume, shut the compute down, then reload the checkpoint locally and run
one evaluation* — with a ``--local`` path that does the identical sequence now,
with tiny timesteps and **no Modal** (so it runs without any auth).

    # Runs now: fast mock multi-agent env, no Modal, no CybORG build.
    python scripts/modal_poc.py --local

    # Faithful proof on the real (simulated) CybORG CC4 env (~30 s to build it).
    #   set PYTHONPATH to the CybORG submodule first, as the other trainers do:
    #   $env:PYTHONPATH="external\\cage-challenge-4"   (PowerShell)
    python scripts/modal_poc.py --local --cyborg --timesteps 400

The Modal path is driven by `modal run` (documented below / in modal_app/README.md),
never launched here:

    pip install modal && modal token new
    modal run scripts/modal_poc.py            # -> the local_entrypoint() at the bottom

That entrypoint trains on Modal, commits to the Volume, releases the container,
and prints the exact `modal volume get` + local `--eval-only` commands to reload
and evaluate the pulled checkpoint on your machine.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemesis.platform import ArtifactStore, Side
from nemesis.platform.metrics import summarize
from nemesis.platform.storage import LocalBackend
from modal_app.train import (
    default_run_id,
    load_team,
    make_dummy_team_env,
    run_evaluation,
    run_training,
)


def _build_env_factory(args):
    """Return (env_factory, spec). For --cyborg, build the real CC4 env."""
    if args.cyborg:
        from nemesis.platform.envs import blue_training_spec

        spec = blue_training_spec(steps=args.steps, seed=args.seed)

        def factory():
            from nemesis.platform import envs

            return envs.build(spec)

        return factory, spec

    def factory():
        return make_dummy_team_env(
            n_agents=args.n_agents, episode_len=args.episode_len, seed=args.seed
        )

    return factory, None


def run_local(args) -> int:
    root = os.path.abspath(args.store_dir)
    run_id = args.run_id or default_run_id("blue-poc")
    env_factory, spec = _build_env_factory(args)

    print(f"POC store: {root}")
    print(f"POC env:   {'CybORG CC4 (simulated)' if args.cyborg else 'DummyTeamEnv (mock)'}")

    if not args.eval_only:
        # 1. TRAIN, writing the checkpoint through ArtifactStore(LocalBackend).
        print(f"[1/4] train blue (run_id={run_id}, {args.timesteps} timesteps) ...")
        store = ArtifactStore(LocalBackend(root))
        result = run_training(
            store=store,
            side=Side.BLUE,
            run_id=run_id,
            timesteps=args.timesteps,
            env_factory=env_factory,
            spec=spec,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            fresh=args.fresh,
            progress=True,
        )
        print(f"[2/4] saved checkpoint -> {result['checkpoint']} "
              f"({len(result['agents'])} agents, {result['timesteps']} timesteps)")
        # "Shut the compute down": drop the store/learners entirely.
        del store, result

    # 3. RELOAD locally from a *fresh* store over the same root (= the Volume).
    print("[3/4] reload checkpoint from a fresh store ...")
    store2 = ArtifactStore(LocalBackend(root))
    learners, ckpt = load_team(store2, Side.BLUE, run_id)
    print(f"      reloaded {len(learners)} learners from {ckpt.rel_path}")

    # 4. EVALUATE one episode with the reloaded policies.
    print(f"[4/4] evaluate {args.eval_episodes} episode(s) ...")
    eval_env = env_factory()
    metrics = run_evaluation(eval_env, learners, episodes=args.eval_episodes)
    s = summarize(metrics)
    print("      per-episode:", [f"{m.blue_reward:.1f}" for m in metrics])
    print("      summary:", s.as_dict())
    print("POC complete: trained -> persisted -> reloaded -> evaluated.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", action="store_true",
                   help="run the whole train->persist->reload->eval flow locally (no Modal)")
    p.add_argument("--cyborg", action="store_true",
                   help="use the real (simulated) CybORG CC4 env instead of the mock env")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; reload an existing checkpoint and evaluate it")
    p.add_argument("--store-dir", default=os.path.join("checkpoints", "modal_poc"),
                   help="artifact store root (gitignored by default)")
    p.add_argument("--run-id", default=None, help="run id (default: timestamped)")
    p.add_argument("--timesteps", type=int, default=256, help="env steps to train (tiny by default)")
    p.add_argument("--n-steps", type=int, default=64, help="transitions per PPO rollout/update")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-agents", type=int, default=2, help="mock env only: number of blue agents")
    p.add_argument("--episode-len", type=int, default=10, help="mock env only: episode length")
    p.add_argument("--steps", type=int, default=500, help="cyborg only: CC4 episode length")
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="discard any existing checkpoints for the run")
    args = p.parse_args()

    if not (args.local or args.eval_only):
        p.error("pass --local to run the local proof of concept now "
                "(the Modal path runs via `modal run scripts/modal_poc.py`; see the module docstring).")
    if args.eval_only and args.run_id is None:
        p.error("--eval-only needs --run-id (the run whose checkpoint to evaluate)")
    return run_local(args)


# --------------------------------------------------------------------------- #
# Modal entrypoint — only active under `modal run` (modal installed + authed).
# --------------------------------------------------------------------------- #
try:
    from modal_app.images import MODAL_AVAILABLE, VOLUME_NAME, app
except Exception:  # pragma: no cover
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:  # pragma: no cover - needs modal installed + authed
    from modal_app.train import default_run_id as _default_run_id
    from modal_app.train import train_blue

    @app.local_entrypoint()
    def modal_poc(timesteps: int = 50_000, steps: int = 500):
        """Modal POC: train on Modal + commit to Volume, then print pull+eval cmds."""
        run_id = _default_run_id("blue-poc")
        print(f"[1/3] training blue on Modal (run_id={run_id}) ...")
        result = train_blue.remote(
            timesteps=timesteps, run_id=run_id, steps=steps, save_every=timesteps
        )
        print(f"[2/3] trained + committed to the Volume: {result['checkpoint']}")
        print("[3/3] compute is released. Pull the checkpoint and evaluate locally:")
        # The volume root holds `artifacts/`; pull it, then point the store there.
        print(f"      modal volume get {VOLUME_NAME} artifacts ./poc_pull")
        print(f"      python scripts/modal_poc.py --eval-only --cyborg "
              f"--store-dir ./poc_pull/artifacts --run-id {run_id}")


if __name__ == "__main__":
    raise SystemExit(main())
