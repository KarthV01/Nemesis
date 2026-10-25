"""Run Nemesis adversarial self-play: blue PPO vs a frozen-opponent league.

Blue (five PPO policies) trains against a league of frozen red opponents,
sampled newest-with-probability ``--latest-prob`` else uniformly from history,
so it does not overfit to the latest rival. Each generation freezes the current
blue weights into a new league snapshot and periodically evaluates blue against
a *set* of frozen opponents. Red learning is the integrator hook (see
``docs/selfplay.md`` / ``nemesis.league.protocol``); without it the loop uses
the scripted red roster and runs end-to-end today.

Outputs (checkpoints, experiment logs, events) go under ``--artifacts``, which
defaults to a .gitignore'd directory.

Usage:
    $env:PYTHONPATH="external\\cage-challenge-4"   # if CybORG is not pip-installed
    .venv\\Scripts\\python.exe scripts\\train_selfplay.py --smoke
    .venv/bin/python scripts/train_selfplay.py --generations 10 --blue-steps 8000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemesis.league.selfplay_loop import SelfPlayConfig, run_selfplay


def build_config(args) -> SelfPlayConfig:
    if args.smoke:
        overrides = {}
        if args.name:
            overrides["name"] = args.name
        if args.artifacts:
            overrides["artifacts_root"] = args.artifacts
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.verbose:
            overrides["verbose"] = args.verbose
        return SelfPlayConfig.smoke(**overrides)

    kwargs = dict(
        generations=args.generations,
        blue_steps_per_gen=args.blue_steps,
        red_steps_per_gen=args.red_steps,
        episode_steps=args.episode_steps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_opponents=args.eval_opponents,
        latest_prob=args.latest_prob,
        seed=args.seed,
        artifacts_root=args.artifacts,
        verbose=args.verbose,
    )
    if args.name:
        kwargs["name"] = args.name
    if args.red_roster:
        kwargs["scripted_red_roster"] = tuple(args.red_roster)
    if args.blue_win_threshold is not None:
        kwargs["blue_win_threshold"] = args.blue_win_threshold
    return SelfPlayConfig(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="tiny fast run that exercises the whole loop")
    p.add_argument("--name", default=None, help="experiment name (dir under <artifacts>/experiments)")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--blue-steps", type=int, default=4000, help="env steps of blue training per generation")
    p.add_argument("--red-steps", type=int, default=4000, help="env steps of red training per generation (red hook)")
    p.add_argument("--episode-steps", type=int, default=500, help="CC4 episode length")
    p.add_argument("--n-steps", type=int, default=2048, help="transitions per PPO rollout/update per blue learner")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=1, help="evaluate every N generations (0 disables)")
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--eval-opponents", type=int, default=3, help="distinct frozen red opponents per evaluation")
    p.add_argument("--latest-prob", type=float, default=0.5, help="prob of sampling the newest opponent vs a historical one")
    p.add_argument("--red-roster", nargs="*", default=None, help="scripted red agent class names forming the frozen roster")
    p.add_argument("--blue-win-threshold", type=float, default=None, help="label blue_win when team reward >= this")
    p.add_argument("--artifacts", default=os.path.join("runs", "selfplay"), help="artifact store root (keep under a .gitignore'd dir)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", type=int, default=0, help="1 prints SB3 training stats")
    args = p.parse_args()

    config = build_config(args)
    print(f"Self-play '{config.name}': {config.generations} gen x {config.blue_steps_per_gen} blue steps, "
          f"roster={list(config.scripted_red_roster)} -> {config.artifacts_root}")
    league = run_selfplay(config)

    print(f"\nDone. {len(league)} league snapshots. Standings (hardest opponents first):")
    for s in league.standings(side="red"):
        parts = [f"  {s.opponent:20} games={s.games:3d}"]
        if s.mean_blue_reward is not None:
            parts.append(f"mean_blue_reward={s.mean_blue_reward:.2f}")
        else:
            parts.append("(no eval yet)")
        if s.win_rate is not None:
            parts.append(f"win_rate={s.win_rate:.2f}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
