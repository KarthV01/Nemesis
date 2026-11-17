"""Launch production Nemesis PPO training on Modal.

Examples:
    modal run scripts/modal_train.py --side selfplay --generations 5
    modal run scripts/modal_train.py --side blue --timesteps 200000
    modal run scripts/modal_train.py --side red --timesteps 200000
"""

from __future__ import annotations

from typing import Optional

from modal_app.images import MODAL_AVAILABLE, app

if not MODAL_AVAILABLE:  # pragma: no cover - Modal CLI imports this after install
    raise RuntimeError("Install Modal first: pip install modal")

from modal_app.train import train_blue, train_red, train_selfplay


@app.local_entrypoint()
def main(
    side: str = "selfplay",
    timesteps: int = 200_000,
    run_id: Optional[str] = None,
    steps: int = 500,
    n_steps: int = 2048,
    batch_size: int = 256,
    save_every: int = 20_000,
    seed: Optional[int] = None,
    generations: int = 5,
    blue_steps: int = 4_000,
    red_steps: int = 4_000,
):
    """Start a cloud job; no training executes on the caller's machine."""
    if side not in {"selfplay", "blue", "red"}:
        raise ValueError("side must be 'selfplay', 'blue', or 'red'")
    if side == "selfplay":
        result = train_selfplay.remote(
            run_id=run_id,
            generations=generations,
            blue_steps=blue_steps,
            red_steps=red_steps,
            episode_steps=steps,
            n_steps=n_steps,
            batch_size=batch_size,
            seed=seed,
        )
        print(result)
        return
    function = train_blue if side == "blue" else train_red
    result = function.remote(
        timesteps=timesteps,
        run_id=run_id,
        steps=steps,
        n_steps=n_steps,
        batch_size=batch_size,
        save_every=save_every,
        seed=seed,
    )
    print(result)
