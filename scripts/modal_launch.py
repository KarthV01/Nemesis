"""Durably start a deployed Nemesis self-play job and return immediately.

Unlike ``modal run scripts/modal_train.py``, this launcher looks up the already
deployed function and uses ``spawn``. The training input therefore survives the
local CLI process disconnecting.
"""

from __future__ import annotations

from typing import Optional

import modal

from modal_app.images import APP_NAME


launcher = modal.App("nemesis-launcher")


@launcher.local_entrypoint()
def main(
    run_id: Optional[str] = None,
    generations: int = 5,
    blue_steps: int = 4_000,
    red_steps: int = 4_000,
    episode_steps: int = 500,
    n_steps: int = 2_048,
    batch_size: int = 256,
    eval_every: int = 1,
    eval_episodes: int = 3,
    seed: Optional[int] = None,
):
    function = modal.Function.from_name(APP_NAME, "train_selfplay")
    call = function.spawn(
        run_id=run_id,
        generations=generations,
        blue_steps=blue_steps,
        red_steps=red_steps,
        episode_steps=episode_steps,
        n_steps=n_steps,
        batch_size=batch_size,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
    )
    print(f"Started deployed train_selfplay call: {call.object_id}")
