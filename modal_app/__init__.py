"""Nemesis Modal cloud-compute stage.

Runs PPO training jobs on Modal and persists results to a Modal Volume, with the
*same* code path runnable locally. Every ``import modal`` is guarded, so this
package imports and its non-Modal paths run (and test) with ``modal`` absent;
``MODAL_AVAILABLE`` says whether the real Modal objects are live.

Public surface:

* :data:`MODAL_AVAILABLE`, :data:`APP_NAME`, :data:`VOLUME_NAME`,
  :data:`VOLUME_MOUNT_PATH`, :data:`ARTIFACTS_PATH`, and the ``app`` / ``image``
  / ``volume`` objects (``None`` without modal) — from :mod:`modal_app.images`.
* :class:`VolumeBackend` — a ``nemesis.platform.storage.Backend`` over a mounted
  Volume path.
* ``modal_app.train.run_training`` / ``run_evaluation`` — the plain, Modal-free
  training and evaluation callables both the Modal functions and ``--local``
  use. They live in :mod:`modal_app.train` (imported on demand, since they pull
  in torch/SB3) rather than being re-exported here.
"""

from modal_app.images import (
    APP_NAME,
    ARTIFACTS_PATH,
    MODAL_AVAILABLE,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
    app,
    image,
    volume,
)
from modal_app.volume_backend import VolumeBackend, maybe_commit

__all__ = [
    "MODAL_AVAILABLE",
    "APP_NAME",
    "VOLUME_NAME",
    "VOLUME_MOUNT_PATH",
    "ARTIFACTS_PATH",
    "app",
    "image",
    "volume",
    "VolumeBackend",
    "maybe_commit",
]
