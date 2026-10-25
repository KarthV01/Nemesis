"""A ``nemesis.platform.storage.Backend`` backed by a mounted Modal Volume.

A Modal Volume is a *real filesystem* inside the running container: it is mounted
at a path (e.g. ``/vol``) and reads/writes under it are ordinary ``os`` calls.
That makes :class:`VolumeBackend` almost identical to
:class:`~nemesis.platform.storage.LocalBackend` — it subclasses it to inherit the
path-join/escape-guard logic and all five protocol methods, and therefore
satisfies the *same* ``Backend`` protocol and passes the *same* behavioural tests
against a local temp dir.

The one real difference is durability. Writes to the mount are immediately
visible to *this* container, but only become durable and visible to other
containers / later runs when the Volume is **committed**:

* :meth:`commit` persists everything written so far. Modal auto-commits on
  container exit, but we commit explicitly right after writing a checkpoint so a
  crash or a timeout never loses the last saved one.
* :meth:`reload` pulls in changes other containers have committed.

Because the mount is a real fs, :meth:`local_path` returns the real path (so
SB3's ``model.save``, which writes several files into a directory, works
directly — no temp-dir staging). With **no** Modal Volume object attached
(``volume=None``, e.g. a local temp dir used for tests or ``--local`` runs) both
:meth:`commit` and :meth:`reload` are deliberate no-ops: the local filesystem is
already durable, so the same training code runs unchanged off-Modal.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from nemesis.platform.storage import LocalBackend, PathLike


class VolumeBackend(LocalBackend):
    """Store backend writing to a mounted Modal Volume path.

    Parameters
    ----------
    root : path
        A directory on the mounted Volume, e.g. ``/vol/artifacts``. (For tests
        and ``--local`` runs this is just a temp dir — the behaviour is
        identical.)
    volume : modal.Volume, optional
        The Modal Volume object, used only for :meth:`commit` / :meth:`reload`.
        Omit it (``None``) off-Modal; both methods then no-op.
    """

    def __init__(self, root: PathLike, volume: Optional[Any] = None):
        super().__init__(root)
        self.volume = volume
        os.makedirs(self.root, exist_ok=True)

    def commit(self) -> None:
        """Persist writes so far to the Volume (durable + visible elsewhere).

        No-op when no Volume is attached — a plain local dir is already durable.
        """
        if self.volume is not None:
            self.volume.commit()

    def reload(self) -> None:
        """Pull in changes committed by other containers. No-op off-Modal."""
        if self.volume is not None:
            self.volume.reload()


def maybe_commit(backend: Any) -> None:
    """Commit ``backend`` if it is a Volume-style backend, else do nothing.

    Lets training code persist after each checkpoint without caring whether it
    was handed a :class:`VolumeBackend` or a plain ``LocalBackend``.
    """
    commit = getattr(backend, "commit", None)
    if callable(commit):
        commit()


__all__ = ["VolumeBackend", "maybe_commit"]
