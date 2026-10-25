"""The Nemesis artifact store — where trained policies and experiment logs live.

Every training stage writes into one directory tree so a checkpoint produced by
one stage (say blue training) can be found and loaded by another (self-play,
evaluation) without the two agreeing on anything but this module::

    <root>/
        red/
            <run_id>/
                checkpoint_000001/      # a policy snapshot (opaque to this module)
                checkpoint_000002/
            best_policy/                # symlink-free copy of the chosen snapshot
        blue/
            <run_id>/ ...
            best_policy/
        selfplay/
            generation_000001/
            generation_000002/
        experiments/
            <experiment>/
                reward_history.jsonl
                evaluations.jsonl
                config.yaml

This module owns *paths and bookkeeping*, not file formats: a "checkpoint" is
just a directory a trainer fills however it likes (e.g. with
:mod:`nemesis.checkpoints`). That keeps the store agnostic to SB3 vs. RLlib vs.
LoRA adapters.

Backends
--------
``ArtifactStore`` does all its I/O through a small :class:`Backend` protocol
(mkdir / open / list / exists / ...). :class:`LocalBackend` is the default. A
Modal-Volume backend implements the same five-ish methods so the identical
training code persists to the cloud; see ``nemesis/platform/README.md``.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import IO, Iterable, List, Optional, Protocol, Union

PathLike = Union[str, os.PathLike]


class Side(str, Enum):
    """Which part of the tree a run belongs to."""

    RED = "red"
    BLUE = "blue"
    SELFPLAY = "selfplay"

    def __str__(self) -> str:  # so os.path.join etc. see "red", not "Side.RED"
        return self.value


# --------------------------------------------------------------------------- #
# Backend seam
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    """Minimal filesystem the store needs. Paths are POSIX-style, store-relative.

    A backend never interprets paths beyond joining them to its own root; all
    layout logic lives in :class:`ArtifactStore`. Implement this to persist
    somewhere other than local disk (e.g. a Modal Volume)."""

    def makedirs(self, path: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def listdir(self, path: str) -> List[str]: ...
    def open(self, path: str, mode: str) -> IO: ...
    def remove_tree(self, path: str) -> None: ...
    def local_path(self, path: str) -> Optional[str]:
        """A real local filesystem path for ``path`` if the backend has one.

        Trainers that must hand a plain directory to a library that only speaks
        ``os`` paths (SB3's ``model.save`` writes several files into a dir) use
        this. A remote backend returns ``None``; callers then stage into a temp
        dir and :meth:`ArtifactStore.put_dir` it back."""
        ...


class LocalBackend:
    """Default backend: a directory on the local disk."""

    def __init__(self, root: PathLike):
        self.root = os.path.abspath(str(root))

    def _abs(self, path: str) -> str:
        # path is store-relative and POSIX; keep it inside root.
        parts = [p for p in PurePosixPath(path).parts if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise ValueError(f"path escapes store root: {path!r}")
        return os.path.join(self.root, *parts)

    def makedirs(self, path: str) -> None:
        os.makedirs(self._abs(path), exist_ok=True)

    def exists(self, path: str) -> bool:
        return os.path.exists(self._abs(path))

    def listdir(self, path: str) -> List[str]:
        abs_path = self._abs(path)
        return sorted(os.listdir(abs_path)) if os.path.isdir(abs_path) else []

    def open(self, path: str, mode: str) -> IO:
        abs_path = self._abs(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        return open(abs_path, mode)

    def remove_tree(self, path: str) -> None:
        shutil.rmtree(self._abs(path), ignore_errors=True)

    def local_path(self, path: str) -> Optional[str]:
        return self._abs(path)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Checkpoint:
    """A handle to one checkpoint directory in the store."""

    store: "ArtifactStore"
    rel_path: str  # store-relative POSIX path to the checkpoint directory

    @property
    def name(self) -> str:
        return PurePosixPath(self.rel_path).name

    def local_dir(self) -> Optional[str]:
        return self.store.backend.local_path(self.rel_path)


class ArtifactStore:
    """Resolve and manage the Nemesis checkpoint / experiment layout.

    Parameters
    ----------
    root : path or Backend
        A local directory (wrapped in :class:`LocalBackend`) or any backend.
    """

    CHECKPOINT_PREFIX = "checkpoint_"
    GENERATION_PREFIX = "generation_"
    BEST = "best_policy"

    def __init__(self, root: Union[PathLike, Backend] = "artifacts"):
        self.backend: Backend = root if _is_backend(root) else LocalBackend(root)

    # -- path helpers ------------------------------------------------------- #
    @staticmethod
    def _join(*parts: str) -> str:
        return PurePosixPath(*[str(p) for p in parts]).as_posix()

    def run_dir(self, side: Union[Side, str], run_id: str) -> str:
        return self._join(str(side), run_id)

    def _seq_name(self, prefix: str, n: int) -> str:
        return f"{prefix}{n:06d}"

    # -- checkpoints -------------------------------------------------------- #
    def new_checkpoint(self, side: Union[Side, str], run_id: str) -> Checkpoint:
        """Allocate the next ``checkpoint_NNNNNN`` dir under ``<side>/<run_id>``."""
        run = self.run_dir(side, run_id)
        existing = [
            int(name[len(self.CHECKPOINT_PREFIX):])
            for name in self.backend.listdir(run)
            if name.startswith(self.CHECKPOINT_PREFIX)
            and name[len(self.CHECKPOINT_PREFIX):].isdigit()
        ]
        nxt = (max(existing) + 1) if existing else 1
        rel = self._join(run, self._seq_name(self.CHECKPOINT_PREFIX, nxt))
        self.backend.makedirs(rel)
        return Checkpoint(self, rel)

    def list_checkpoints(self, side: Union[Side, str], run_id: str) -> List[Checkpoint]:
        run = self.run_dir(side, run_id)
        return [
            Checkpoint(self, self._join(run, name))
            for name in self.backend.listdir(run)
            if name.startswith(self.CHECKPOINT_PREFIX)
        ]

    def latest_checkpoint(self, side: Union[Side, str], run_id: str) -> Optional[Checkpoint]:
        ckpts = self.list_checkpoints(side, run_id)
        return ckpts[-1] if ckpts else None

    def new_generation(self, n: Optional[int] = None) -> Checkpoint:
        """Allocate a self-play ``generation_NNNNNN`` dir (auto-incremented)."""
        base = str(Side.SELFPLAY)
        if n is None:
            existing = [
                int(name[len(self.GENERATION_PREFIX):])
                for name in self.backend.listdir(base)
                if name.startswith(self.GENERATION_PREFIX)
                and name[len(self.GENERATION_PREFIX):].isdigit()
            ]
            n = (max(existing) + 1) if existing else 1
        rel = self._join(base, self._seq_name(self.GENERATION_PREFIX, n))
        self.backend.makedirs(rel)
        return Checkpoint(self, rel)

    def best_policy(self, side: Union[Side, str]) -> Checkpoint:
        """The ``<side>/best_policy`` handle (created on demand by writers)."""
        rel = self._join(str(side), self.BEST)
        return Checkpoint(self, rel)

    def promote_best(self, checkpoint: Checkpoint, side: Union[Side, str]) -> Checkpoint:
        """Copy ``checkpoint`` to ``<side>/best_policy`` (local backends only).

        Remote backends should stage the copy themselves; this convenience uses
        real paths and raises if either side lacks one."""
        src = checkpoint.local_dir()
        dst_handle = self.best_policy(side)
        dst = dst_handle.local_dir()
        if src is None or dst is None:
            raise NotImplementedError(
                "promote_best needs a local backend; stage the copy manually for remote stores"
            )
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        return dst_handle

    # -- experiment logs ---------------------------------------------------- #
    def experiment_dir(self, name: str) -> str:
        rel = self._join("experiments", name)
        self.backend.makedirs(rel)
        return rel

    def experiment_path(self, name: str, filename: str) -> str:
        return self._join(self.experiment_dir(name), filename)

    def open_experiment_file(self, name: str, filename: str, mode: str = "a") -> IO:
        return self.backend.open(self.experiment_path(name, filename), mode)

    def write_json(self, rel_path: str, obj) -> None:
        with self.backend.open(rel_path, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)

    def read_json(self, rel_path: str):
        with self.backend.open(rel_path, "r") as fh:
            return json.load(fh)


def _is_backend(obj) -> bool:
    return all(hasattr(obj, m) for m in ("makedirs", "exists", "listdir", "open"))


__all__ = ["ArtifactStore", "Backend", "LocalBackend", "Checkpoint", "Side"]
