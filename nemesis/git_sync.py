"""Commit training outputs to a git branch and push it — without touching the checkout.

Training runs for hours in a working tree other sessions may also be using, so
switching branches or staging files there would trample their work. Instead,
``commit_paths_to_branch`` builds the commit with git plumbing against a
throwaway index file:

    parent = tip of <branch> (local or remote), else HEAD
    tree   = parent's tree with <paths> replaced by what is on disk
    commit = that tree on top of parent; refs/heads/<branch> advances to it

The current branch, the real index and the working files are never modified,
so it is safe to call from a training loop. The target branch must not be
checked out anywhere (its worktree would silently fall out of sync), and
``main``/``master`` are refused outright.

Typical use, from a training loop::

    sync = PeriodicGitSync(["weights"], branch="training/weights", every_minutes=30)
    ...
    sync.maybe_push("blue_agent_0 @ 40000 steps")   # no-op until 30 min have passed
    ...
    sync.push("final weights")                      # at the end, unconditionally
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

PathLike = Union[str, os.PathLike]

DEFAULT_BRANCH = "training/weights"
PROTECTED_BRANCHES = {"main", "master"}
GIT_TIMEOUT_S = 300


class GitSyncError(RuntimeError):
    pass


def commit_paths_to_branch(
    paths: Sequence[PathLike],
    branch: str = DEFAULT_BRANCH,
    message: str = "Update trained weights",
    *,
    repo: Optional[PathLike] = None,
    remote: Optional[str] = "origin",
) -> Optional[str]:
    """Commit the current contents of ``paths`` onto ``branch`` and push it.

    Parameters
    ----------
    paths : sequence of path
        Files or directories (inside the repo) to snapshot. Directories are
        taken whole, including deletions.
    branch : str
        Branch to commit onto; created from HEAD if it doesn't exist yet.
    message : str
        Commit message.
    repo : path, optional
        Any directory inside the repository; defaults to the current directory.
    remote : str or None
        Remote to push to; None commits locally only.

    Returns
    -------
    str or None
        The new commit's sha, or None when nothing changed since the branch tip.
    """
    _check_branch(branch)
    top = Path(_git(["rev-parse", "--show-toplevel"], cwd=repo or os.getcwd()))
    if branch in _checked_out_branches(top):
        raise GitSyncError(
            f"{branch!r} is checked out in a worktree; push weights to a branch nobody has checked out"
        )

    rel_paths = [_relative_to(top, p) for p in paths]
    local_ref = f"refs/heads/{branch}"
    parent = _pick_parent(top, branch, remote)
    old_local = _rev_parse(top, local_ref)

    fd, index_file = tempfile.mkstemp(prefix="nemesis-sync-index-")
    os.close(fd)
    try:
        env = {"GIT_INDEX_FILE": index_file}
        _git(["read-tree", parent], cwd=top, env=env)
        # -f: commit the files even if a .gitignore pattern covers them.
        _git(["add", "-A", "-f", "--", *rel_paths], cwd=top, env=env)
        tree = _git(["write-tree"], cwd=top, env=env)
    finally:
        os.unlink(index_file)

    if tree == _git(["rev-parse", f"{parent}^{{tree}}"], cwd=top):
        # Nothing new on disk; still make sure the local branch matches the
        # parent we picked (e.g. it only existed on the remote so far).
        if old_local != parent:
            _git(["update-ref", local_ref, parent, old_local or ""], cwd=top)
        return None

    commit = _git(["commit-tree", tree, "-p", parent, "-m", message], cwd=top)
    # The old-value check makes this fail rather than clobber a concurrent update.
    _git(["update-ref", local_ref, commit, old_local or ""], cwd=top)
    if remote:
        _git(["push", "--quiet", remote, f"{local_ref}:{local_ref}"], cwd=top)
    return commit


class PeriodicGitSync:
    """Push ``paths`` to ``branch`` at most once every ``every_minutes``.

    Failures (no network, rejected push, ...) are reported and swallowed:
    losing one sync must never kill a long training run. The next sync will
    carry the same changes.
    """

    def __init__(
        self,
        paths: Sequence[PathLike],
        branch: str = DEFAULT_BRANCH,
        *,
        every_minutes: float = 30.0,
        remote: Optional[str] = "origin",
        repo: Optional[PathLike] = None,
    ):
        _check_branch(branch)
        self.paths = list(paths)
        self.branch = branch
        self.every_s = every_minutes * 60
        self.remote = remote
        self.repo = repo
        self._last_push = time.time()

    def maybe_push(self, message: str) -> Optional[str]:
        if time.time() - self._last_push < self.every_s:
            return None
        return self.push(message)

    def push(self, message: str) -> Optional[str]:
        self._last_push = time.time()
        try:
            commit = commit_paths_to_branch(
                self.paths, self.branch, message, repo=self.repo, remote=self.remote
            )
        except (GitSyncError, OSError, subprocess.TimeoutExpired) as exc:
            print(f"[git-sync] push to {self.branch} failed, will retry next time: {exc}", flush=True)
            return None
        where = f"{self.remote}/{self.branch}" if self.remote else self.branch
        if commit is None:
            print(f"[git-sync] {where}: weights unchanged, nothing to push", flush=True)
        else:
            print(f"[git-sync] {where} <- {commit[:10]} {message}", flush=True)
        return commit


def _check_branch(branch: str) -> None:
    if branch in PROTECTED_BRANCHES:
        raise GitSyncError(f"refusing to push training outputs to {branch!r}; use a feature branch")


def _pick_parent(top: Path, branch: str, remote: Optional[str]) -> str:
    """Newest of the local and remote branch tips, else HEAD for a new branch."""
    local = _rev_parse(top, f"refs/heads/{branch}")
    remote_tip = None
    if remote:
        # The branch may not exist on the remote yet; that's fine.
        try:
            _git(["fetch", "--quiet", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}"], cwd=top)
        except GitSyncError:
            pass
        remote_tip = _rev_parse(top, f"refs/remotes/{remote}/{branch}")

    if local and remote_tip and local != remote_tip:
        if _is_ancestor(top, local, remote_tip):
            return remote_tip
        if not _is_ancestor(top, remote_tip, local):
            raise GitSyncError(
                f"local and {remote}/{branch} have diverged; reconcile them by hand (never force-pushing)"
            )
    return local or remote_tip or _git(["rev-parse", "HEAD"], cwd=top)


def _checked_out_branches(top: Path) -> List[str]:
    out = _git(["worktree", "list", "--porcelain"], cwd=top)
    prefix = "branch refs/heads/"
    return [line[len(prefix):] for line in out.splitlines() if line.startswith(prefix)]


def _relative_to(top: Path, path: PathLike) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(top.resolve()).as_posix()
    except ValueError:
        raise GitSyncError(f"{path} is outside the repository at {top}") from None


def _rev_parse(top: Path, ref: str) -> Optional[str]:
    try:
        return _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=top)
    except GitSyncError:
        return None


def _is_ancestor(top: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=top, capture_output=True, timeout=GIT_TIMEOUT_S,
    )
    return result.returncode == 0


def _git(args: List[str], *, cwd: PathLike, env: Optional[dict] = None) -> str:
    # Never block a headless training run on a credential prompt.
    full_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})}
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=full_env, capture_output=True, text=True, timeout=GIT_TIMEOUT_S
    )
    if result.returncode != 0:
        raise GitSyncError(f"git {' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()
