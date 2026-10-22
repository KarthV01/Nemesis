"""Tests for nemesis.git_sync against a throwaway repo and bare remote."""

import subprocess

import pytest

from nemesis.git_sync import GitSyncError, PeriodicGitSync, commit_paths_to_branch


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    git(tmp_path, "init", "--bare", "-q", str(remote))
    git(tmp_path, "init", "-q", "-b", "main", str(work))
    git(work, "config", "user.email", "t@example.com")
    git(work, "config", "user.name", "t")
    (work / "README.md").write_text("hi\n")
    git(work, "add", "README.md")
    git(work, "commit", "-q", "-m", "init")
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-q", "origin", "main")
    return work, remote


def test_commits_and_pushes_without_touching_checkout(repo):
    work, remote = repo
    (work / "weights" / "a").mkdir(parents=True)
    (work / "weights" / "a" / "policy.pt").write_bytes(b"v1")
    (work / "staged.txt").write_text("user's work in progress\n")
    git(work, "add", "staged.txt")

    commit = commit_paths_to_branch([work / "weights"], "training/weights", "w1", repo=work)

    assert git(remote, "rev-parse", "training/weights") == commit
    assert git(remote, "show", "training/weights:weights/a/policy.pt") == "v1"
    assert git(remote, "show", "training/weights:README.md") == "hi"
    # The checkout, its index and the user's staged file are untouched.
    assert git(work, "branch", "--show-current") == "main"
    assert git(work, "diff", "--cached", "--name-only") == "staged.txt"
    assert "staged.txt" not in git(remote, "ls-tree", "-r", "--name-only", "training/weights")


def test_second_push_builds_on_the_first_and_skips_no_ops(repo):
    work, remote = repo
    (work / "weights").mkdir()
    (work / "weights" / "policy.pt").write_bytes(b"v1")
    first = commit_paths_to_branch([work / "weights"], "training/weights", "w1", repo=work)
    assert commit_paths_to_branch([work / "weights"], "training/weights", "same", repo=work) is None

    (work / "weights" / "policy.pt").write_bytes(b"v2")
    second = commit_paths_to_branch([work / "weights"], "training/weights", "w2", repo=work)
    assert git(work, "rev-parse", f"{second}^") == first
    assert git(remote, "show", "training/weights:weights/policy.pt") == "v2"


def test_ignored_weights_are_still_committed(repo):
    work, remote = repo
    (work / ".gitignore").write_text("weights/\n")
    (work / "weights").mkdir()
    (work / "weights" / "policy.pt").write_bytes(b"v1")
    commit_paths_to_branch([work / "weights"], "training/weights", "w1", repo=work)
    assert git(remote, "show", "training/weights:weights/policy.pt") == "v1"


def test_refuses_main_and_checked_out_branches(repo):
    work, _ = repo
    with pytest.raises(GitSyncError):
        commit_paths_to_branch([work], "main", "x", repo=work)
    git(work, "checkout", "-q", "-b", "feature/x")
    with pytest.raises(GitSyncError):
        commit_paths_to_branch([work], "feature/x", "x", repo=work)


def test_periodic_sync_waits_for_interval_and_swallows_failures(repo, capsys):
    work, remote = repo
    (work / "weights").mkdir()
    (work / "weights" / "policy.pt").write_bytes(b"v1")

    sync = PeriodicGitSync([work / "weights"], "training/weights", every_minutes=60, repo=work)
    assert sync.maybe_push("too soon") is None
    assert sync.push("now") is not None

    git(work, "remote", "set-url", "origin", str(work.parent / "missing.git"))
    (work / "weights" / "policy.pt").write_bytes(b"v2")
    assert sync.push("offline") is None
    assert "failed" in capsys.readouterr().out
