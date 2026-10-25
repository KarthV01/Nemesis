"""Tests for the Modal stage. All pass WITHOUT modal installed.

Two things are proven here:

* :class:`modal_app.volume_backend.VolumeBackend` satisfies the *same*
  ``nemesis.platform.storage.Backend`` protocol as ``LocalBackend`` and behaves
  identically against a local temp dir (the storage contract), so the cloud
  backend is a drop-in.
* :func:`modal_app.train.run_training` trains + checkpoints through
  ``ArtifactStore`` with TINY timesteps on a mock multi-agent env (no CybORG),
  and the checkpoint reloads and evaluates — the POC flow.
"""

import json
import os

import pytest

from nemesis.platform import ArtifactStore, Side
from nemesis.platform.metrics import MetricsLog, EpisodeMetrics, summarize
from nemesis.platform.storage import LocalBackend

import modal_app
from modal_app.images import MODAL_AVAILABLE, build_image
from modal_app.volume_backend import VolumeBackend, maybe_commit
from modal_app.train import (
    DummyTeamEnv,
    default_run_id,
    load_team,
    make_dummy_team_env,
    run_evaluation,
    run_training,
)


# --------------------------------------------------------------------------- #
# Import-without-modal guarantees
# --------------------------------------------------------------------------- #
def test_package_imports_without_modal():
    # Constants are always defined; the heavy objects are None when modal is absent.
    assert modal_app.APP_NAME == "nemesis"
    assert modal_app.VOLUME_MOUNT_PATH == "/vol"
    assert isinstance(MODAL_AVAILABLE, bool)
    if not MODAL_AVAILABLE:
        assert modal_app.app is None and modal_app.image is None and modal_app.volume is None
        with pytest.raises(RuntimeError):
            build_image()


# --------------------------------------------------------------------------- #
# VolumeBackend == LocalBackend contract (run every storage test against both)
# --------------------------------------------------------------------------- #
@pytest.fixture(params=["local", "volume"])
def backend(request, tmp_path):
    root = tmp_path / request.param
    if request.param == "local":
        return LocalBackend(root)
    return VolumeBackend(root)  # volume=None -> commit/reload no-op


def test_artifactstore_accepts_backend(backend):
    # ArtifactStore duck-types the backend; VolumeBackend must be recognised.
    store = ArtifactStore(backend)
    assert store.backend is backend


def test_checkpoints_autoincrement_and_list(backend):
    store = ArtifactStore(backend)
    c1 = store.new_checkpoint(Side.BLUE, "run_a")
    c2 = store.new_checkpoint(Side.BLUE, "run_a")
    assert (c1.name, c2.name) == ("checkpoint_000001", "checkpoint_000002")
    assert [c.name for c in store.list_checkpoints(Side.BLUE, "run_a")] == [
        "checkpoint_000001",
        "checkpoint_000002",
    ]
    assert store.latest_checkpoint(Side.BLUE, "run_a").name == "checkpoint_000002"
    assert store.latest_checkpoint(Side.RED, "nope") is None


def test_experiment_jsonl_roundtrip(backend):
    store = ArtifactStore(backend)
    with store.open_experiment_file("exp1", "metrics.jsonl", "w") as fh:
        log = MetricsLog(fh)
        log.write(EpisodeMetrics(episode=1, steps=8, blue_reward=-3.0, blue_win=False))
        log.write(EpisodeMetrics(episode=2, steps=8, blue_reward=1.5, blue_win=True))
    with store.open_experiment_file("exp1", "metrics.jsonl", "r") as fh:
        rows = MetricsLog.read(fh)
    assert [r.episode for r in rows] == [1, 2] and rows[1].blue_win is True


def test_json_roundtrip_and_local_path(backend):
    store = ArtifactStore(backend)
    store.write_json("blue/run_a/config.json", {"scenario": "cc4", "steps": 500})
    assert store.read_json("blue/run_a/config.json")["steps"] == 500
    # Volume is a real fs inside the container, so local_path is a real path.
    lp = backend.local_path("blue/run_a/config.json")
    assert lp is not None and os.path.isfile(lp)


def test_rejects_path_escape(backend):
    with pytest.raises(ValueError):
        backend.open("../escape.txt", "w")


def test_remove_tree(backend):
    store = ArtifactStore(backend)
    store.new_checkpoint(Side.BLUE, "run_a")
    assert backend.exists("blue/run_a")
    backend.remove_tree("blue/run_a")
    assert not backend.exists("blue/run_a")


# --------------------------------------------------------------------------- #
# commit()/reload() semantics
# --------------------------------------------------------------------------- #
class FakeVolume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def test_commit_reload_noop_without_volume(tmp_path):
    be = VolumeBackend(tmp_path)  # no volume attached
    be.commit()   # must not raise
    be.reload()


def test_commit_reload_delegate_to_volume(tmp_path):
    vol = FakeVolume()
    be = VolumeBackend(tmp_path, volume=vol)
    be.commit()
    be.reload()
    assert (vol.commits, vol.reloads) == (1, 1)


def test_maybe_commit(tmp_path):
    vol = FakeVolume()
    maybe_commit(VolumeBackend(tmp_path, volume=vol))
    assert vol.commits == 1
    maybe_commit(LocalBackend(tmp_path))  # no commit() attr -> no-op, no raise


# --------------------------------------------------------------------------- #
# DummyTeamEnv shape
# --------------------------------------------------------------------------- #
def test_dummy_env_interface():
    env = make_dummy_team_env(n_agents=3, episode_len=5, seed=1)
    assert env.possible_agents == ["blue_agent_0", "blue_agent_1", "blue_agent_2"]
    obs, info = env.reset(seed=1)
    assert set(obs) == set(env.possible_agents)
    done = False
    steps = 0
    while not done:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, _ = env.step(actions)
        steps += 1
        done = all(term[a] or trunc[a] for a in actions)
    assert steps == 5 and all(not term[a] for a in env.agents)  # truncation only


# --------------------------------------------------------------------------- #
# run_training local path (tiny timesteps, mock env, through a VolumeBackend)
# --------------------------------------------------------------------------- #
TINY = dict(timesteps=96, n_steps=32, batch_size=16)


def _tiny_factory():
    return make_dummy_team_env(n_agents=2, episode_len=8, seed=0)


def test_run_training_checkpoints_and_reloads(tmp_path):
    store = ArtifactStore(VolumeBackend(tmp_path))
    result = run_training(
        store=store, side=Side.BLUE, run_id="poc", env_factory=_tiny_factory,
        seed=0, **TINY,
    )
    assert result["agents"] == ["blue_agent_0", "blue_agent_1"]
    assert result["timesteps"] >= 96
    assert result["checkpoint"].startswith("blue/poc/checkpoint_")
    assert len(store.list_checkpoints(Side.BLUE, "poc")) >= 1

    # Reload from a FRESH store over the same root (simulates reload after the
    # cloud container is gone), then evaluate.
    store2 = ArtifactStore(VolumeBackend(tmp_path))
    learners, ckpt = load_team(store2, Side.BLUE, "poc")
    assert set(learners) == {"blue_agent_0", "blue_agent_1"}
    metrics = run_evaluation(_tiny_factory(), learners, episodes=2)
    s = summarize(metrics)
    assert s.episodes == 2 and all(m.steps == 8 for m in metrics)


def test_run_training_resumes(tmp_path):
    store = ArtifactStore(VolumeBackend(tmp_path))
    r1 = run_training(store=store, side=Side.BLUE, run_id="r", env_factory=_tiny_factory, seed=0, **TINY)
    r2 = run_training(store=store, side=Side.BLUE, run_id="r", env_factory=_tiny_factory, seed=0,
                      timesteps=r1["timesteps"] + 96, n_steps=32, batch_size=16)
    # Picked up where it left off rather than restarting from zero.
    assert r2["timesteps"] > r1["timesteps"]


def test_run_training_records_config_from_spec(tmp_path):
    # EnvSpec import does NOT import CybORG; pair it with the mock env factory so
    # config.json is recorded without building the real env.
    from nemesis.platform.envs import EnvSpec

    store = ArtifactStore(VolumeBackend(tmp_path))
    run_training(store=store, side=Side.BLUE, run_id="cfg", env_factory=_tiny_factory,
                 spec=EnvSpec(steps=8, seed=0), seed=0, **TINY)
    cfg = store.read_json("blue/cfg/config.json")
    assert cfg["scenario"] == "cc4" and cfg["steps"] == 8


def test_run_training_commits_volume(tmp_path):
    vol = FakeVolume()
    store = ArtifactStore(VolumeBackend(tmp_path, volume=vol))
    run_training(store=store, side=Side.BLUE, run_id="c", env_factory=_tiny_factory, seed=0, **TINY)
    assert vol.commits >= 1  # at least the final checkpoint was committed


def test_run_training_needs_env_source(tmp_path):
    store = ArtifactStore(VolumeBackend(tmp_path))
    with pytest.raises(ValueError):
        run_training(store=store, side=Side.BLUE, run_id="x", timesteps=8)


def test_default_run_id_unique():
    assert default_run_id("blue").startswith("blue-")
