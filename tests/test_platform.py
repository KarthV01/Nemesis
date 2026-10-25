"""Tests for nemesis.platform contracts (no CybORG needed)."""

import io

import pytest

from nemesis.platform import (
    ArtifactStore,
    EpisodeMetrics,
    Event,
    EventKind,
    EventLog,
    MetricsLog,
    Side,
)
from nemesis.platform.metrics import summarize
from nemesis.platform.storage import LocalBackend


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
def test_checkpoints_autoincrement_and_list(tmp_path):
    store = ArtifactStore(tmp_path)
    c1 = store.new_checkpoint(Side.BLUE, "run_a")
    c2 = store.new_checkpoint(Side.BLUE, "run_a")
    assert c1.name == "checkpoint_000001"
    assert c2.name == "checkpoint_000002"
    assert [c.name for c in store.list_checkpoints(Side.BLUE, "run_a")] == [
        "checkpoint_000001",
        "checkpoint_000002",
    ]
    assert store.latest_checkpoint(Side.BLUE, "run_a").name == "checkpoint_000002"
    assert store.latest_checkpoint(Side.RED, "nope") is None


def test_generations_autoincrement(tmp_path):
    store = ArtifactStore(tmp_path)
    assert store.new_generation().name == "generation_000001"
    assert store.new_generation().name == "generation_000002"
    assert store.new_generation(10).name == "generation_000010"


def test_promote_best_copies_dir(tmp_path):
    store = ArtifactStore(tmp_path)
    ckpt = store.new_checkpoint(Side.RED, "run_x")
    with store.backend.open(store._join(ckpt.rel_path, "policy.pt"), "w") as fh:
        fh.write("weights")
    best = store.promote_best(ckpt, Side.RED)
    import os

    assert os.path.exists(os.path.join(best.local_dir(), "policy.pt"))


def test_experiment_jsonl_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    with store.open_experiment_file("exp1", "metrics.jsonl", "w") as fh:
        log = MetricsLog(fh)
        log.write(EpisodeMetrics(episode=1, steps=500, blue_reward=-3.0, blue_win=False))
        log.write(EpisodeMetrics(episode=2, steps=500, blue_reward=1.5, blue_win=True))
    with store.open_experiment_file("exp1", "metrics.jsonl", "r") as fh:
        rows = MetricsLog.read(fh)
    assert [r.episode for r in rows] == [1, 2]
    assert rows[1].blue_win is True


def test_local_backend_rejects_escape(tmp_path):
    backend = LocalBackend(tmp_path)
    with pytest.raises(ValueError):
        backend.open("../escape.txt", "w")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_summarize_skips_none_fields():
    batch = [
        EpisodeMetrics(episode=1, steps=10, blue_reward=2.0, attack_success_rate=0.5, blue_win=True),
        EpisodeMetrics(episode=2, steps=10, blue_reward=4.0, attack_success_rate=None, blue_win=False),
    ]
    s = summarize(batch)
    assert s.episodes == 2
    assert s.blue_reward == 3.0
    assert s.attack_success_rate == 0.5  # the None was skipped
    assert s.blue_win_rate == 0.5


def test_episode_metrics_extra_roundtrip():
    m = EpisodeMetrics(episode=1, steps=5, blue_reward=0.0, extra={"opponent": "gen3"})
    back = EpisodeMetrics.from_dict(__import__("json").loads(m.to_json()))
    assert back.extra["opponent"] == "gen3"


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def test_event_log_assigns_ids_and_serializes_like_live():
    buf = io.StringIO()
    log = EventLog(buf)
    e = log.log(EventKind.RED, "Exploit -> host_6", episode=4, host="host_6")
    assert e.id == 0 and e.t is not None
    d = e.to_dict()
    # Matches the dict shape nemesis.live already sends to the browser.
    assert d["kind"] == "red" and d["text"] == "Exploit -> host_6"
    assert d["episode"] == 4 and d["host"] == "host_6"


def test_event_log_read_roundtrip():
    buf = io.StringIO()
    log = EventLog(buf)
    log.log(EventKind.SYSTEM, "Episode 1 started", episode=1)
    log.log(EventKind.REWARD, "blue +0.8", red=-0.6)
    buf.seek(0)
    events = EventLog.read(buf)
    assert [str(e.kind) for e in events] == ["system", "reward"]
    assert events[1].extra["red"] == -0.6
