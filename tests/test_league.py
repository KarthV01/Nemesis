"""Tests for the self-play league.

Most tests are pure logic on an ``ArtifactStore`` + a fake policy and need no
CybORG. The single end-to-end smoke test (scripted red, tiny episode) is marked
``slow`` and skips automatically if CybORG/SB3 are not importable.
"""

from __future__ import annotations

import os
import random

import pytest

from nemesis.platform.metrics import EpisodeMetrics
from nemesis.platform.storage import ArtifactStore, Side
from nemesis.league.league import League, Opponent, Standing


# --------------------------------------------------------------------------- #
# A fake policy: proves the league never needs a real learner.
# --------------------------------------------------------------------------- #
class FakePolicy:
    """Stand-in for a learner: just a tag and a counter we can checkpoint."""

    def __init__(self, tag: str):
        self.tag = tag
        self.steps = 0

    def save(self, path: str) -> None:  # the store gives a dir; we drop a file
        import os

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "policy.txt"), "w") as fh:
            fh.write(f"{self.tag}:{self.steps}")


def _metrics(reward: float, win=None, red_reward=None, episode=0) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode=episode, steps=20, blue_reward=reward, blue_win=win, red_reward=red_reward
    )


# --------------------------------------------------------------------------- #
# Registration + manifests
# --------------------------------------------------------------------------- #
def test_register_scripted_allocates_generation_and_manifest(tmp_path):
    store = ArtifactStore(tmp_path)
    league = League(store, name="t")

    opp = league.register(Side.RED, "scripted", "FiniteStateRedAgent")
    assert opp.side == "red"
    assert opp.kind == "scripted"
    assert opp.ref == "FiniteStateRedAgent"
    assert opp.generation == 1
    assert opp.name == "red_gen_000001"
    # Manifest written into the generation dir and reads back.
    manifest = store.read_json(store._join(opp.gen_dir, "opponent.json"))
    assert manifest["name"] == "red_gen_000001"
    assert Opponent.from_dict(manifest) == opp


def test_register_learned_creates_weight_dir_and_checkpoint_handle(tmp_path):
    store = ArtifactStore(tmp_path)
    league = League(store, name="t")

    opp = league.register(Side.BLUE, "learned", meta={"generation": 1})
    assert opp.kind == "learned"
    assert opp.is_learned
    # Default ref is <gen_dir>/<side> and exists for the trainer to fill.
    assert opp.ref == store._join(opp.gen_dir, "blue")
    assert store.backend.exists(opp.ref)
    # A FakePolicy can checkpoint into it via the handle's local dir.
    pol = FakePolicy("blue-v1")
    pol.save(opp.checkpoint(store).local_dir())
    import os

    assert os.path.exists(os.path.join(opp.local_weights_dir(store), "policy.txt"))


def test_scripted_roster_is_idempotent(tmp_path):
    store = ArtifactStore(tmp_path)
    league = League(store, name="t")
    roster = ["FiniteStateRedAgent", "RedCaptain"]

    first = league.register_scripted_roster(Side.RED, roster)
    assert len(first) == 2
    second = league.register_scripted_roster(Side.RED, roster)
    assert second == []  # nothing new registered the second time
    assert len(league.opponents(Side.RED)) == 2


def test_state_persists_across_instances(tmp_path):
    store = ArtifactStore(tmp_path)
    league = League(store, name="t")
    league.register(Side.RED, "scripted", "FiniteStateRedAgent")
    league.register(Side.BLUE, "learned")

    # A fresh League over the same store sees the whole history.
    reloaded = League(ArtifactStore(tmp_path), name="t")
    assert len(reloaded) == 2
    assert {o.name for o in reloaded.opponents(Side.RED)} == {"red_gen_000001"}
    assert reloaded.latest(Side.BLUE).name == "blue_gen_000002"


# --------------------------------------------------------------------------- #
# Opponent sampling
# --------------------------------------------------------------------------- #
def test_sample_returns_none_for_empty_side(tmp_path):
    league = League(ArtifactStore(tmp_path), name="t")
    assert league.sample(Side.RED) is None
    assert league.sample_eval_set(Side.RED, 3) == []


def test_sample_single_opponent_is_deterministic(tmp_path):
    league = League(ArtifactStore(tmp_path), name="t")
    only = league.register(Side.RED, "scripted", "FiniteStateRedAgent")
    for _ in range(10):
        assert league.sample(Side.RED, latest_prob=0.0).name == only.name


def test_sample_latest_prob_biases_to_newest(tmp_path):
    league = League(ArtifactStore(tmp_path), name="t")
    for cls in ["A", "B", "C", "D"]:
        league.register(Side.RED, "scripted", cls)
    latest = league.latest(Side.RED)
    rng = random.Random(0)

    # latest_prob=1.0 always yields the newest snapshot.
    assert all(league.sample(Side.RED, latest_prob=1.0, rng=rng).name == latest.name for _ in range(20))

    # latest_prob=0.0 draws uniformly over history: other opponents appear.
    names = {league.sample(Side.RED, latest_prob=0.0, rng=rng).name for _ in range(200)}
    assert len(names) > 1


def test_sample_eval_set_includes_latest_and_is_distinct(tmp_path):
    league = League(ArtifactStore(tmp_path), name="t")
    for cls in ["A", "B", "C", "D"]:
        league.register(Side.RED, "scripted", cls)
    latest = league.latest(Side.RED)

    s = league.sample_eval_set(Side.RED, 3, rng=random.Random(1))
    assert len(s) == 3
    assert latest.name in {o.name for o in s}
    assert len({o.name for o in s}) == 3  # no duplicates

    # Asking for more than exist returns the whole pool, still distinct.
    allset = league.sample_eval_set(Side.RED, 99, rng=random.Random(1))
    assert len({o.name for o in allset}) == 4


def test_sample_only_considers_requested_side(tmp_path):
    league = League(ArtifactStore(tmp_path), name="t")
    league.register(Side.RED, "scripted", "FiniteStateRedAgent")
    league.register(Side.BLUE, "learned")
    assert league.sample(Side.RED).side == "red"
    assert league.sample(Side.BLUE).side == "blue"


# --------------------------------------------------------------------------- #
# Win / score table from EpisodeMetrics
# --------------------------------------------------------------------------- #
def test_record_result_builds_win_score_table(tmp_path):
    store = ArtifactStore(tmp_path)
    league = League(store, name="t")
    opp = league.register(Side.RED, "scripted", "FiniteStateRedAgent")

    league.record_result(opp, _metrics(-3.0, win=False, red_reward=3.0))
    league.record_result(opp, _metrics(-1.0, win=True, red_reward=1.0))
    league.record_result(opp, _metrics(1.0, win=True, red_reward=-1.0))

    st = league.standing(opp)
    assert st.games == 3
    assert st.wins == 2
    assert st.win_labeled_games == 3
    assert st.win_rate == pytest.approx(2 / 3)
    assert st.mean_blue_reward == pytest.approx((-3.0 - 1.0 + 1.0) / 3)
    assert st.mean_red_reward == pytest.approx((3.0 + 1.0 - 1.0) / 3)


def test_record_result_tolerates_unlabeled_wins(tmp_path):
    league = League(ArtifactStore(tmp_path), name="t")
    opp = league.register(Side.RED, "scripted", "FiniteStateRedAgent")
    league.record_result(opp, _metrics(-2.0))  # blue_win=None, red_reward=None
    st = league.standing(opp)
    assert st.games == 1
    assert st.win_labeled_games == 0
    assert st.win_rate is None
    assert st.mean_red_reward is None
    assert st.mean_blue_reward == pytest.approx(-2.0)


def test_standings_sorted_hardest_first(tmp_path):
    league = League(ArtifactStore(tmp_path), name="t")
    easy = league.register(Side.RED, "scripted", "Easy")
    hard = league.register(Side.RED, "scripted", "Hard")
    league.record_result(easy, _metrics(5.0))
    league.record_result(hard, _metrics(-10.0))
    rows = league.standings(side="red")
    assert rows[0].opponent == hard.name  # lowest blue reward (hardest) first
    assert rows[-1].opponent == easy.name


def test_standings_persist_across_instances(tmp_path):
    store = ArtifactStore(tmp_path)
    league = League(store, name="t")
    opp = league.register(Side.RED, "scripted", "FiniteStateRedAgent")
    league.record_result(opp, _metrics(-3.0, win=False))

    reloaded = League(ArtifactStore(tmp_path), name="t")
    st = reloaded.standing(opp.name)
    assert st is not None and st.games == 1 and st.wins == 0


# --------------------------------------------------------------------------- #
# End-to-end smoke test with scripted red (CybORG + SB3). Skips if slow/missing.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_selfplay_smoke_scripted_red(tmp_path):
    pytest.importorskip("CybORG")
    pytest.importorskip("stable_baselines3")
    from nemesis.league.selfplay_loop import SelfPlayConfig, run_selfplay

    config = SelfPlayConfig.smoke(
        name="smoke",
        artifacts_root=str(tmp_path),
        episode_steps=20,
        blue_steps_per_gen=40,
        n_steps=32,
        batch_size=16,
        eval_episodes=1,
        eval_opponents=2,
        scripted_red_roster=("FiniteStateRedAgent",),
        seed=0,
    )
    store = ArtifactStore(tmp_path)
    league = run_selfplay(config, store=store)

    # A blue snapshot was frozen and the red roster registered.
    assert league.latest(Side.BLUE) is not None
    assert len(league.opponents(Side.RED)) == 1
    # best_policy promoted and experiment logs written.
    import os

    assert store.backend.exists(store._join(str(Side.BLUE), "best_policy"))
    exp = store.experiment_dir("smoke")
    assert store.backend.exists(store._join(exp, "evaluations.jsonl"))
    assert store.backend.exists(store._join(exp, "events.jsonl"))
    # The score table recorded the evaluation episode(s).
    assert any(s.games > 0 for s in league.standings(side="red"))
