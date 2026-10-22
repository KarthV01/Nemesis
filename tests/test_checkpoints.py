"""Round-trip tests for nemesis.checkpoints on a tiny gym task (no CybORG needed)."""

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nemesis.checkpoints import (
    LATEST_FILE,
    LearnedCheckpointCallback,
    list_checkpoints,
    load_learned,
    read_meta,
    save_learned,
)


def make_env():
    return gym.make("CartPole-v1")


def tiny_ppo(env):
    return PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0)


def assert_same_weights(a, b):
    for (name, p), (_, q) in zip(a.policy.state_dict().items(), b.policy.state_dict().items()):
        assert torch.equal(p.cpu(), q.cpu()), name


def test_ppo_round_trip_keeps_weights_and_timesteps(tmp_path):
    model = tiny_ppo(make_env())
    model.learn(128)

    path = save_learned(model, tmp_path, extra={"agent": "blue_agent_0"})
    restored = load_learned(tmp_path, env=make_env())

    assert path.name == "step_000000128"
    assert isinstance(restored, PPO)
    assert restored.num_timesteps == 128
    assert_same_weights(model, restored)
    assert read_meta(tmp_path)["extra"] == {"agent": "blue_agent_0"}
    obs = np.zeros(4, dtype=np.float32)
    assert model.predict(obs, deterministic=True)[0] == restored.predict(obs, deterministic=True)[0]


def test_off_policy_replay_buffer_is_restored(tmp_path):
    model = DQN("MlpPolicy", make_env(), learning_starts=10, buffer_size=1_000, seed=0)
    model.learn(100)

    save_learned(model, tmp_path)
    restored = load_learned(tmp_path, env=make_env())

    assert read_meta(tmp_path)["has_replay_buffer"]
    assert restored.replay_buffer.size() == model.replay_buffer.size() > 0
    np.testing.assert_array_equal(
        restored.replay_buffer.observations[:10], model.replay_buffer.observations[:10]
    )


def test_vec_normalize_stats_are_restored(tmp_path):
    env = VecNormalize(DummyVecEnv([make_env]))
    model = tiny_ppo(env)
    model.learn(128)

    save_learned(model, tmp_path)
    restored = load_learned(tmp_path, env=make_env(), training=False)

    restored_norm = restored.get_vec_normalize_env()
    assert restored_norm is not None and not restored_norm.training
    np.testing.assert_allclose(restored_norm.obs_rms.mean, env.obs_rms.mean)
    np.testing.assert_allclose(restored_norm.obs_rms.var, env.obs_rms.var)


def test_callback_saves_periodically_and_prunes(tmp_path):
    model = tiny_ppo(make_env())
    model.learn(320, callback=LearnedCheckpointCallback(tmp_path, save_freq=64, keep_last=2))

    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == ["step_000000256", "step_000000320"]
    assert json.loads((tmp_path / LATEST_FILE).read_text()) == {"checkpoint": "step_000000320"}
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_continues_timestep_count(tmp_path):
    model = tiny_ppo(make_env())
    model.learn(128, callback=LearnedCheckpointCallback(tmp_path, save_freq=1_000))

    resumed = load_learned(tmp_path, env=make_env())
    resumed.learn(128, reset_num_timesteps=False, callback=LearnedCheckpointCallback(tmp_path, save_freq=1_000))

    assert read_meta(tmp_path)["num_timesteps"] == 256
    assert [p.name for p in list_checkpoints(tmp_path)][-1] == "step_000000256"


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_learned(tmp_path)
