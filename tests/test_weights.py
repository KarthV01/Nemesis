"""Tests for nemesis.weights on a tiny gym task (no CybORG needed)."""

import json

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nemesis.weights import (
    META_FILE,
    POLICY_FILE,
    export_sb3_weights,
    load_policy_weights,
    load_sb3_weights,
    save_policy_weights,
)


def tiny_ppo(env):
    return PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0)


def test_sb3_export_loads_into_a_fresh_model(tmp_path):
    model = tiny_ppo(gym.make("CartPole-v1"))
    model.learn(128)

    path = export_sb3_weights(model, "blue_agent_0", root=tmp_path, extra={"note": "x"})
    assert path == tmp_path / "sb3_ppo" / "blue_agent_0" / POLICY_FILE

    fresh = tiny_ppo(gym.make("CartPole-v1"))
    payload = load_sb3_weights(fresh, path.parent)
    for (name, p), (_, q) in zip(model.policy.state_dict().items(), fresh.policy.state_dict().items()):
        assert torch.equal(p.cpu(), q.cpu()), name

    obs = np.zeros(4, dtype=np.float32)
    assert model.predict(obs, deterministic=True)[0] == fresh.predict(obs, deterministic=True)[0]
    assert payload["meta"]["num_timesteps"] == model.num_timesteps
    assert payload["obs_rms"] is None


def test_meta_json_describes_the_file(tmp_path):
    model = tiny_ppo(gym.make("CartPole-v1"))
    path = export_sb3_weights(model, "blue_agent_3", root=tmp_path)
    meta = json.loads((path.parent / META_FILE).read_text())
    assert meta["agent"] == "blue_agent_3"
    assert meta["trainer"] == "sb3_ppo"
    assert meta["parameters"] == sum(p.numel() for p in model.policy.state_dict().values())
    assert len(meta["sha256"]) == 64


def test_obs_normalisation_stats_are_exported(tmp_path):
    env = VecNormalize(DummyVecEnv([lambda: gym.make("CartPole-v1")]), norm_obs=True)
    model = tiny_ppo(env)
    model.learn(64)
    payload = load_policy_weights(export_sb3_weights(model, "a", root=tmp_path))
    assert torch.allclose(payload["obs_rms"]["mean"], torch.from_numpy(env.obs_rms.mean))


def test_numpy_state_dict_is_stored_as_tensors(tmp_path):
    # RLlib's policy.get_weights() hands back numpy arrays.
    path = save_policy_weights({"w": np.ones((2, 3), dtype=np.float32)}, tmp_path)
    payload = torch.load(path, weights_only=True)
    assert torch.equal(payload["state_dict"]["w"], torch.ones(2, 3))


def test_re_export_overwrites_in_place(tmp_path):
    save_policy_weights({"w": torch.zeros(1)}, tmp_path)
    save_policy_weights({"w": torch.ones(1)}, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [META_FILE, POLICY_FILE]
    assert load_policy_weights(tmp_path)["state_dict"]["w"].item() == 1.0
