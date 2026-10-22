"""Tests for nemesis.ppo_learner on a tiny gym task (no CybORG needed)."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from nemesis.checkpoints import load_learned
from nemesis.ppo_learner import PPOLearner

PPO_KWARGS = dict(n_steps=64, batch_size=32, n_epochs=2, seed=0)


def make_env():
    return gym.make("CartPole-v1")


def drive(learner, env, timesteps, seed=0):
    """The loop a multi-agent harness runs: act, step, record, reset on episode end."""
    obs, _ = env.reset(seed=seed)
    updates = 0
    for _ in range(timesteps):
        action = learner.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        updates += learner.record(reward, obs, terminated, truncated)
        if terminated or truncated:
            obs, _ = env.reset()
    return updates


def weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}


def test_update_runs_once_per_full_rollout():
    env = make_env()
    learner = PPOLearner(env.observation_space, env.action_space, **PPO_KWARGS)
    before = weights(learner.model)

    assert drive(learner, env, 63) == 0
    assert all(torch.equal(before[k], v) for k, v in weights(learner.model).items())
    assert drive(learner, env, 1) == 1

    assert learner.num_timesteps == 64 and learner.num_updates == 1
    assert any(not torch.equal(before[k], v) for k, v in weights(learner.model).items())


def test_matches_sb3_learn_exactly():
    reference = PPO("MlpPolicy", make_env(), **PPO_KWARGS)
    reference.learn(128)

    env = make_env()
    learner = PPOLearner(env.observation_space, env.action_space, **PPO_KWARGS)
    drive(learner, env, 128)

    for name, value in weights(reference).items():
        torch.testing.assert_close(weights(learner.model)[name], value, msg=name)


def test_act_and_record_must_alternate():
    env = make_env()
    learner = PPOLearner(env.observation_space, env.action_space, **PPO_KWARGS)
    obs, _ = env.reset(seed=0)
    with pytest.raises(RuntimeError):
        learner.record(1.0, obs, False)
    learner.act(obs)
    with pytest.raises(RuntimeError):
        learner.act(obs)


def test_checkpoint_round_trip_and_resume(tmp_path):
    env = make_env()
    learner = PPOLearner(env.observation_space, env.action_space, **PPO_KWARGS)
    drive(learner, env, 128)
    learner.save(tmp_path, extra={"agent": "blue_agent_0"})

    resumed = PPOLearner.load(tmp_path)
    assert resumed.num_timesteps == 128
    for name, value in weights(learner.model).items():
        assert torch.equal(weights(resumed.model)[name], value), name

    drive(resumed, make_env(), 64)
    assert resumed.num_timesteps == 192 and resumed.num_updates == 1


def test_checkpoint_loads_as_plain_sb3_ppo(tmp_path):
    env = make_env()
    learner = PPOLearner(env.observation_space, env.action_space, **PPO_KWARGS)
    drive(learner, env, 64)
    learner.save(tmp_path)

    model = load_learned(tmp_path, env=make_env())
    assert isinstance(model, PPO)
    obs = np.zeros(4, dtype=np.float32)
    assert model.predict(obs, deterministic=True)[0] == learner.predict(obs)
    model.learn(64, reset_num_timesteps=False)  # keeps training under PPO.learn
    assert model.num_timesteps == 128


def test_learns_cartpole():
    env = make_env()
    learner = PPOLearner(env.observation_space, env.action_space, n_steps=512, batch_size=64, seed=0)
    drive(learner, env, 20_000)
    lengths = [e["l"] for e in learner.model.ep_info_buffer]
    assert np.mean(lengths[-20:]) > 100  # random play averages ~22 steps
