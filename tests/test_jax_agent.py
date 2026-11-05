"""JAX PPO tests; no CybORG installation is required."""

import gymnasium as gym
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nemesis.jax import JAXPPOAgent, JAXPPOConfig


def make_agent(rollout_steps=8):
    return JAXPPOAgent(
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        gym.spaces.Discrete(2),
        JAXPPOConfig(rollout_steps=rollout_steps, minibatch_size=4, epochs=1, seed=7),
    )


def test_act_record_updates_on_full_rollout():
    agent = make_agent()
    observation = np.zeros(4, dtype=np.float32)
    before = {name: np.asarray(value).copy() for name, value in agent.params.items()}
    for step in range(8):
        assert agent.act(observation) in (0, 1)
        updated = agent.record(1.0, observation, False)
        assert updated is (step == 7)
    assert agent.num_timesteps == 8
    assert agent.num_updates == 1
    assert any(not np.array_equal(before[name], np.asarray(value)) for name, value in agent.params.items())


def test_act_and_record_alternate():
    agent = make_agent()
    observation = np.zeros(4, dtype=np.float32)
    with pytest.raises(RuntimeError):
        agent.record(0.0, observation, False)
    agent.act(observation)
    with pytest.raises(RuntimeError):
        agent.act(observation)


def test_checkpoint_round_trip(tmp_path):
    agent = make_agent(rollout_steps=2)
    observation = np.ones(4, dtype=np.float32)
    for _ in range(2):
        agent.act(observation)
        agent.record(1.0, observation, False)
    agent.save(tmp_path)
    restored = JAXPPOAgent.load(tmp_path)
    assert restored.num_timesteps == 2
    assert restored.num_updates == 1
    assert restored.predict(observation) == agent.predict(observation)

