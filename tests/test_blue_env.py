"""SingleBlueEnv drives one CC4 blue agent and checkpoints through nemesis.checkpoints."""

from stable_baselines3 import PPO

from nemesis.blue_env import SingleBlueEnv
from nemesis.checkpoints import load_learned, read_meta, save_learned


def test_episode_runs_and_ends_on_truncation():
    env = SingleBlueEnv("blue_agent_4", steps=5)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert len(env.action_labels()) == env.action_space.n

    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float) and not terminated
        if truncated:
            break
    assert truncated


def test_ppo_on_cyborg_round_trips(tmp_path):
    env = SingleBlueEnv(steps=10)
    model = PPO("MlpPolicy", env, n_steps=20, batch_size=10, n_epochs=1, seed=0)
    model.learn(20)

    save_learned(model, tmp_path, extra={"agent": env.agent})
    restored = load_learned(tmp_path, env=SingleBlueEnv(steps=10))

    assert restored.num_timesteps == 20
    assert read_meta(tmp_path)["recent_episode_reward_mean"] is not None
    obs, _ = env.reset(seed=0)
    assert model.predict(obs, deterministic=True)[0] == restored.predict(obs, deterministic=True)[0]
