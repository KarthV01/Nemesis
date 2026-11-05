# JAX agents

Nemesis supports reinforcement-learning policies implemented directly in JAX.
These policies consume numeric CybORG observations and select actions from the
environment's legal discrete action space; they do not load or call language
models.

## Install

Install the project normally, then add the JAX CPU runtime:

```bash
uv pip install --python .venv/bin/python -r requirements-jax.txt
```

On Windows, use `.venv\Scripts\python.exe` for the Python path. Accelerator
builds should follow the official JAX installation matrix instead of the CPU
extra pinned in `requirements-jax.txt`.

## Step-at-a-time API

`JAXPPOAgent` is designed for Nemesis' shared multi-agent loop:

```python
from nemesis.jax import JAXPPOAgent

agent = JAXPPOAgent(env.observation_space(agent_id), env.action_space(agent_id))
action = agent.act(observation)
next_observation, reward, terminated, truncated = step(action)
updated = agent.record(reward, next_observation, terminated, truncated)
```

Each agent slot owns an independent policy and optimizer. `record` returns
`True` when its rollout fills and a PPO update completes. Use `predict` for
evaluation, and `save`/`load` for portable JAX checkpoints.

