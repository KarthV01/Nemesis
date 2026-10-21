# Nemesis
Adversarial multi-agent reinforcement learning framework where autonomous cybersecurity agents evolve through offensive and defensive self-play.

Built on [CybORG](https://github.com/cage-challenge/cage-challenge-4) (CAGE Challenge 4), vendored as a git submodule in `external/cage-challenge-4`.

## Setup (Windows)

Prerequisites: `git` and [`uv`](https://docs.astral.sh/uv/) (uv downloads Python 3.10 for you).

```powershell
git clone --recurse-submodules https://github.com/KarthV01/Nemesis
cd Nemesis
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates `.venv` (Python 3.10, required because CC4 pins numpy 1.26 / torch 2.2), installs CybORG and its
RL dependencies (RLlib, Stable-Baselines3, PettingZoo, PyTorch), and runs a smoke-test episode.

## Setup (Linux)

```bash
git clone --recurse-submodules https://github.com/KarthV01/Nemesis
cd Nemesis
bash setup.sh   # installs uv if missing; CPU-only PyTorch unless TORCH_CUDA=1
```

## Cloud training on Docker Agentic Platform

See [docs/docker-sandbox.md](docs/docker-sandbox.md) to run time-budgeted training in a Docker sandbox
(`scripts/sandbox_train.sh`, which pushes results to a `runs/<id>` branch) and Claude Code on Sonnet 5.5
with your Claude subscription (`scripts/sandbox_claude.sh`).

## Usage

```powershell
# One episode with random blue agents vs. the FiniteStateRedAgent
.venv\Scripts\python.exe scripts\smoke_test.py

# Train independent PPO policies for the 5 blue agents (RLlib); checkpoints go to checkpoints/ppo_cc4
.venv\Scripts\python.exe scripts\train_ppo_rllib.py --iterations 50 --workers 4

# CybORG's own test suite (slow)
cd external\cage-challenge-4; ..\..\.venv\Scripts\python.exe -m pytest CybORG\Tests\test_cc4
```

## Environment notes

- Scenario: `EnterpriseScenarioGenerator`, with 5 blue agents (`blue_agent_0..3` have obs 92 / 82 actions;
  `blue_agent_4` has obs 210 / 242 actions), green users, and a red attacker.
- Wrappers: `BlueFlatWrapper` (PettingZoo-style, flat vectors) and `EnterpriseMAE` (RLlib `MultiAgentEnv`).
- A step takes ~50 ms, so add rollout workers (`--workers`) to speed up training.
- Official examples are in `external/cage-challenge-4/CybORG/Evaluation/training_example/`.
