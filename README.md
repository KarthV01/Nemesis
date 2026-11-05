# Nemesis

> [*Νέμεσις*](https://www.theoi.com/Daimon/Nemesis.html) — the Greek goddess of divine retribution and rightful measure. In Hesiod's telling, she was a daughter of **Nyx**, the primordial goddess of night. She punished **hubris**: the arrogance of those who believed their power or good fortune placed them beyond consequence. Artists gave her a measuring rod to mark the limits, a bridle to restrain excess, and scales to weigh what was due. Her domain was the restoration of balance whenever pride, cruelty, or undeserved fortune disturbed it. She was the reckoning that arrived when confidence outran judgment.
> Nemesis gives a security policy an adversary that can expose the distance between appearing capable and surviving an attack.

A multi-agent reinforcement learning framework for cybersecurity, built around repeated attack-and-defense episodes in a simulated enterprise network. Nemesis supports PyTorch and JAX PPO policies; language-model agents are intentionally out of scope. The current training path trains five blue-team defenders against CybORG's scripted red attacker, saving policy checkpoints between runs. Offensive and defensive self-play is the broader direction.

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

## Local configuration and secrets

Copy `.env.example` to `.env` and replace only the values needed by your setup.
`.env` and credential files are intentionally ignored by Git; never commit real
webhook keys, API tokens, private keys, or passwords.

Nemesis reads configuration from environment variables. Load `.env` into the
current shell before starting a component that needs it:

```powershell
# Windows PowerShell
Get-Content .env | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    Set-Item -Path "Env:$name" -Value $value
}
```

```bash
# Linux/macOS
set -a
source .env
set +a
```

Modal CLI authentication should be configured with Modal's own setup
command/account secret store. The `MODAL_WEBHOOK_*` variables are only
for deployments that expose a Modal webhook.

On Windows, `scripts/modal_with_env.ps1` safely loads Modal account credentials
from the ignored `.env` file and maps legacy `token_id` / `token_secret` names
to the environment variables expected by Modal. For example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/modal_with_env.ps1 app list
powershell -ExecutionPolicy Bypass -File scripts/modal_with_env.ps1 run scripts/modal_poc.py
```

## Default Docker runtime

Nemesis runs training inside a locked-down Docker container by default. The
container hosts the CybORG/Gymnasium simulation and the agent policies, has no
network interface, drops Linux capabilities, and writes only to the mounted
`runs/` artifact directory. A separate read-only dashboard container follows
those artifacts and publishes its UI only on host loopback.

On Windows (Docker Desktop with Linux containers):

```powershell
# Build, start the localhost dashboard, and run self-play training.
.\scripts\nemesis.ps1 up -RunName live

# Fast end-to-end validation.
.\scripts\nemesis.ps1 smoke -RunName smoke

# Pass additional self-play options after the named parameters.
.\scripts\nemesis.ps1 train -RunName experiment-1 -- --generations 10 --blue-steps 8000

# Stop the dashboard container.
.\scripts\nemesis.ps1 down
```

On Linux/macOS:

```bash
NEMESIS_RUN_NAME=live bash scripts/nemesis.sh up
NEMESIS_RUN_NAME=smoke bash scripts/nemesis.sh smoke
```

While a run is active, open <http://localhost:8001>. The dashboard follows the
run's event stream and evaluation metrics from `runs/selfplay/experiments/<run>`.
Set `NEMESIS_DASHBOARD_PORT` (shell script) or `-Port` (PowerShell) to use a
different host port.

The Docker runtime is different from `nemesis.sandbox.targets`: normal training
uses the in-process simulated CC4 network, while the target lab is an optional,
separately isolated set of intentionally vulnerable containers used only by the
sandbox evaluation adapter. Neither network is exposed to the training agent.

Direct host execution remains available for development and debugging, but it
is no longer the documented default.

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

# Train all 5 blue agents together, one PPOLearner each; resumable SB3 checkpoints go to checkpoints/ppo_team/<agent>
$env:PYTHONPATH="external\cage-challenge-4"; .venv\Scripts\python.exe scripts\train_ppo_team.py --timesteps 100000

# Train one blue agent with Stable-Baselines3 PPO; checkpoints (weights, optimizer, normalisation stats)
# go to checkpoints/sb3_<agent> every --save-every steps and on Ctrl+C. Re-run to resume; --fresh starts over.
$env:PYTHONPATH="external\cage-challenge-4"; .venv\Scripts\python.exe scripts\train_ppo_sb3.py --agent blue_agent_0 --timesteps 200000

# Watch red vs. blue live in a terminal-style web page at http://localhost:8000
$env:PYTHONPATH="external\cage-challenge-4"; .venv\Scripts\python.exe -m nemesis.live --delay 0.75

# CybORG's own test suite (slow)
cd external\cage-challenge-4; ..\..\.venv\Scripts\python.exe -m pytest CybORG\Tests\test_cc4
```

## Environment notes

- Scenario: `EnterpriseScenarioGenerator`, with 5 blue agents (`blue_agent_0..3` have obs 92 / 82 actions;
  `blue_agent_4` has obs 210 / 242 actions), green users, and a red attacker.
- Wrappers: `BlueFlatWrapper` (PettingZoo-style, flat vectors) and `EnterpriseMAE` (RLlib `MultiAgentEnv`).
- A step takes ~50 ms, so add rollout workers (`--workers`) to speed up training.
- Official examples are in `external/cage-challenge-4/CybORG/Evaluation/training_example/`.
