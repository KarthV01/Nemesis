# Nemesis platform architecture

Nemesis is a multi-agent reinforcement-learning platform for cybersecurity,
built around repeated attack-and-defense episodes in CybORG's **simulated**
CAGE-Challenge-4 enterprise network. The goal is a reproducible pipeline:

1. **Red training** — attacker policies learn offensive strategy in simulation.
2. **Blue training** — defender policies learn to detect and mitigate.
3. **Adversarial self-play** — red and blue improve against each other, with a
   league of frozen past opponents to avoid overfitting to the latest rival.
4. **Evaluation** — trained policies are scored against baselines, held-out
   scenarios, and (behind an explicit adapter) isolated, authorized sandbox
   targets.
5. **LLM agents (experiment)** — small open-weight models play the same game
   with outputs constrained to legal actions, compared against PPO, with LoRA
   RL fine-tuning on cloud GPU.

The default runtime is a local Docker sandbox: CybORG/Gymnasium and the agent
policies execute together in a networkless training container. The laptop is
the control plane and exposes only the loopback-bound dashboard. The same image
can run on remote Docker compute; Modal remains an optional compute backend,
not the security boundary for normal runs.

```
                      ┌──────────────────────────────────────────┐
                      │   Control plane (laptop)                  │
                      │   config · metrics · live dashboard       │
                      └───────────────┬──────────────────────────┘
                                      │  nemesis.platform contracts
       ┌───────────────┬─────────────┼───────────────┬────────────────┐
       ▼               ▼             ▼               ▼                ▼
   Red training    Blue training  Self-play       Modal jobs      Dashboard
   (CybORG + RL)  (CybORG + RL)   league          (cloud compute) (events)
       │               │             │               │
       └───── checkpoints + experiment logs (ArtifactStore) ──────┘
                                      │
                                      ▼
                      ┌──────────────────────────────────────────┐
                      │   Sandbox evaluation (nemesis.sandbox)    │
                      │   isolated Docker targets · action adapter │
                      └──────────────────────────────────────────┘
```

## The shared contract layer: `nemesis.platform`

Everything new depends on `nemesis.platform`, never directly on another stage.
See [`nemesis/platform/README.md`](../nemesis/platform/README.md). The four
contracts: `envs` (build environments), `storage` (checkpoint/experiment
layout), `metrics` (evaluation outcomes), `events` (dashboard stream).

## Parallel build: branch ownership

Each workstream owns its own files and branch, and pushes meaningful commits as
it goes. An integrator keeps everything merging cleanly into `main`. To avoid
conflicts, **only the integrator edits shared files** (`README.md`,
`nemesis/agents/__init__.py`, `setup.*`); each stage adds its own modules and its
own doc page under `docs/` and its own tests.

| Branch | Owns (new paths) | Depends on |
| --- | --- | --- |
| `platform/contracts` | `nemesis/platform/**`, `docs/architecture.md` | — (foundation) |
| `stage/eval-harness` | `nemesis/eval/**`, `scripts/evaluate.py`, `docs/evaluation.md` | platform |
| `stage/red-training` | `nemesis/red_env.py`, `scripts/train_red.py`, `docs/red-training.md` | platform |
| `stage/modal` | `modal_app/**`, `scripts/modal_*.py`, `docs/modal.md` | platform, storage |
| `stage/selfplay` | `nemesis/league/**`, `scripts/train_selfplay.py`, `docs/selfplay.md` | platform, eval |
| `stage/sandbox-eval` | `nemesis/sandbox/**`, `docs/sandbox-eval.md` | platform, metrics |
| `stage/llm-agents` | `nemesis/llm/**`, `modal_app/llm_*.py`, `docs/llm-agents.md` | platform, modal |
| `stage/dashboard` | `nemesis/dashboard/**`, `web/dashboard/**`, `docs/dashboard.md` | events, metrics |

## Rules for every stage

- Build inside CybORG's simulator. Agents emit only the simulator's abstract
  action objects. The sandbox stage runs real containers only on an isolated
  internal network with no internet, against intentionally-vulnerable lab
  targets, and maps agent decisions to an explicit allow-list of test actions.
- Modal code is written and locally testable but **not launched** until the user
  provides credentials. Keep a `--dry-run`/local path for every Modal entrypoint.
- Record the `EnvSpec` with every run so results are reproducible.
- Prefer reward **plus** the metric vector in `nemesis.platform.metrics`; win
  rate alone is misleading while both sides learn.

## Runtime isolation

`compose.yaml` defines two services with deliberately different responsibilities:

- `trainer` runs the simulator and policies with `network_mode: none`, a
  read-only root filesystem, no Linux capabilities, and `/artifacts` as its only
  persistent writable mount.
- `dashboard` reads the same artifact mount and serves the event/metric UI on
  `127.0.0.1:8001`. Its Docker network is internal and cannot reach the internet.

The containers under `nemesis/sandbox/targets` are a separate evaluation lab.
They do not host the Gym environment or ordinary training agents. Keeping that
boundary explicit prevents an abstract CybORG action from silently becoming a
real network action.
