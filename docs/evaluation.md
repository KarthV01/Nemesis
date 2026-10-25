# Evaluation harness (`nemesis.eval`)

The evaluation stage scores a blue policy by running it for N episodes against the
scripted CC4 opponents and recording, per episode, the metric vector defined in
[`nemesis.platform.metrics`](../nemesis/platform/README.md) plus a summary. It is
the common yardstick the self-play league and the sandbox stage reuse, so the
exact derivation of every metric is documented here.

Everything runs inside CybORG's **simulated** CAGE-Challenge-4 enterprise
network. No real hosts, traffic, or exploits are involved.

## What CC4 gives us (research findings)

These findings come from reading the vendored simulator
(`external/cage-challenge-4/CybORG`) and from probe runs; the file/line
references are what each derivation relies on.

### Episode shape

* CC4 episodes **never terminate early**. `EnterpriseScenarioGenerator.determine_done`
  returns `step_count >= steps - 1`, i.e. the episode only ends as a *truncation*
  at the step limit. So there is no natural "blue survived / red finished" signal
  to read a win from (see `blue_win` below).
* One scenario has **5 blue agents** (one per security zone), **6 red agents**
  (one per starting zone; only the contractor-network one is active at reset and
  the others wake as red spreads), and **one green agent per user host**
  (~40-50 of them). Teams are read from
  `env.unwrapped.environment_controller.team` (`"Blue"`/`"Red"`/`"Green"`).

### Reward: `BlueRewardMachine`

`CybORG/Shared/BlueRewardMachine.py` is the only non-empty reward calculator.
Red and Green use `EmptyRewardCalculator` (always `0.0`). Each step the blue team
reward is a **sum of negative penalties** taken from a per-mission-phase,
per-subnet table:

| Key | Meaning | When it fires |
| --- | --- | --- |
| `LWF` | Local Work Fails | a green agent's `GreenLocalWork` returns `success=False` |
| `ASF` | Access Service Fails | a green agent's `GreenAccessService` returns `success=False` |
| `RIA` | Red Impact/Access | a red agent's `Impact` action returns `success=True` |

So blue is punished both when **users can't work** (availability lost — whatever
the cause, red *or* blue's own disruption) and when **red lands an impact**. The
reward is `<= 0`; closer to `0` is better defence.

The raw per-step team reward is at
`environment_controller.reward["Blue"]["BlueRewardMachine"]` (there is also an
`action_cost` entry, which is `0` in CC4). Reading it directly from the
controller makes the metric independent of which wrapper the policy uses.

### Ground-truth state

Every wrapper exposes the live simulator: `wrapped_env.unwrapped` is the `CybORG`
object and `cyborg.environment_controller` is the `SimulationController`. After a
step the controller holds, keyed by agent name:

* `.reward[team][calc]` — the reward just computed (reset each step).
* `.action[agent]` — a *list* of the action object(s) that agent just executed.
* `.observation[agent]` — an `ObservationSet`; `.observations[0].data["success"]`
  is the action's outcome as a `TernaryEnum` (`TRUE`/`FALSE`/`UNKNOWN`). This is
  exactly the field `BlueRewardMachine` reads for green failures.
* `.state.sessions[agent]` — `{id: Session}`; a `Session` has `.hostname` and
  `.has_privileged_access()` (True when `username in {"root","SYSTEM"}`). This is
  how the renderer (`SimulationController.add_red_actions`) decides a host is
  red-owned, and we use the same rule.
* `.state.hosts` — all hosts. Routers are named `*_router` and the internet node
  is `root_internet_host_0`; the remaining `*_user_host_*` / `*_server_host_*`
  are the attackable targets.

## How each metric is derived

Derivation lives in `nemesis/eval/harness.py` (`MetricProbe`), which reads the
controller after every step. Let a green **attempt** be a step in which a green
agent executed `GreenLocalWork` or `GreenAccessService` (not `Sleep`).

| Metric | Derivation |
| --- | --- |
| `blue_reward` | Σ over steps of `reward["Blue"]["BlueRewardMachine"]`. The canonical CC4 blue score. |
| `red_reward` | `-blue_reward`. **Derived proxy:** CC4 has no red reward (`EmptyRewardCalculator`); red's goal is exactly to maximise blue's penalties, so the zero-sum negative is the honest stand-in. Documented so the self-play league knows it is not an independent signal. |
| `attack_success_rate` | `|hosts red ever held a root session on| / |target hosts|`, where target hosts are the `*_user_host_*` + `*_server_host_*` (routers and the internet node excluded), and "root session" = a red `Session` with `has_privileged_access()`. Accumulated over the episode, so it measures red's total penetration regardless of later eviction. Range `[0,1]`. |
| `defender_availability` | `successful green attempts / total green attempts` over the episode — the fraction of user activity that worked. This is the complement of the LWF/ASF penalty pressure, normalised to `[0,1]`. `None` only if there were zero attempts. |
| `false_positives` | Count of green attempts that **failed** while **no red agent had any session on that green agent's host**. A user's work failing with no attacker on their host means blue's own disruption (a `Restore` that wiped services, or a `BlockTrafficZone` that cut the route) caused it — a defender false positive. Integer per episode. |
| `blue_win` | `None` by default: **CC4 defines no intrinsic win condition** (every episode truncates at the step limit; there is no objective flag to read). If an evaluation passes `win_reward_threshold`, we set `blue_win = blue_reward >= threshold`, which lets the league pick a relative bar. Left `None` otherwise rather than inventing one. |

`extra` additionally carries: `red_root_hosts_final` and `red_foothold_hosts`
(any-session reach) at episode end, `red_root_peak` (max concurrent root
footprint fraction), `green_attempts`, `green_failures`, `target_hosts`,
`wrapper`, `red_agent`, `blue_policy`, and the episode `seed`.

### Caveats

* `defender_availability` folds together red-caused and blue-caused outages; the
  blue-caused share is isolated by `false_positives`.
* `false_positives` uses "no red session **on the green host**" as the
  no-red-presence test. A `GreenAccessService` to a remote server can also fail
  from a block when red is on neither endpoint — that is still blue-caused and
  still counted. The test is deliberately conservative (red anywhere on the
  user's own host suppresses the count), so it under- rather than over-reports.
* `attack_success_rate` counts *root* footholds (the privilege level that enables
  `Impact`). Low-privilege-only footholds are reported separately in
  `extra["red_foothold_hosts"]`.

## The policy interface other stages call

One small interface unifies the three evaluable blue policies (see
`nemesis/eval/policies.py`):

```python
class BluePolicy(Protocol):
    wrapper: str                      # the EnvSpec.wrapper this policy needs
    def reset(self, env) -> None: ...
    def act(self, env, observations) -> dict: ...   # {agent: action | index}
```

* `random_blue()` → wrapper `"blue_flat"`; samples each blue agent's `Discrete`
  action space. The floor baseline.
* `scripted_blue(factory=BlueCaptain)` → wrapper `"none"`; drives the raw CybORG
  dict interface with a Nemesis heuristic agent per blue slot, reading
  `env.get_observation(agent)` / `env.get_action_space(agent)`.
* `trained_blue(checkpoint, agent="blue_agent_0")` → wrapper `"blue_flat"`; loads
  an SB3 model with `nemesis.checkpoints.load_learned` and acts greedily
  (`predict(..., deterministic=True)`); the other blue agents `Sleep`. If the
  checkpoint carries `VecNormalize` obs stats they are applied before predicting.

The harness picks the env wrapper from `policy.wrapper` (overriding the spec's),
so a caller only chooses a policy and an `EnvSpec`; metric derivation is identical
across wrappers because it reads the unwrapped controller.

### Running an evaluation (what callers use)

```python
from nemesis.platform import envs
from nemesis.eval import EvalConfig, evaluate, random_blue, scripted_blue, trained_blue

cfg = EvalConfig(spec=envs.evaluation_spec(steps=500), episodes=10)
episodes = evaluate(scripted_blue(), cfg)          # -> list[EpisodeMetrics]
summary  = summarize(episodes)                      # from nemesis.platform.metrics
```

CLI: `scripts/evaluate.py` evaluates a baseline or a checkpoint and writes
`<store-root>/experiments/<name>/evaluations.jsonl` (+ `config.json` recording the
`EnvSpec`) via `ArtifactStore` + `MetricsLog`, then prints the summary. The store
root defaults to `runs/` (git-ignored) so evaluations never land in the repo.

## NOTE FOR INTEGRATOR

* **No shared files were changed.** `nemesis.eval` and `scripts/evaluate.py` are
  additive. The harness only *reads* `env.unwrapped.environment_controller`; it
  does not touch `nemesis/platform/**`.
* **No new dependencies.** Uses only what the shared venv already has (CybORG,
  stable_baselines3, gymnasium, numpy, torch).
* If the self-play / sandbox stages want `red_reward` to be a *real* signal rather
  than `-blue_reward`, that requires a non-empty red reward calculator in
  `EnterpriseScenarioGenerator._generate_team_calcs` (a platform/scenario change,
  not an eval change).
