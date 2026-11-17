# Arena matches

The arena dashboard is both the frontend and the match job controller. A match
submission is validated, assigned an ID, and launched in a fresh Docker
container. The match container has no network, a read-only root filesystem, no
Linux capabilities, a process/memory/CPU limit, and only its match artifact
directory mounted writable.

Build the runtime image once, then start the controller from the repository
root:

```powershell
docker build -t nemesis-runtime:local .
$env:PYTHONPATH="external/cage-challenge-4"
.venv\Scripts\python.exe scripts\dashboard.py --arena --port 8001
```

Open <http://localhost:8001>, choose the two teams, scenario, match format,
roles (for a role-specific matchup), length, seed, and stream delay, then select
**RUN MATCH**. A 5v5 uses the selected team captains/generalists. A 1v1 keeps
CybORG's structural enterprise team slots but activates only `red_agent_0` and
`blue_agent_0`; every other red/blue slot sleeps. The selected specialists are
therefore the only policies taking actions.

Each match is stored under `runs/arena/<match-id>/`:

- `config.json` is the submitted immutable configuration.
- `events.jsonl` is appended while the match runs and contains actions,
  per-step rewards, state changes, and the final result.
- `result.json` is the final score, winner, state, and configuration.

The live viewer and saved replay buttons both consume `events.jsonl` through
the same SSE endpoint. Restarting the dashboard does not remove completed
matches; it discovers them from the artifact directory.

The checkpoint selectors list complete blue and red PPO snapshots from the
artifact store. A submitted checkpoint is mounted read-only into the sandbox
and loaded by the arena runner. Either side may use a learned checkpoint or a
built-in scripted policy. In 1v1, only each side's `agent_0` checkpoint acts;
in 5v5, the blue team's five policy directories are loaded and a single red
checkpoint controls `red_agent_0` while the remaining red slots use the
selected built-in roster. Invalid, incomplete, cross-side, and path-traversal
references are rejected before Docker is launched.
