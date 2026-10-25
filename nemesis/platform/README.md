# `nemesis.platform` — shared contracts

The platform layer is the seam the parallel training stages build against so
they stay compatible. Depend on these modules, not on each other.

| Module | Owns | Used by |
| --- | --- | --- |
| `envs.py` | CybORG env construction (`EnvSpec` → wrapped env) | every trainer & evaluator |
| `storage.py` | the `red/ blue/ selfplay/ experiments/` layout; `Backend` seam | every trainer, Modal jobs |
| `metrics.py` | `EpisodeMetrics` / `summarize` / JSONL log | every evaluator |
| `events.py` | the dashboard/commentator event stream (matches `nemesis.live`) | live feed, dashboard, cloud jobs |

## The storage backend seam

`ArtifactStore` does all I/O through the `Backend` protocol. `LocalBackend` is
the default. To persist to a **Modal Volume**, implement the same methods against
the mounted volume path and construct `ArtifactStore(MyModalBackend(...))`; the
training code is identical. A remote backend whose files are not on the local
disk returns `None` from `local_path`, and callers stage into a temp dir.

## Reproducibility

Record the `EnvSpec.as_dict()` alongside each run (e.g.
`experiments/<name>/config.yaml`) so a checkpoint can be re-evaluated on the
exact scenario it trained on.

## Everything is simulated

Builders here produce CybORG CAGE-Challenge-4 environments only: a simulated
enterprise network with no real hosts or traffic. Sandbox evaluation against
real containers is a separate, explicitly-authorized stage that lives behind its
own adapter (`nemesis/sandbox/`), not here.
