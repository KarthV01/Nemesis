# Production deployment

Nemesis uses two deployments with separate responsibilities:

- **Vercel** serves the static, selectable arena dashboard. It never executes
  agents and contains no cloud credentials.
- **Modal** runs PyTorch/CybORG training and matches in isolated containers,
  persists checkpoints and replays in `nemesis-artifacts`, and exposes the
  dashboard API and event streams.

Vercel alone cannot train or run the arena: its deployment is the viewer and
control surface. The Modal API URL shown by `modal deploy` must be entered in
the dashboard's **Modal API URL** field. The launch token stays in browser
`sessionStorage`; read-only match viewing remains shareable.

## 1. Create the Modal launch secret

Generate a long random token and set the Vercel origin you intend to use:

```powershell
$launchToken = .venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
powershell -ExecutionPolicy Bypass -File scripts/modal_with_env.ps1 secret create nemesis-api `
  "NEMESIS_API_TOKEN=$launchToken" `
  "NEMESIS_ALLOWED_ORIGINS=https://YOUR-PROJECT.vercel.app,http://localhost:8001"
```

Save `$launchToken` in a password manager. Do not put its real value in Git or
Vercel's public frontend bundle.

## 2. Deploy Modal compute and API

```powershell
powershell -ExecutionPolicy Bypass -File scripts/modal_deploy.ps1
```

The deploy wrapper intentionally uses the `modal_app.api` module entrypoint so
Modal imports the current package source rather than retaining a stale
file-entrypoint container.

That one deployment registers:

- `train_selfplay`, `train_blue`, and `train_red` on 4 physical CPU cores and
  8 GiB RAM with CPU-only PyTorch;
- `run_arena_match` on a network-blocked per-match container;
- `dashboard_api`, the authenticated launch/read-only monitoring API.

Copy the `dashboard_api` URL printed by Modal. A first real alternating run can
then be launched entirely on Modal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/modal_with_env.ps1 run scripts/modal_train.py `
  --side selfplay --generations 5 --blue-steps 4000 --red-steps 4000
```

No training happens on the laptop. The local process submits the job and shows
logs; checkpoints remain on the Modal Volume.

For a durable background run that continues after the local terminal exits,
launch the already-deployed function asynchronously:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/modal_with_env.ps1 run scripts/modal_launch.py `
  --run-id selfplay-prod --generations 5 --blue-steps 4000 --red-steps 4000
```

## 3. Deploy the static dashboard to Vercel

The repository's `vercel.json` skips a build and publishes `web/dashboard`.
Connect the GitHub repository to Vercel with the **Other** framework preset, or
use the Vercel CLI:

```powershell
vercel --prod
```

Open the deployed site, enter the Modal API URL and launch token, then save the
connection. The checkpoint selectors populate from the Modal Volume. You can
launch 5v5 or literal 1v1 matches and watch their live event stream; saved
matches remain selectable as replays. Modal self-play runs also appear under
**training runs** with their live train/metric events and evaluation charts.

## Compute choice

The current agents are small Stable-Baselines3 PPO networks. CybORG environment
stepping is the expensive part, so the default is CPU, not GPU. Start with the
configured 4 physical cores/8 GiB. Add an L4 only after profiling demonstrates
that batched neural-network updates—not simulation—are the bottleneck; doing so
also requires switching the image from the CPU PyTorch wheel to a CUDA build.
