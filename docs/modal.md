# Modal cloud compute for Nemesis

This stage (`stage/modal`, owning `modal_app/**`, `scripts/modal_*.py`,
`docs/modal.md`) runs Nemesis training jobs on [Modal](https://modal.com) and
persists their results to a **Modal Volume**, with the *same code path* runnable
locally. It depends only on `nemesis.platform` (the storage + envs contracts)
and on the existing `nemesis.ppo_learner` / `nemesis.checkpoints` training code.

> **Nothing here is launched yet.** The Modal CLI has no auth token in this
> workspace. Every entrypoint has a `--local` / `run_training(...)` path that
> runs now without Modal installed or authenticated. `import modal` is guarded
> everywhere, so every module imports and every test passes with `modal` absent.
> Run the Modal commands in [Running on Modal](#running-on-modal) only after you
> add your key.

Everything trained here runs inside CybORG's **simulated** CAGE-Challenge-4
enterprise network (no real hosts, services, or exploits), exactly as the rest
of Nemesis does.

## How it fits the platform

`nemesis.platform.storage.ArtifactStore` does all its I/O through a tiny
`Backend` protocol (`makedirs / exists / listdir / open / remove_tree /
local_path`). `LocalBackend` is the default; a Modal Volume is just a real
filesystem mounted inside the container, so persisting to the cloud is one new
backend and *identical* training code:

```
ArtifactStore(LocalBackend("artifacts"))            # laptop / sandbox
ArtifactStore(VolumeBackend("/vol/artifacts", vol)) # inside a Modal container
```

The checkpoint layout (`<side>/<run_id>/checkpoint_NNNNNN/<agent>/`) is the
store's, not Modal's, so a checkpoint trained on Modal loads with plain
`nemesis.checkpoints.load_learned` / `PPOLearner.load` on the laptop.

## Key findings (research)

### Image

- CC4 pins **numpy 1.26 / torch 2.2**, which have no wheels for Python ≥ 3.13,
  so the image is **Python 3.10** (matches `setup.ps1` / `setup.sh` and the
  shared venv: Python 3.10.21, sb3 2.3.2, torch 2.2.0+cpu, numpy 1.26.4,
  pettingzoo 1.24.3).
- **Torch is the CPU build.** CybORG/PPO training is CPU-bound (see
  `docs/docker-sandbox.md`), so we install `torch==2.2.0` from the PyTorch CPU
  index (`https://download.pytorch.org/whl/cpu`) — the same wheel `setup.sh`
  pre-installs — avoiding the ~2 GB CUDA download and GPU cost entirely.
- **CybORG** is the vendored submodule `external/cage-challenge-4`. It installs
  the same way as locally: `pip install -r Requirements.txt` then
  `pip install -e . --config-settings editable_mode=compat` (CC4's `setup.py`
  declares no packages, so compat mode is what puts the repo dir on `sys.path`
  and makes `import CybORG` work). The submodule dir is copied into the image at
  build time (`add_local_dir(..., copy=True)`), then those two installs run.
- Extra deps installed explicitly: `numpy==1.26.4`, `stable-baselines3==2.3.2`,
  `pettingzoo==1.24.3`, `gymnasium`.
- `nemesis` + `modal_app` are added as local Python source so the functions can
  `import nemesis.platform` etc.

### Volume

- `modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)` is mounted at
  `/vol`; the store writes under `/vol/artifacts`.
- A Modal Volume **is a real filesystem inside the container**, so
  `VolumeBackend.local_path()` returns the real path and SB3's `model.save`
  (which writes several files into a directory) works directly — no temp-dir
  staging needed. `VolumeBackend` therefore subclasses `LocalBackend` and
  satisfies the identical `Backend` protocol; it passes the same behavioural
  tests against a local temp dir.
- **`commit()` semantics.** Writes to the mounted path are durable to *that*
  container immediately, but only become durable/visible elsewhere when the
  Volume is committed. Modal auto-commits on container exit; we also call
  `volume.commit()` right after writing each checkpoint so a crash or a timeout
  mid-run never loses the last saved checkpoint. `reload()` pulls in changes
  other containers have committed. With no Modal Volume attached (local temp
  dir, `volume=None`) both are documented no-ops because the local filesystem
  is already durable.

### CPU vs GPU, and cost

- **PPO/CybORG training → CPU.** `train_blue` / `train_red` request CPU cores
  (default `cpu=8.0`), **no GPU**. CybORG stepping is the bottleneck; a GPU sits
  idle and only adds cost.
- **GPU (L4) is for the LLM-agents stage**, not this one: LoRA RL fine-tuning of
  a small open-weight model (see `docs/architecture.md`). That stage reuses this
  image/Volume but adds a GPU to its own function. An **L4** (24 GB) is the
  cheapest GPU that comfortably fits a 7–8 B model in bf16 for LoRA.
- **Approximate Modal rates** (verify current numbers at
  <https://modal.com/pricing> — billed per second, you pay only while the
  function runs):
  - CPU ≈ $0.135 / physical core / hour, memory ≈ $0.024 / GiB / hour.
  - GPU: **L4 ≈ $0.80/hr**, T4 ≈ $0.59/hr, A10G ≈ $1.10/hr, A100-40GB ≈
    $2.10/hr, H100 ≈ $3.95/hr (GPU price includes a CPU/RAM allotment).
  - A 50-minute blue training run on 8 CPU cores ≈ 8 × $0.135 × (50/60) ≈
    **$0.90**, plus a little RAM. A 2-hour L4 LoRA job ≈ **$1.60** of GPU.
  - Idle Volume storage is a few cents per GB-month.

### Parallel-worker safety

Modal Volumes are **not** safe for concurrent writes to the *same files* — a
commit is last-writer-wins and overlapping concurrent commits can lose data.
The store layout already avoids this: give **each worker a unique `run_id`** so
every worker writes under its own `<side>/<run_id>/` subtree and no two touch
the same path. `new_checkpoint` auto-increments by listing a run dir, so two
workers must *never* share a `run_id` (that listing would race). `modal_poc.py`
and `train_blue` default `run_id` to a UTC timestamp for exactly this reason.

## The proof of concept (`scripts/modal_poc.py`)

The flow the user asked for — *train a blue policy on Modal, save the checkpoint
to the Volume, shut the compute down, then reload the checkpoint locally and run
one evaluation* — implemented so the `--local` path does the identical sequence
now, with tiny timesteps and no Modal:

1. **Train** `run_training(...)` for a few hundred timesteps, writing a
   checkpoint through `ArtifactStore(backend)` (`VolumeBackend` on Modal,
   `LocalBackend` locally).
2. **Persist + release compute.** On Modal the function returns and the
   container is torn down (Volume committed); locally this is just the store
   root on disk.
3. **Reload locally.** A fresh `ArtifactStore` over the same root loads the
   latest checkpoint with `PPOLearner.load` (one learner per blue agent).
4. **Evaluate.** Run one deterministic episode and report
   `nemesis.platform.metrics.EpisodeMetrics` / `summarize`.

```bash
# Runs now, no Modal, no auth — mock multi-agent env, instant:
.venv/Scripts/python.exe scripts/modal_poc.py --local

# Faithful proof on the real (simulated) CybORG CC4 env (~30 s to build it):
.venv/Scripts/python.exe scripts/modal_poc.py --local --cyborg --timesteps 400
```

## Running on Modal

See `modal_app/README.md` for the exact `modal token new` + `modal run`
commands, GPU/CPU choices and Volume inspection commands. In short, once you
have a key:

```bash
pip install modal
modal token new                       # one-time browser auth
modal run scripts/modal_poc.py        # runs the POC's Modal path
modal run modal_app/train.py::train_blue --timesteps 200000 --run-id blue-$(date +%s)
```

## NOTE FOR INTEGRATOR

Shared files this stage must **not** edit; requests for the integrator:

- **No shared-file edits were made.** `README.md`, `nemesis/agents/__init__.py`,
  `setup.*`, `nemesis/live.py`, `web/index.html` and all of `nemesis/platform/*`
  are untouched. Everything new lives under `modal_app/`, `scripts/modal_poc.py`,
  `docs/modal.md`, `tests/test_modal_app.py`.
- **`modal` is a new, optional dependency.** It is intentionally *not* added to
  `setup.*` or any requirements file — it is needed only to launch cloud jobs,
  the whole stage imports and tests without it, and adding it to the shared venv
  is the integrator's call (`pip install modal`). Suggested: an optional extra
  rather than a hard dependency.
- **`ArtifactStore` has no `put_dir` yet.** `nemesis/platform/README.md`
  describes a fully-remote backend that returns `None` from `local_path` and has
  callers "stage into a temp dir and `ArtifactStore.put_dir` it back", but
  `put_dir` is not implemented in `storage.py`. This stage does **not** need it:
  a Modal Volume is a real local fs, so `VolumeBackend.local_path` returns a real
  path and SB3 saves straight to it. If a future backend (e.g. S3) is truly
  remote, the integrator should add `ArtifactStore.put_dir(rel_path, local_dir)`
  to the platform layer; `VolumeBackend` would not need it.

### How the `stage/llm-agents` stage reuses this

Per `docs/architecture.md`, `stage/llm-agents` owns `modal_app/llm_*.py` and
depends on `platform, modal`. It should:

- **Reuse the same `App` + `Volume`** from `modal_app.images`
  (`app`, `volume`, `VOLUME_MOUNT_PATH`, `VolumeBackend`) so LLM checkpoints land
  in the same artifact tree and the laptop reads them the same way.
- **Extend the image, don't fork it**: start from `modal_app.images.image` and
  add GPU-side deps (`transformers`, `peft`, `trl`, a CUDA torch build) in
  `llm_*.py`, and request a GPU on *its* function (`gpu="L4"`), leaving
  `train_blue`/`train_red` on CPU.
- **Write under a distinct `run_id`/side subtree** on the Volume to stay within
  the parallel-worker-safety rule above.
