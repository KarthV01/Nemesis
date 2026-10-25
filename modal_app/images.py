"""Modal image, App and Volume declarations for the Nemesis cloud stage.

Everything here is guarded so the module imports (and the whole ``modal_app``
package is testable) **without** ``modal`` installed. When ``modal`` is absent,
``MODAL_AVAILABLE`` is ``False`` and ``image`` / ``app`` / ``volume`` are
``None``; the plain constants (names, mount path, dependency lists) are always
defined so docs, tests and the ``--local`` paths can use them.

The image mirrors the local setup (``setup.sh`` / ``setup.ps1``): Python 3.10,
the PyTorch **CPU** build (CybORG/PPO training is CPU-bound), numpy 1.26,
stable-baselines3 2.3.2, pettingzoo, and the vendored CybORG CAGE-Challenge-4
submodule installed in editable-compat mode (its ``setup.py`` declares no
packages, so compat mode is what makes ``import CybORG`` work).
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Plain constants — always defined, with or without modal.
# --------------------------------------------------------------------------- #
APP_NAME = "nemesis"
VOLUME_NAME = "nemesis-artifacts"
#: Where the Volume is mounted inside the container.
VOLUME_MOUNT_PATH = "/vol"
#: Store root on the Volume (what ``ArtifactStore`` writes under).
ARTIFACTS_PATH = f"{VOLUME_MOUNT_PATH}/artifacts"

PYTHON_VERSION = "3.10"
TORCH_VERSION = "2.2.0"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
#: Deps installed on top of the CPU torch build. Pins match the shared venv.
PIP_PACKAGES = [
    "numpy==1.26.4",
    "stable-baselines3==2.3.2",
    "pettingzoo==1.24.3",
    "gymnasium",
]

#: Repo root and the vendored CybORG submodule (copied into the image at build).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC4_LOCAL_DIR = os.path.join(REPO_ROOT, "external", "cage-challenge-4")
CC4_REMOTE_DIR = "/root/cage-challenge-4"

try:  # pragma: no cover - exercised only where modal is installed
    import modal

    MODAL_AVAILABLE = True
except Exception:  # ImportError, or a partial install
    modal = None  # type: ignore
    MODAL_AVAILABLE = False


def build_image():
    """Construct the Modal image. Requires ``modal`` to be installed.

    Build-time steps, in order:

    1. Python 3.10 Debian slim base.
    2. ``torch==2.2.0`` from the PyTorch **CPU** index (no CUDA download).
    3. numpy / SB3 / pettingzoo / gymnasium.
    4. Copy the CybORG submodule in (``copy=True`` so the later ``run_commands``
       can see it at build time) and install its requirements, then the package
       itself in editable-compat mode.
    5. Add ``nemesis`` and ``modal_app`` as local Python source.
    """
    if not MODAL_AVAILABLE:
        raise RuntimeError(
            "modal is not installed; `pip install modal` to build the image. "
            "All non-Modal code paths work without it."
        )
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        # CPU torch wheel: satisfies CC4's torch==2.2.0 pin without CUDA.
        .pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_CPU_INDEX)
        .pip_install(*PIP_PACKAGES)
        .add_local_dir(CC4_LOCAL_DIR, remote_path=CC4_REMOTE_DIR, copy=True)
        .run_commands(
            f"pip install -r {CC4_REMOTE_DIR}/Requirements.txt",
            f"pip install -e {CC4_REMOTE_DIR} --config-settings editable_mode=compat",
        )
        # The Nemesis code the functions import. Added last so edits to it don't
        # bust the (expensive) dependency layers above.
        .add_local_python_source("nemesis", "modal_app")
    )


if MODAL_AVAILABLE:  # pragma: no cover - needs modal installed
    image = build_image()
    app = modal.App(APP_NAME)
    #: Persistent artifact store. ``create_if_missing`` makes the first run self
    #: -provisioning; every stage shares this one Volume.
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
else:
    image = None
    app = None
    volume = None


__all__ = [
    "APP_NAME",
    "VOLUME_NAME",
    "VOLUME_MOUNT_PATH",
    "ARTIFACTS_PATH",
    "PYTHON_VERSION",
    "TORCH_VERSION",
    "PIP_PACKAGES",
    "CC4_LOCAL_DIR",
    "CC4_REMOTE_DIR",
    "MODAL_AVAILABLE",
    "build_image",
    "image",
    "app",
    "volume",
]
