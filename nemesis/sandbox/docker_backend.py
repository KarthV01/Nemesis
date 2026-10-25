"""A local-Docker :class:`SandboxBackend` via guarded ``docker`` CLI subprocess calls.

Why the CLI and not the Docker SDK? So the sandbox stage adds **no hard
dependency**: the ``docker`` Python package need not be installed, and when the
daemon is off everything degrades to a clean :class:`SandboxUnavailable` instead
of an import error. Every call shells out to ``docker`` through
:meth:`_docker`, which is wrapped so a missing binary or dead daemon surfaces as
``SandboxUnavailable``.

Safety is built into *how* containers are launched, not bolted on afterwards:

* the network is created with ``--internal`` — **no gateway, no internet**;
* containers are started with ``--cap-drop ALL``, ``--security-opt
  no-new-privileges``, a pids/memory cap, and **no published ports** (``-p`` is
  never passed), so a target is unreachable from the host;
* ``--pull never`` — images must already be built locally; the daemon never
  reaches out to a registry;
* probes are **read-only** (``docker inspect`` / ``docker logs``); the backend
  never ``exec``s an arbitrary command inside a container.

Nothing here starts Docker or launches a container on import.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import List, Optional

from nemesis.sandbox.backend import (
    BENIGN_PROBES,
    ProbeResult,
    SandboxBackend,
    SandboxContainer,
    SandboxNetwork,
    SandboxUnavailable,
)

#: Marker the lab defender prints when it believes the target was attacked, and
#: the health lines it prints. The ``collect_logs`` probe scans for these so the
#: verifier gets an independent detection/availability signal. Kept in sync with
#: ``nemesis/sandbox/targets/defender/monitor.py``.
INTRUSION_LOG_MARKER = "INTRUSION-DETECTED"
SERVICE_UP_MARKER = "SERVICE-UP"
SERVICE_DOWN_MARKER = "SERVICE-DOWN"

_DOCKER_TIMEOUT_S = 30
_START_WAIT_S = 20


class DockerBackend(SandboxBackend):
    """Run isolated lab targets with the local ``docker`` CLI.

    Parameters
    ----------
    docker_bin : str
        Name or path of the docker executable.
    timeout_s : int
        Per-command timeout.
    """

    def __init__(self, docker_bin: str = "docker", timeout_s: int = _DOCKER_TIMEOUT_S):
        self.docker_bin = docker_bin
        self.timeout_s = timeout_s
        self._available: Optional[bool] = None  # cached

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        """True iff the docker binary exists and the daemon answers. Never raises."""
        if self._available is not None:
            return self._available
        available = False
        if shutil.which(self.docker_bin) is not None:
            try:
                result = subprocess.run(
                    [self.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                available = result.returncode == 0 and bool(result.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                available = False
        self._available = available
        return available

    # ------------------------------------------------------------------ #
    # Network
    # ------------------------------------------------------------------ #
    def create_network(self, name: str) -> SandboxNetwork:
        self.require_available()
        # --internal: no gateway is attached, so containers on this network
        # cannot reach the host or the internet.
        out = self._docker(
            ["network", "create", "--driver", "bridge", "--internal", name]
        )
        return SandboxNetwork(name=name, internal=True, id=out.strip() or None)

    def remove_network(self, network: SandboxNetwork) -> None:
        self.require_available()
        self._docker(["network", "rm", network.name], check=False)

    # ------------------------------------------------------------------ #
    # Container lifecycle
    # ------------------------------------------------------------------ #
    def start_target(
        self, image: str, name: str, network: SandboxNetwork
    ) -> SandboxContainer:
        self.require_available()
        if not network.internal:
            raise SandboxUnavailable(
                "refusing to start a target on a non-internal network; "
                "the lab must have no route to the host or internet"
            )
        args = [
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            network.name,
            "--pull",
            "never",            # never reach a registry; image must be local
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            "256m",
            "--restart",
            "no",
            # NOTE: no "-p"/"--publish" ever — the target stays unreachable from the host.
            image,
        ]
        cid = self._docker(args).strip()
        return SandboxContainer(name=name, image=image, network=network.name, id=cid or None)

    def stop_target(self, container: SandboxContainer) -> None:
        self.require_available()
        # Best-effort: stop then remove; ignore "no such container".
        self._docker(["stop", "--time", "5", container.name], check=False)
        self._docker(["rm", "--force", container.name], check=False)

    def disconnect_target(self, container: SandboxContainer, network: SandboxNetwork) -> None:
        """Benign "isolate": cut the container off the lab network."""
        self.require_available()
        self._docker(["network", "disconnect", network.name, container.name], check=False)

    def connect_target(self, container: SandboxContainer, network: SandboxNetwork) -> None:
        """Benign "reconnect": re-attach the container to the lab network."""
        self.require_available()
        self._docker(["network", "connect", network.name, container.name], check=False)

    # ------------------------------------------------------------------ #
    # Probes (read-only)
    # ------------------------------------------------------------------ #
    def probe(self, container: SandboxContainer, probe: str) -> ProbeResult:
        self.require_available()
        if probe not in BENIGN_PROBES:
            raise ValueError(
                f"{probe!r} is not a benign probe; allowed: {sorted(BENIGN_PROBES)}"
            )
        if probe == "service_health":
            return self._probe_health(container)
        return self._probe_logs(container)

    def _probe_health(self, container: SandboxContainer) -> ProbeResult:
        """Read the container's health/running state — no exec, no network call."""
        fmt = "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}"
        out = self._docker(
            ["inspect", "--format", fmt, container.name], check=False
        ).strip()
        up = out in ("healthy", "running")
        return ProbeResult(
            probe="service_health",
            ok=bool(out),
            detail=out or "no such container",
            raw={"service_up": up, "state": out},
        )

    def _probe_logs(self, container: SandboxContainer, tail: int = 50) -> ProbeResult:
        """Read recent log lines and scan them for the lab's known markers."""
        out = self._docker(
            ["logs", "--tail", str(tail), container.name], check=False
        )
        intrusion = INTRUSION_LOG_MARKER in out
        # Availability = the most recent SERVICE-UP/SERVICE-DOWN line seen.
        service_up = False
        for line in out.splitlines():
            if SERVICE_UP_MARKER in line:
                service_up = True
            elif SERVICE_DOWN_MARKER in line:
                service_up = False
        return ProbeResult(
            probe="collect_logs",
            ok=True,
            detail=out[-500:],
            raw={"intrusion_detected": intrusion, "service_up": service_up},
        )

    # ------------------------------------------------------------------ #
    # Low-level docker call
    # ------------------------------------------------------------------ #
    def _docker(self, args: List[str], *, check: bool = True) -> str:
        """Run ``docker <args>`` and return stdout.

        A missing binary or unreachable daemon becomes :class:`SandboxUnavailable`.
        When ``check`` is False a non-zero exit returns stdout rather than raising
        (used for best-effort teardown and inspections of maybe-absent objects).
        """
        try:
            result = subprocess.run(
                [self.docker_bin, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailable(f"docker binary not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxUnavailable(f"docker command timed out: docker {' '.join(args)}") from exc
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if _looks_like_daemon_down(stderr):
                raise SandboxUnavailable(f"docker daemon not reachable: {stderr}")
            if check:
                raise SandboxUnavailable(
                    f"docker {' '.join(args)} failed: {stderr or result.stdout.strip()}"
                )
        return result.stdout


def _looks_like_daemon_down(stderr: str) -> bool:
    low = stderr.lower()
    return any(
        s in low
        for s in (
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "error during connect",
            "the system cannot find the file specified",
            "dockerdesktoplinuxengine",
        )
    )


__all__ = [
    "DockerBackend",
    "INTRUSION_LOG_MARKER",
    "SERVICE_UP_MARKER",
    "SERVICE_DOWN_MARKER",
]
