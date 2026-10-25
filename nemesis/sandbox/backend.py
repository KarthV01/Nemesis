"""Abstract sandbox backend — the seam between the evaluation harness and a
real container runtime.

Sandbox evaluation takes a *trained, simulated* blue policy and lets it act
against **isolated, authorized, intentionally-vulnerable lab targets** so we can
check that its defensive decisions produce good *real* outcomes (the service
stayed available, an intrusion was detected) and not just good simulator reward.

A :class:`SandboxBackend` exposes only a handful of **benign** verbs:

* create / remove an *isolated* network (no gateway, no internet),
* start / stop a pre-built lab target container on that network,
* disconnect / reconnect a container from the network (the one benign
  containment primitive, used for an "isolate" response),
* run a **benign probe** (a liveness/health check or a log read).

That is the whole vocabulary. There is deliberately no verb for running an
arbitrary command, publishing a port, exploiting, or moving laterally: the
harness can only ever launch the vendor's vulnerable target and observe it.

The backend is abstract so the evaluation code never imports Docker directly.
:class:`~nemesis.sandbox.docker_backend.DockerBackend` is the real
implementation; tests use a fake. When no runtime is usable the backend raises
:class:`SandboxUnavailable`, which every caller must treat as *skip / no-op* —
sandbox evaluation is an optional stage and the rest of Nemesis runs without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

#: The only probe names a backend will honour. Probes are allow-listed too:
#: both are read-only observations of the target, never changes to it.
#:   * ``service_health`` — is the target's service up? (liveness/health check)
#:   * ``collect_logs``   — read recent log lines for independent verification.
BENIGN_PROBES = frozenset({"service_health", "collect_logs"})


class SandboxUnavailable(RuntimeError):
    """The sandbox runtime (e.g. the Docker daemon) is not usable right now.

    Raised by a backend when Docker is not installed, the daemon is not running,
    or a required feature is missing. Callers should catch it and **skip**
    sandbox evaluation cleanly rather than failing the run.
    """


@dataclass(frozen=True)
class SandboxNetwork:
    """An isolated container network.

    ``internal`` must stay ``True``: an internal Docker network has no gateway,
    so containers on it cannot reach the host or the internet.
    """

    name: str
    internal: bool = True
    id: Optional[str] = None


@dataclass(frozen=True)
class SandboxContainer:
    """A running (or stopped) lab target container."""

    name: str
    image: str
    network: Optional[str] = None
    id: Optional[str] = None


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of one benign probe against a container."""

    probe: str
    ok: bool
    detail: str = ""
    #: Parsed structured fields, e.g. ``{"service_up": True, "intrusion": False}``.
    raw: Dict[str, object] = field(default_factory=dict)


class SandboxBackend(ABC):
    """Abstract container-runtime backend for sandbox evaluation.

    Concrete backends implement the eight primitives below. The non-abstract
    helpers (:meth:`require_available`, :meth:`restart_target`,
    :meth:`apply_test_action`) are built on those primitives and are shared by
    every backend, including test fakes.
    """

    # ------------------------------------------------------------------ #
    # Primitives — concrete backends must implement these.
    # ------------------------------------------------------------------ #
    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` iff the runtime is usable (never raise)."""

    @abstractmethod
    def create_network(self, name: str) -> SandboxNetwork:
        """Create an **internal** (no-gateway) network and return it."""

    @abstractmethod
    def remove_network(self, network: SandboxNetwork) -> None:
        """Remove a network created by :meth:`create_network`."""

    @abstractmethod
    def start_target(
        self, image: str, name: str, network: SandboxNetwork
    ) -> SandboxContainer:
        """Start a pre-built lab target container attached to ``network``.

        Implementations must not publish ports, grant extra capabilities, or
        give the container internet access.
        """

    @abstractmethod
    def stop_target(self, container: SandboxContainer) -> None:
        """Stop and remove a container started by :meth:`start_target`."""

    @abstractmethod
    def disconnect_target(self, container: SandboxContainer, network: SandboxNetwork) -> None:
        """Detach ``container`` from ``network`` — the benign "isolate" action."""

    @abstractmethod
    def connect_target(self, container: SandboxContainer, network: SandboxNetwork) -> None:
        """Re-attach ``container`` to ``network`` — the benign "reconnect" action."""

    @abstractmethod
    def probe(self, container: SandboxContainer, probe: str) -> ProbeResult:
        """Run one benign probe (``probe`` must be in :data:`BENIGN_PROBES`)."""

    # ------------------------------------------------------------------ #
    # Shared helpers built on the primitives.
    # ------------------------------------------------------------------ #
    def require_available(self) -> None:
        """Raise :class:`SandboxUnavailable` unless the runtime is usable."""
        if not self.is_available():
            raise SandboxUnavailable(
                "no usable container runtime; start the sandbox runtime (e.g. "
                "Docker Desktop) or skip sandbox evaluation"
            )

    def restart_target(
        self, container: SandboxContainer, network: SandboxNetwork
    ) -> SandboxContainer:
        """Benign "restart service": stop the container and start it again."""
        self.stop_target(container)
        return self.start_target(container.image, container.name, network)

    def apply_test_action(
        self,
        test_action: str,
        container: SandboxContainer,
        network: SandboxNetwork,
    ) -> Optional[ProbeResult]:
        """Execute a benign test action (as named by the action adapter).

        This is the bridge from the adapter's allow-listed action string to the
        backend primitives. It accepts **only** the benign test-action names and
        raises :class:`ValueError` for anything else — a second line of defence
        after the adapter. Read actions return their :class:`ProbeResult`;
        change actions (restart/isolate/reconnect) return ``None``.
        """
        # Imported here to keep backend independent of the adapter at import time.
        from nemesis.sandbox.action_adapter import TestAction

        if test_action == TestAction.CHECK_HEALTH:
            return self.probe(container, "service_health")
        if test_action == TestAction.COLLECT_LOGS:
            return self.probe(container, "collect_logs")
        if test_action == TestAction.RESTART_SERVICE:
            self.restart_target(container, network)
            return None
        if test_action == TestAction.ISOLATE_CONTAINER:
            self.disconnect_target(container, network)
            return None
        if test_action == TestAction.RECONNECT_CONTAINER:
            self.connect_target(container, network)
            return None
        if test_action == TestAction.NOOP:
            return None
        raise ValueError(
            f"{test_action!r} is not a benign test action; refusing to execute it"
        )


__all__ = [
    "BENIGN_PROBES",
    "SandboxUnavailable",
    "SandboxNetwork",
    "SandboxContainer",
    "ProbeResult",
    "SandboxBackend",
]
