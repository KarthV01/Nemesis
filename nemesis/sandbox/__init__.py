"""Sandbox evaluation — the sim-to-reality adapter for trained Nemesis policies.

This is the only stage that touches a real container runtime, and it is strictly
defensive and bounded by construction (see :mod:`nemesis.sandbox.backend` and
:mod:`nemesis.sandbox.action_adapter`):

* targets run **only** on an isolated, internal Docker network (no gateway, no
  internet) and are intentionally-vulnerable **lab** apps we define locally;
* a trained blue policy's abstract decision is passed through an **allow-list**
  (:class:`~nemesis.sandbox.action_adapter.ActionAdapter`) that maps it to one of
  a few benign test actions and **refuses + logs** everything else — there is no
  path to real exploitation, lateral movement, or offensive payloads here;
* outcomes are verified **independently** (:func:`~nemesis.sandbox.verify.verify`)
  and reported as the same :class:`~nemesis.platform.metrics.EpisodeMetrics` and
  events every other Nemesis evaluator uses.

The whole stage cleanly **skips** when no container runtime is available: the
backend raises :class:`~nemesis.sandbox.backend.SandboxUnavailable`, which callers
treat as no-op. Importing this package pulls in no Docker or CybORG dependency.
"""

from nemesis.sandbox.backend import (
    BENIGN_PROBES,
    ProbeResult,
    SandboxBackend,
    SandboxContainer,
    SandboxNetwork,
    SandboxUnavailable,
)
from nemesis.sandbox.action_adapter import (
    AdaptResult,
    ActionAdapter,
    DEFAULT_ALLOW_LIST,
    TestAction,
    adapt,
)
from nemesis.sandbox.verify import StepOutcome, VerifyConfig, summary_event, verify
from nemesis.sandbox.docker_backend import DockerBackend

__all__ = [
    # backend
    "SandboxBackend",
    "DockerBackend",
    "SandboxNetwork",
    "SandboxContainer",
    "ProbeResult",
    "SandboxUnavailable",
    "BENIGN_PROBES",
    # action adapter
    "ActionAdapter",
    "AdaptResult",
    "TestAction",
    "DEFAULT_ALLOW_LIST",
    "adapt",
    # verify
    "verify",
    "StepOutcome",
    "VerifyConfig",
    "summary_event",
]
