"""Independent outcome verification for a sandbox episode -> EpisodeMetrics.

The point of sandbox evaluation is not to trust the policy's own view of how it
did, but to **independently** measure what actually happened to the real lab
targets and express it in the same :class:`~nemesis.platform.metrics.EpisodeMetrics`
vector every other Nemesis evaluator produces, so a sandbox run is directly
comparable to a simulated one.

"Independent" means the numbers come from observed probe outcomes
(:class:`~nemesis.sandbox.backend.ProbeResult`), not from the agent's claims:

* ``defender_availability`` — fraction of steps the target's service was up,
  from health probes.
* ``attack_success_rate`` — fraction of the steps where an intrusion was present
  that the service was actually degraded (i.e. the attack "landed").
* ``false_positives`` — disruptive responses (restart/isolate) taken when there
  was no intrusion to justify them.
* ``blue_win`` — availability held, every present intrusion was detected, and no
  false positives.

``blue_reward`` is a transparent, documented proxy derived from those outcomes
(there is no simulator reward out here); the weights live in
:class:`VerifyConfig` so the proxy is explicit rather than magic.

This module is pure logic — no Docker, no CybORG, no network — so it is fully
unit-testable and can also re-score a logged run offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nemesis.platform.events import Event, EventKind
from nemesis.platform.metrics import EpisodeMetrics
from nemesis.sandbox.action_adapter import DISRUPTIVE_ACTIONS, TestAction


@dataclass
class StepOutcome:
    """What a probe observed at one evaluation step.

    Parameters
    ----------
    step : int
        Step index within the episode.
    service_up : bool
        Did the health probe see the target's service responding?
    intrusion_present : bool
        Ground truth for the lab this step: because *we* launch the known
        intentionally-vulnerable target and trigger it, we know when an intrusion
        is actually present. Used to score detection and false positives.
    intrusion_detected : bool
        Did an independent signal (the defender service / log probe) flag it?
    action : str or TestAction, optional
        The benign test action the harness applied this step, if any.
    """

    step: int
    service_up: bool
    intrusion_present: bool = False
    intrusion_detected: bool = False
    action: Optional[object] = None


@dataclass
class VerifyConfig:
    """Thresholds and the (explicit) weights of the derived ``blue_reward`` proxy."""

    availability_target: float = 0.95
    reward_availability_weight: float = 1.0
    reward_false_positive_penalty: float = 0.5
    reward_detection_bonus: float = 1.0
    reward_compromise_penalty: float = 1.0


def _is_disruptive(action: object) -> bool:
    if action is None:
        return False
    if isinstance(action, TestAction):
        return action in DISRUPTIVE_ACTIONS
    # Compare by value so a plain string ("restart_service") also works.
    return any(action == a.value for a in DISRUPTIVE_ACTIONS)


def verify(
    steps: List[StepOutcome],
    *,
    episode: int = 0,
    config: Optional[VerifyConfig] = None,
    extra: Optional[Dict[str, object]] = None,
) -> EpisodeMetrics:
    """Turn a sequence of observed step outcomes into :class:`EpisodeMetrics`.

    Raises :class:`ValueError` on an empty episode (there is nothing to verify).
    """
    if not steps:
        raise ValueError("cannot verify an empty episode (no step outcomes)")
    cfg = config or VerifyConfig()

    total = len(steps)
    up_steps = sum(1 for s in steps if s.service_up)
    availability = up_steps / total

    present = [s for s in steps if s.intrusion_present]
    detected_present = [s for s in present if s.intrusion_detected]
    # Attack "success" = intrusion present and the service was degraded that step.
    compromised = sum(1 for s in present if not s.service_up)
    attack_success_rate = (compromised / len(present)) if present else 0.0

    false_positives = sum(
        1 for s in steps if _is_disruptive(s.action) and not s.intrusion_present
    )

    all_present_detected = len(detected_present) == len(present)
    detection_rate = (len(detected_present) / len(present)) if present else None

    blue_win = (
        availability >= cfg.availability_target
        and all_present_detected
        and false_positives == 0
    )

    blue_reward = (
        cfg.reward_availability_weight * availability
        - cfg.reward_false_positive_penalty * false_positives
        + (cfg.reward_detection_bonus if (present and all_present_detected) else 0.0)
        - cfg.reward_compromise_penalty * attack_success_rate
    )

    extras: Dict[str, object] = {
        "source": "sandbox",
        "intrusion_steps": len(present),
        "detection_rate": detection_rate,
    }
    if extra:
        extras.update(extra)

    return EpisodeMetrics(
        episode=episode,
        steps=total,
        blue_reward=float(blue_reward),
        attack_success_rate=attack_success_rate,
        defender_availability=availability,
        false_positives=false_positives,
        blue_win=blue_win,
        extra=extras,
    )


def summary_event(metrics: EpisodeMetrics) -> Event:
    """A one-line :class:`~nemesis.platform.events.Event` summarising the result.

    Lets a sandbox run feed the same dashboard/commentator stream as everything
    else. Emit it through an :class:`~nemesis.platform.events.EventLog`.
    """
    text = (
        f"sandbox ep{metrics.episode}: "
        f"availability={_fmt(metrics.defender_availability)}, "
        f"attack_success={_fmt(metrics.attack_success_rate)}, "
        f"false_positives={metrics.false_positives}, "
        f"win={metrics.blue_win}"
    )
    return Event(
        kind=EventKind.METRIC,
        text=text,
        episode=metrics.episode,
        extra={
            "defender_availability": metrics.defender_availability,
            "attack_success_rate": metrics.attack_success_rate,
            "false_positives": metrics.false_positives,
            "blue_win": metrics.blue_win,
            "source": "sandbox",
        },
    )


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


__all__ = ["StepOutcome", "VerifyConfig", "verify", "summary_event"]
