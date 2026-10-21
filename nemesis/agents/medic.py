"""BlueMedic — the remediation specialist.

Class fantasy: the medic who patches the team back up. Once a host is confirmed
compromised, the Medic evicts the attacker. It reaches for the cheap option
first — ``Remove`` kills the red session and its malicious processes — and keeps
the heavier ``Restore`` (a full reimage that also disrupts the host's own
services, and so costs reward) as the fallback for infections ``Remove`` can't
shake, such as a privileged foothold.

The captain hands the Medic the specific host to treat via ``preferred_targets``.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class BlueMedic(NemesisAgent):
    """Remediation-focused blue specialist: evict attackers from compromised hosts."""

    role = "remediate"
    strict_priority = True
    action_priority = [
        "Remove",    # cheap: kill the red session / malicious processes
        "Restore",   # expensive: full reimage, for footholds Remove can't clear
    ]

    def __init__(self, name: str = "blue_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
