"""RedInfiltrator — the privilege-escalation specialist.

Class fantasy: the rogue who, once inside, quietly becomes an administrator. The
Breacher gets a low-privilege foothold; the Infiltrator deepens it into root/
SYSTEM control of the host via ``PrivilegeEscalate``. That elevated access is
what later lets the Saboteur actually cause impact, so the Infiltrator is the
bridge between getting in and doing damage.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class RedInfiltrator(NemesisAgent):
    """Escalation-only red specialist: turn a user foothold into privileged access."""

    role = "escalate"
    strict_priority = True
    action_priority = [
        "PrivilegeEscalate",
    ]

    def __init__(self, name: str = "red_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
