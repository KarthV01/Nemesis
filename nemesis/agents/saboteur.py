"""RedSaboteur — the impact specialist.

Class fantasy: the demolitions expert who converts access into results. Once the
Infiltrator holds privileged access on a valuable host, the Saboteur spends it:
``Impact`` to disrupt the mission-critical service outright, or ``DegradeServices``
to quietly erode it. This is where red's score actually comes from, so the
captain saves the Saboteur for hosts that are both owned and worth hitting.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class RedSaboteur(NemesisAgent):
    """Impact-only red specialist: spend privileged access to disrupt services."""

    role = "impact"
    strict_priority = True
    action_priority = [
        "Impact",           # full disruption of a mission-critical service
        "DegradeServices",  # quieter, partial degradation
    ]

    def __init__(self, name: str = "red_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
