"""RedBreacher — the exploitation specialist.

Class fantasy: the assault/entry specialist who forces the door. Given a host
whose services the Scout has already fingerprinted, the Breacher's whole job is
to turn a known service into a foothold via ``ExploitRemoteService``. It does
one thing and does it well; when there's nothing left to exploit it passes the
turn back to the captain.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class RedBreacher(NemesisAgent):
    """Exploitation-only red specialist: turn discovered services into footholds."""

    role = "exploit"
    strict_priority = True
    action_priority = [
        "ExploitRemoteService",
    ]

    def __init__(self, name: str = "red_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
