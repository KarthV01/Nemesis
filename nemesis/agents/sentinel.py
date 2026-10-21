"""BlueSentinel — the detection specialist.

Class fantasy: the watcher on the wall. The Sentinel's job is awareness, not
response. Every blue agent already runs ``Monitor`` automatically each step to
collect network/process events; the Sentinel spends its *action* on ``Analyse``,
the deeper scan that inspects a host's files for the tell-tale density of
malware and confirms whether a suspicious signal is a real compromise.

It is the captain's eyes: point it at a host that looks off and it tells you
whether to call in the Medic.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class BlueSentinel(NemesisAgent):
    """Detection-focused blue specialist: analyse hosts to confirm compromise."""

    role = "detect"
    strict_priority = True
    action_priority = [
        "Analyse",   # deep per-host scan that confirms a compromise
        "Monitor",   # explicit broad sweep (also runs automatically each step)
    ]

    def __init__(self, name: str = "blue_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
