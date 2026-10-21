"""BlueTrapper — the deception specialist.

Class fantasy: the trapper who seeds the ground with snares. Rather than react to
red, the Trapper shapes the board ahead of time: ``DeployDecoy`` stands up fake
services on a host so that when red tries to exploit one, the attempt fails and
*announces red's presence* instead of yielding a foothold. ``Misinform`` feeds
red misleading observations where the scenario supports it.

Decoys are most valuable before contact, so the captain leans on the Trapper
during the quiet early game to harden hosts proactively.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class BlueTrapper(NemesisAgent):
    """Deception-focused blue specialist: pre-place decoys to catch and slow red."""

    role = "deceive"
    strict_priority = True
    action_priority = [
        "DeployDecoy",   # stand up a fake service that trips exploiting red
        "Misinform",     # feed red misleading observations, where supported
    ]

    def __init__(self, name: str = "blue_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
