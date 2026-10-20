"""Nemesis red agent — the attacker.

A simple, greedy kill-chain climber. Each step it takes the most "advanced"
action the simulator will currently allow, which naturally walks a host from
reconnaissance toward impact:

    discover hosts -> discover services -> exploit -> escalate privileges -> impact

It does not plan or remember across steps yet — it just always reaches for the
furthest-along action available. Because actions like ``PrivilegeEscalate`` and
``Impact`` only become legal once their prerequisites are met, this ordering is
enough to produce coherent, progressing attacks that the blue agent must react
to. All actions are simulated inside CybORG; nothing touches a real network.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class NemesisRedAgent(NemesisAgent):
    """Greedy offensive policy that always pushes the attack as far as it can.

    The priority list is the whole strategy. Read it top-to-bottom as "if I can
    cause impact, do that; otherwise escalate; otherwise exploit; otherwise keep
    looking around." Swap or reorder entries to change red's temperament.
    """

    action_priority = [
        # --- Objective: do damage once we have a privileged foothold. ---
        "Impact",
        "DegradeServices",
        # --- Deepen access. ---
        "PrivilegeEscalate",
        "ExploitRemoteService",
        # --- Reconnaissance: find new hosts and the services to exploit. ---
        "AggressiveServiceDiscovery",
        "StealthServiceDiscovery",
        "DiscoverNetworkServices",
        "DiscoverRemoteSystems",
        # --- Situational awareness: spot blue's decoys so we waste less effort. ---
        "DiscoverDeception",
    ]

    def __init__(self, name: str = "red_agent_0", epsilon: float = 0.1, np_random=None):
        # A little exploration by default so self-play doesn't collapse into one
        # rigid attack path; drop epsilon to 0.0 to watch the pure heuristic.
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
