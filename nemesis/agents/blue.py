"""Nemesis blue agent — the defender.

A simple monitor-and-respond policy. Its instinct is to investigate and clean up
anything suspicious, and to keep watching the network when nothing needs doing:

    analyse a suspect host -> remove the intruder -> restore if compromised
    -> lay decoys to slow red down -> otherwise just monitor

Like the red agent, it is greedy and memoryless for now: it takes the
highest-priority legal action each step. ``Monitor`` sits at the bottom as the
safe default, so when the network looks clean blue simply keeps its eyes open.
Everything happens inside the CybORG simulator.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class NemesisBlueAgent(NemesisAgent):
    """Reactive defensive policy: investigate, evict, recover, deceive, watch.

    The priority list encodes a cautious defender. ``Restore`` is expensive
    (it disrupts the host's services, which costs reward), so it sits below the
    cheaper ``Remove``. Reorder entries to make blue more aggressive or more
    conservative about restoring.
    """

    action_priority = [
        # --- Investigate first so responses are informed, not blind. ---
        "Analyse",
        # --- Evict the attacker. Remove is cheap; Restore is a heavier reset. ---
        "Remove",
        "Restore",
        # --- Slow red down with deception. ---
        "DeployDecoy",
        "Misinform",
        # --- Traffic control as a blunt containment tool. ---
        "BlockTrafficZone",
        "AllowTrafficZone",
        # --- Default: keep watching. Always legal, so this is the safety net. ---
        "Monitor",
    ]

    def __init__(self, name: str = "blue_agent_0", epsilon: float = 0.05, np_random=None):
        # Blue explores less than red — a defender taking random actions is
        # usually just hurting itself. Set to 0.0 for deterministic defence.
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
