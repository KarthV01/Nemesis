"""RedScout — the reconnaissance specialist.

Class fantasy: the ranger who walks point. The Scout doesn't break anything; it
builds the map. It finds hosts on reachable subnets, fingerprints the services
running on them, and sniffs out blue's decoys so the rest of the squad doesn't
waste moves attacking traps. It prefers *stealth* discovery over the louder
aggressive variant, trading speed for a smaller footprint.

Hand the Scout the wheel early in an episode, or whenever the squad has run out
of known targets to act on and needs fresh intelligence.
"""

from __future__ import annotations

from nemesis.agents.base import NemesisAgent


class RedScout(NemesisAgent):
    """Recon-only red specialist: discover hosts, map services, spot deception."""

    role = "recon"
    strict_priority = True  # scouts scout; if there's nothing to find, they pass.
    action_priority = [
        "StealthServiceDiscovery",     # quiet service fingerprinting, preferred
        "AggressiveServiceDiscovery",  # faster but noisier fallback
        "DiscoverRemoteSystems",       # find new hosts on a reachable subnet
        "DiscoverDeception",           # identify blue's decoys to avoid them
    ]

    def __init__(self, name: str = "red_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)
