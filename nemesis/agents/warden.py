"""BlueWarden — the containment specialist.

Class fantasy: the warden who slams the gates. When red has a foothold in one
zone, the Warden limits the blast radius with firewall-style rules:
``BlockTrafficZone`` severs traffic from the attacker's subnet toward the zone
it's reaching for; ``AllowTrafficZone`` reopens a path once the threat is dealt
with.

These actions take *subnet names*, which CybORG does not enumerate in the action
space, so the Warden builds them itself — deriving subnet names from the hosts it
can see and preferring to wall off the subnet of whatever host the captain has
flagged. If it can't identify two distinct subnets to separate, it passes.
"""

from __future__ import annotations

import re
from typing import List, Optional

from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)

from nemesis.agents.base import NemesisAgent

# Hostnames look like "restricted_zone_b_subnet_server_host_5"; the subnet name
# is everything before the trailing "_server_host_N" / "_user_host_N".
_HOST_SUFFIX = re.compile(r"_(?:server|user)_host_\d+$")


def subnet_of(hostname: str) -> Optional[str]:
    """Return the subnet name embedded in a hostname, or None if it doesn't fit."""
    stripped = _HOST_SUFFIX.sub("", hostname)
    return stripped if stripped != hostname else None


class BlueWarden(NemesisAgent):
    """Containment-focused blue specialist: block/allow traffic between subnets."""

    role = "contain"
    strict_priority = True
    action_priority = [
        "BlockTrafficZone",
        "AllowTrafficZone",
    ]

    def __init__(self, name: str = "blue_agent_0", epsilon: float = 0.0, np_random=None):
        super().__init__(name=name, epsilon=epsilon, np_random=np_random)

    def _instantiate(self, action_class, action_space: dict):
        """Build traffic actions ourselves; defer everything else to the base."""
        name = action_class.__name__
        if name not in ("BlockTrafficZone", "AllowTrafficZone"):
            return super()._instantiate(action_class, action_space)

        from_subnet, to_subnet = self._choose_subnets(action_space)
        if from_subnet is None or to_subnet is None or from_subnet == to_subnet:
            return None  # not enough of a picture to act; let the agent pass.

        session = self._single_legal(action_space, "session", default=0)
        agent = self._single_legal(action_space, "agent", default=self.name)
        return action_class(
            session=session, agent=agent, from_subnet=from_subnet, to_subnet=to_subnet
        )

    def _choose_subnets(self, action_space: dict):
        """Pick (from_subnet, to_subnet): wall off a flagged host's subnet."""
        known = self._known_subnets(action_space)
        if len(known) < 2:
            return None, None

        # from_subnet: the subnet of a captain-flagged host, if any.
        flagged = [subnet_of(t) for t in self.preferred_targets if isinstance(t, str)]
        flagged = [s for s in flagged if s in known]
        from_subnet = flagged[0] if flagged else self.np_random.choice(known)

        others = [s for s in known if s != from_subnet]
        to_subnet = self.np_random.choice(others) if others else None
        return from_subnet, to_subnet

    def _known_subnets(self, action_space: dict) -> List[str]:
        """Subnet names we can name, drawn from the action space's subnet list."""
        subnets = {
            str(s) for s in action_space.get("subnet", {}) if isinstance(s, str)
        }
        # Fall back to parsing hostnames if the subnet list came through empty.
        for host in action_space.get("hostname", {}):
            s = subnet_of(str(host))
            if s:
                subnets.add(s)
        return sorted(subnets)

    @staticmethod
    def _single_legal(action_space: dict, key: str, default):
        legal = [v for v, ok in action_space.get(key, {}).items() if ok]
        return legal[0] if legal else default
