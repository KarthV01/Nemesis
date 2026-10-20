"""Nemesis agents: the red (attacker) and blue (defender) policies that spar."""

from nemesis.agents.base import NemesisAgent
from nemesis.agents.red import NemesisRedAgent
from nemesis.agents.blue import NemesisBlueAgent

__all__ = ["NemesisAgent", "NemesisRedAgent", "NemesisBlueAgent"]
