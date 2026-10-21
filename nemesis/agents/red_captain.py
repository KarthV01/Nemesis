"""RedCaptain — the offensive team captain.

Like its blue counterpart, the Captain takes no low-level actions itself. It
commands a squad of red specialists and, each step, fields the one whose phase
of the attack is furthest along but still actionable:

    - privileged foothold on a valuable host -> Saboteur    (cause impact)
    - low-privilege foothold                 -> Infiltrator (escalate it)
    - a discovered but unexploited service    -> Breacher    (exploit it)
    - otherwise                               -> Scout       (find more targets)

The Captain reads its own sessions out of the observation to know which hosts it
holds and at what privilege, and aims the Saboteur and Infiltrator at the right
hosts via ``preferred_targets``. Specialists are *strict*: because CybORG only
makes an action legal once its prerequisites are met (you can't Impact a host you
don't hold at privilege), a specialist with nothing to do returns ``Sleep`` and
the Captain falls through to the next phase down.
"""

from __future__ import annotations

from typing import List

from CybORG.Simulator.Actions import Sleep

from nemesis.agents.base import NemesisAgent
from nemesis.agents.breacher import RedBreacher
from nemesis.agents.infiltrator import RedInfiltrator
from nemesis.agents.saboteur import RedSaboteur
from nemesis.agents.scout import RedScout

# Session usernames that mean we hold the host at elevated privilege.
_PRIVILEGED = {"root", "SYSTEM", "administrator", "Administrator"}


class RedCaptain(NemesisAgent):
    """Routes each step to the red specialist whose attack phase fits best."""

    role = "captain"

    def __init__(self, name: str = "red_agent_0", np_random=None):
        super().__init__(name=name, epsilon=0.0, np_random=np_random)
        self.scout = RedScout(name, np_random=self.np_random)
        self.breacher = RedBreacher(name, np_random=self.np_random)
        self.infiltrator = RedInfiltrator(name, np_random=self.np_random)
        self.saboteur = RedSaboteur(name, np_random=self.np_random)
        self.squad = [self.saboteur, self.infiltrator, self.breacher, self.scout]

        # Running memory of what we hold (observations can be partial).
        self.footholds: set = set()   # hosts held at user privilege
        self.privileged: set = set()  # hosts held at root/SYSTEM
        self.last_role: str = None

    # ------------------------------------------------------------------ #
    def get_action(self, observation: dict, action_space):
        self._update_holdings(observation)

        for specialist in self._plan():
            action = specialist.get_action(observation, action_space)
            if not isinstance(action, Sleep):
                self.last_role = specialist.role
                return action

        self.last_role = None
        return Sleep()

    # ------------------------------------------------------------------ #
    def _update_holdings(self, observation: dict) -> None:
        """Track which hosts we hold, and at what privilege, from our sessions."""
        for host, data in observation.items():
            if host in ("success", "action", "message") or not isinstance(data, dict):
                continue
            for session in data.get("Sessions", []) or []:
                if session.get("agent") != self.name:
                    continue
                if session.get("username") in _PRIVILEGED:
                    self.privileged.add(host)
                    self.footholds.discard(host)
                else:
                    if host not in self.privileged:
                        self.footholds.add(host)

    def _plan(self) -> List[NemesisAgent]:
        """Reset aim, then order specialists from most- to least-advanced phase."""
        for member in self.squad:
            member.preferred_targets = []

        if self.privileged:
            self.saboteur.preferred_targets = sorted(self.privileged)
        if self.footholds:
            self.infiltrator.preferred_targets = sorted(self.footholds)

        # Always offered in kill-chain order; strict specialists self-skip when
        # their phase isn't actionable yet, so the furthest-along one that *can*
        # move is the one that acts.
        return [self.saboteur, self.infiltrator, self.breacher, self.scout]

    # ------------------------------------------------------------------ #
    def end_episode(self) -> None:
        self.footholds.clear()
        self.privileged.clear()
        self.last_role = None
        for member in self.squad:
            member.end_episode()
