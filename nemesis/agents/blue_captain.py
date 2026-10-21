"""BlueCaptain — the defensive team captain.

The Captain doesn't take low-level actions itself. It commands a squad of blue
specialists and, each step, decides *which* one should act based on how the
network looks right now:

    - nothing suspicious, early game  -> Trapper   (pre-place decoys)
    - nothing suspicious, later       -> Sentinel  (keep analysing / watching)
    - suspicious activity seen        -> Sentinel  (analyse the odd host to confirm)
    - confirmed compromise            -> Medic     (evict the attacker)
    - compromise across many subnets  -> Warden    (contain), then Medic

Because blue observations are *deltas* (a signal shows up once, then the host
goes quiet), the Captain keeps its own running memory of which hosts look
suspicious or confirmed-compromised, and points the chosen specialist straight
at the relevant host via ``preferred_targets``.

Specialists are *strict*: if the one the Captain tries has no useful move this
step it returns ``Sleep``, and the Captain falls through to the next option.
"""

from __future__ import annotations

from typing import Dict, List

from CybORG.Simulator.Actions import Sleep

from nemesis.agents.base import NemesisAgent
from nemesis.agents.medic import BlueMedic
from nemesis.agents.sentinel import BlueSentinel
from nemesis.agents.trapper import BlueTrapper
from nemesis.agents.warden import BlueWarden, subnet_of

# A file this "dense" is treated as malware — matches CybORG's Analyse threshold.
_MALWARE_DENSITY = 0.9
# How many opening steps the Captain spends hardening with decoys before it
# settles into a watch-and-respond posture.
_SETUP_STEPS = 10


class BlueCaptain(NemesisAgent):
    """Routes each step to the blue specialist that best fits the current threat."""

    role = "captain"

    def __init__(self, name: str = "blue_agent_0", np_random=None):
        super().__init__(name=name, epsilon=0.0, np_random=np_random)
        # The squad under command. All share the Captain's name, because CybORG
        # attributes whatever action we return to this one agent slot.
        self.sentinel = BlueSentinel(name, np_random=self.np_random)
        self.medic = BlueMedic(name, np_random=self.np_random)
        self.trapper = BlueTrapper(name, np_random=self.np_random)
        self.warden = BlueWarden(name, np_random=self.np_random)
        self.squad = [self.sentinel, self.medic, self.trapper, self.warden]

        # Running memory across the episode (observations are deltas).
        self.step = 0
        self.suspicious: set = set()   # hosts with odd activity, unconfirmed
        self.compromised: set = set()  # hosts confirmed compromised by Analyse
        self.last_role: str = None     # which specialist acted last (for inspection)

    # ------------------------------------------------------------------ #
    def get_action(self, observation: dict, action_space):
        self.step += 1
        self._update_memory(observation)

        for specialist in self._plan():
            action = specialist.get_action(observation, action_space)
            if not isinstance(action, Sleep):
                self.last_role = specialist.role
                self._note_handled(action)
                return action

        self.last_role = None
        return Sleep()

    # ------------------------------------------------------------------ #
    # Threat assessment
    # ------------------------------------------------------------------ #
    def _update_memory(self, observation: dict) -> None:
        """Fold this step's observation into the running threat picture."""
        for host, data in observation.items():
            if host in ("success", "action", "message") or not isinstance(data, dict):
                continue
            if self._looks_compromised(data):
                self.compromised.add(host)
                self.suspicious.discard(host)
            elif data.get("Processes"):
                # Monitor caught connection/process activity — worth a closer look.
                if host not in self.compromised:
                    self.suspicious.add(host)

    @staticmethod
    def _looks_compromised(host_data: dict) -> bool:
        """True when Analyse reported a malware-dense file on the host."""
        for file in host_data.get("Files", []) or []:
            density = file.get("Density", file.get("density", 0)) or 0
            if density >= _MALWARE_DENSITY:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Planning: order the specialists and aim them
    # ------------------------------------------------------------------ #
    def _plan(self) -> List[NemesisAgent]:
        """Reset aim, then return the specialists to try, best fit first."""
        for member in self.squad:
            member.preferred_targets = []

        if self.compromised:
            targets = sorted(self.compromised)
            self.medic.preferred_targets = targets
            self.warden.preferred_targets = targets
            self.sentinel.preferred_targets = targets
            # Spread across subnets => contain first, then clean up.
            if len({subnet_of(h) for h in targets} - {None}) >= 2:
                return [self.warden, self.medic, self.sentinel]
            return [self.medic, self.sentinel, self.warden]

        if self.suspicious:
            self.sentinel.preferred_targets = sorted(self.suspicious)
            return [self.sentinel, self.trapper]

        # Quiet network: harden early, stay watchful afterwards.
        if self.step <= _SETUP_STEPS:
            return [self.trapper, self.sentinel]
        return [self.sentinel, self.trapper]

    def _note_handled(self, action) -> None:
        """Once we act on a host, stop tracking it until it re-triggers."""
        host = getattr(action, "hostname", None)
        if host is not None:
            self.suspicious.discard(host)
            if self.last_role == "remediate":
                self.compromised.discard(host)

    # ------------------------------------------------------------------ #
    def end_episode(self) -> None:
        self.step = 0
        self.suspicious.clear()
        self.compromised.clear()
        self.last_role = None
        for member in self.squad:
            member.end_episode()
