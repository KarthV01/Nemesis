"""Nemesis agents: the red (attacker) and blue (defender) policies that spar.

Three tiers:

* **Generalists** — ``NemesisRedAgent`` / ``NemesisBlueAgent``: single greedy
  policies that cover a whole side. Good baselines.
* **Specialists** — each trained to be good at one thing, named for its role:
    - Red: ``RedScout`` (recon), ``RedBreacher`` (exploit),
      ``RedInfiltrator`` (privilege escalation), ``RedSaboteur`` (impact).
    - Blue: ``BlueSentinel`` (detect), ``BlueMedic`` (remediate),
      ``BlueTrapper`` (deception), ``BlueWarden`` (containment).
* **Captains** — ``RedCaptain`` / ``BlueCaptain``: command a squad of
  specialists and pick the right one for the situation each step.
"""

from nemesis.agents.base import NemesisAgent
from nemesis.agents.red import NemesisRedAgent
from nemesis.agents.blue import NemesisBlueAgent

# Red specialists
from nemesis.agents.scout import RedScout
from nemesis.agents.breacher import RedBreacher
from nemesis.agents.infiltrator import RedInfiltrator
from nemesis.agents.saboteur import RedSaboteur

# Blue specialists
from nemesis.agents.sentinel import BlueSentinel
from nemesis.agents.medic import BlueMedic
from nemesis.agents.trapper import BlueTrapper
from nemesis.agents.warden import BlueWarden

# Captains
from nemesis.agents.red_captain import RedCaptain
from nemesis.agents.blue_captain import BlueCaptain

__all__ = [
    "NemesisAgent",
    "NemesisRedAgent",
    "NemesisBlueAgent",
    "RedScout",
    "RedBreacher",
    "RedInfiltrator",
    "RedSaboteur",
    "BlueSentinel",
    "BlueMedic",
    "BlueTrapper",
    "BlueWarden",
    "RedCaptain",
    "BlueCaptain",
]
