"""Validated API contracts for arena match jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


TEAM_AGENTS = {
    "nemesis": {"red": "RedCaptain", "blue": "BlueCaptain"},
    "baseline": {"red": "NemesisRedAgent", "blue": "NemesisBlueAgent"},
}

ROLE_AGENTS = {
    "red": {
        "recon": "RedScout",
        "exploit": "RedBreacher",
        "escalate": "RedInfiltrator",
        "impact": "RedSaboteur",
    },
    "blue": {
        "detect": "BlueSentinel",
        "remediate": "BlueMedic",
        "deceive": "BlueTrapper",
        "contain": "BlueWarden",
    },
}


@dataclass(frozen=True)
class MatchRequest:
    red_team: str = "nemesis"
    blue_team: str = "nemesis"
    red_checkpoint: Optional[str] = None
    blue_checkpoint: Optional[str] = None
    scenario: str = "cc4"
    format: str = "5v5"
    red_role: Optional[str] = None
    blue_role: Optional[str] = None
    steps: int = 100
    seed: Optional[int] = None
    delay: float = 0.15

    @classmethod
    def from_dict(cls, payload: Dict) -> "MatchRequest":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
        request = cls(**payload)
        request.validate()
        return request

    def validate(self) -> None:
        if self.red_team not in TEAM_AGENTS:
            raise ValueError(f"unknown red team {self.red_team!r}")
        if self.blue_team not in TEAM_AGENTS:
            raise ValueError(f"unknown blue team {self.blue_team!r}")
        if self.scenario != "cc4":
            raise ValueError("only the cc4 scenario is currently available")
        if self.format not in ("5v5", "1v1"):
            raise ValueError("format must be '5v5' or '1v1'")
        if not 1 <= int(self.steps) <= 500:
            raise ValueError("steps must be between 1 and 500")
        if not 0 <= float(self.delay) <= 5:
            raise ValueError("delay must be between 0 and 5 seconds")
        if self.red_checkpoint or self.blue_checkpoint:
            raise ValueError("the selected team has no arena-compatible checkpoint")
        if self.format == "1v1":
            if self.red_role not in ROLE_AGENTS["red"]:
                raise ValueError("a valid red_role is required for 1v1")
            if self.blue_role not in ROLE_AGENTS["blue"]:
                raise ValueError("a valid blue_role is required for 1v1")

    def agent_classes(self):
        if self.format == "1v1":
            return ROLE_AGENTS["red"][self.red_role], ROLE_AGENTS["blue"][self.blue_role]
        return TEAM_AGENTS[self.red_team]["red"], TEAM_AGENTS[self.blue_team]["blue"]

    def as_dict(self) -> Dict:
        return asdict(self)
