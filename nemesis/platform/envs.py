"""One place that builds CybORG environments for every Nemesis stage.

Trainers and evaluators ask here for an environment by an :class:`EnvSpec`
instead of each constructing ``EnterpriseScenarioGenerator`` with its own
choices. That keeps blue training, red training, self-play and evaluation on
the same scenarios and wrappers, so their checkpoints and metrics compare.

CybORG is imported lazily inside the builders, so importing this module (and the
rest of :mod:`nemesis.platform`) does not require the simulator to be installed.

Everything built here runs entirely inside CybORG's **simulated** enterprise
network: agents emit the simulator's abstract action objects only; there are no
real hosts, services, or exploits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

EPISODE_LENGTH = 500

#: Named CybORG scenarios we train/evaluate on. Only CC4's enterprise scenario
#: ships in the vendored submodule today; the alias lets later work swap it.
SCENARIOS = {
    "cc4": "EnterpriseScenarioGenerator",
}


@dataclass
class EnvSpec:
    """A reproducible description of one environment configuration.

    Attributes
    ----------
    scenario : str
        Key into :data:`SCENARIOS`.
    steps : int
        Episode length (CC4 episodes end only on this limit, as a truncation).
    red_agent : str
        Attacker policy class name from ``CybORG.Agents`` (e.g.
        ``"FiniteStateRedAgent"``) used when red is *scripted*.
    blue_agent : str
        Blue policy class name used when blue is *scripted* (``"SleepAgent"`` so
        a learner can drive one agent while the rest idle).
    green_agent : str
        Background user traffic generator.
    wrapper : str
        ``"blue_flat"`` (PettingZoo-style flat vectors, SB3) or ``"rllib_mae"``
        (RLlib ``MultiAgentEnv``) or ``"none"`` (raw CybORG).
    seed : int, optional
        Scenario/env seed for reproducibility.
    """

    scenario: str = "cc4"
    steps: int = EPISODE_LENGTH
    red_agent: str = "FiniteStateRedAgent"
    blue_agent: str = "SleepAgent"
    green_agent: str = "EnterpriseGreenAgent"
    wrapper: str = "blue_flat"
    seed: Optional[int] = None
    #: extra kwargs forwarded to the scenario generator
    scenario_kwargs: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


# --------------------------------------------------------------------------- #
# Convenience specs per stage. Stages may subclass/extend these.
# --------------------------------------------------------------------------- #
def blue_training_spec(**overrides) -> EnvSpec:
    """Blue learns against the scripted red attacker (the current baseline)."""
    return EnvSpec(**overrides)


def evaluation_spec(seed: int = 12345, **overrides) -> EnvSpec:
    """A fixed-seed spec for comparable evaluation across stages."""
    return EnvSpec(seed=seed, **overrides)


def build(spec: EnvSpec):
    """Instantiate the CybORG environment described by ``spec``.

    Returns the wrapped environment (``BlueFlatWrapper`` / ``EnterpriseMAE`` /
    raw ``CybORG``) ready to ``reset()``. Importing CybORG happens here.
    """
    from CybORG import CybORG
    from CybORG.Agents import (
        EnterpriseGreenAgent,
        FiniteStateRedAgent,
        SleepAgent,
    )
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    agent_classes = {
        "FiniteStateRedAgent": FiniteStateRedAgent,
        "SleepAgent": SleepAgent,
        "EnterpriseGreenAgent": EnterpriseGreenAgent,
    }

    def resolve(name: str):
        if name in agent_classes:
            return agent_classes[name]
        # Allow Nemesis agents by name, e.g. "RedCaptain".
        import nemesis.agents as na

        if hasattr(na, name):
            return getattr(na, name)
        raise ValueError(f"unknown agent class {name!r}")

    if spec.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {spec.scenario!r}; known: {sorted(SCENARIOS)}")

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=resolve(spec.blue_agent),
        green_agent_class=resolve(spec.green_agent),
        red_agent_class=resolve(spec.red_agent),
        steps=spec.steps,
        **spec.scenario_kwargs,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=spec.seed)

    if spec.wrapper == "none":
        return cyborg
    if spec.wrapper == "blue_flat":
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        return BlueFlatWrapper(cyborg)
    if spec.wrapper == "rllib_mae":
        from CybORG.Agents.Wrappers import EnterpriseMAE

        return EnterpriseMAE(cyborg)
    raise ValueError(f"unknown wrapper {spec.wrapper!r}")


__all__ = ["EnvSpec", "SCENARIOS", "EPISODE_LENGTH", "build", "blue_training_spec", "evaluation_spec"]
