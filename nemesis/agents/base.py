"""Shared machinery for Nemesis agents.

Both the red and blue agents are, for now, *heuristic* policies: they rank the
action types they care about and, each step, take the highest-ranked action the
simulator currently says is legal. That is deliberately simple so you can read a
policy top-to-bottom and watch it behave before we bolt on a learned policy.

The CybORG action space arrives as a dict shaped like::

    {
        "action": {SomeActionClass: True, OtherActionClass: False, ...},
        "session": {0: True, ...},
        "hostname": {"restricted_zone_a_subnet_user_host_0": True, ...},
        ...  # one dict of {value: is_valid} per action parameter
    }

``get_action`` must return an *instantiated* action object, e.g.
``ExploitRemoteService(session=0, agent="red_agent_0", ip_address=...)``.
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Dict, List, Union

_EMPTY = Parameter.empty

from gymnasium import Space

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Simulator.Actions import Sleep


class NemesisAgent(BaseAgent):
    """Base for Nemesis heuristic agents.

    Subclasses only need to provide ``action_priority`` — an ordered list of
    action *class names* (strings) from most to least preferred. Everything
    else (finding a legal action, filling in its parameters, the RL-ready
    ``train`` hook) lives here.

    Parameters
    ----------
    name : str
        Agent name, e.g. ``"red_agent_0"`` — must match the name the scenario
        assigns, because actions are attributed to it.
    epsilon : float
        Probability of ignoring the priority list and taking a random legal
        action instead. Pure exploration knob; set to 0.0 for deterministic play.
    np_random : optional
        Seeded numpy RNG supplied by CybORG; leave as ``None`` otherwise.
    """

    #: Ordered, most-preferred first. Subclasses override this.
    action_priority: List[str] = []

    def __init__(self, name: str = None, epsilon: float = 0.0, np_random=None):
        super().__init__(name, np_random)
        self.epsilon = epsilon
        # Maps each action class -> its constructor parameters, filled lazily by
        # set_initial_values() the first time CybORG hands us an action space.
        self.action_params: Dict[type, object] = None

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def get_action(self, observation: dict, action_space: Union[Space, dict]):
        """Pick this step's action from the legal options.

        Strategy: with probability ``epsilon`` act randomly; otherwise walk the
        priority list and take the first action type that is currently legal.
        Falls back to ``Sleep`` when nothing is available.
        """
        if isinstance(action_space, Space):
            # Flat/vectorised space (e.g. a wrapped gym env for an RL policy).
            # A heuristic has nothing clever to say here, so sample uniformly.
            return action_space.sample()
        if not isinstance(action_space, dict):
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        if self.action_params is None:
            self.set_initial_values(action_space, observation)

        legal = {cls: ok for cls, ok in action_space["action"].items() if ok}

        explore = self.np_random.random() < self.epsilon
        ordered = self._random_order(legal) if explore else self._by_priority(legal)

        for action_class in ordered:
            built = self._instantiate(action_class, action_space)
            if built is not None:
                return built
        return Sleep()

    def _by_priority(self, legal: Dict[type, bool]) -> List[type]:
        """Legal action classes, sorted by this agent's preference list.

        Anything not named in ``action_priority`` sorts last (kept so the agent
        still does *something* if the scenario offers an unexpected action).
        """
        rank = {name: i for i, name in enumerate(self.action_priority)}
        last = len(rank)
        return sorted(legal, key=lambda cls: rank.get(cls.__name__, last))

    def _random_order(self, legal: Dict[type, bool]) -> List[type]:
        classes = list(legal)
        self.np_random.shuffle(classes)
        return classes

    def _instantiate(self, action_class: type, action_space: dict):
        """Build an action, choosing a random *legal* value for each parameter.

        Returns ``None`` if any required parameter has no legal value, so the
        caller can move on to the next-preferred action.
        """
        kwargs = {}
        for param_name, param in self.action_params[action_class].items():
            if param_name not in action_space:
                # The scenario offers no values for this parameter. Fine if the
                # constructor has a default; otherwise we can't build it, so skip.
                if param.default is _EMPTY:
                    return None
                continue
            options = [v for v, ok in action_space[param_name].items() if ok]
            if not options:
                return None
            kwargs[param_name] = self.np_random.choice(options)
        return action_class(**kwargs)

    # ------------------------------------------------------------------ #
    # BaseAgent lifecycle hooks
    # ------------------------------------------------------------------ #
    def set_initial_values(self, action_space, observation):
        """Cache the constructor signature of every available action class."""
        if isinstance(action_space, dict):
            self.action_params = {
                cls: signature(cls).parameters for cls in action_space["action"]
            }

    def train(self, results):
        """Learning hook — a no-op for now.

        When we move from heuristics to a learned policy, this is where a step
        of the RL update goes (``results`` carries the reward and next
        observation for the action we just took). Left empty so these agents
        can already run in self-play unchanged.
        """
        pass

    def end_episode(self):
        """Reset any per-episode memory. Nothing to clear yet."""
        pass
