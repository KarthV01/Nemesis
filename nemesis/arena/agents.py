"""Agent-class adapters used to construct honest arena match formats."""

from __future__ import annotations


def single_active_agent_class(delegate_class, active_name: str):
    """Return a CybORG agent class that activates exactly one named slot.

    Enterprise CC4 always creates its normal team slots. For a literal 1v1 we
    keep those structural slots but make every non-selected slot a SleepAgent.
    The selected slot delegates to the requested specialist policy.
    """
    from CybORG.Agents import SleepAgent
    from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

    class SingleActiveAgent(BaseAgent):
        def __init__(self, name=None, np_random=None, **kwargs):
            super().__init__(name)
            if name == active_name:
                try:
                    self.delegate = delegate_class(name=name, np_random=np_random)
                except TypeError:
                    self.delegate = delegate_class(name=name)
            else:
                self.delegate = SleepAgent(name=name)

        def get_action(self, observation, action_space):
            return self.delegate.get_action(observation, action_space)

        def train(self, results):
            train = getattr(self.delegate, "train", None)
            if train is not None:
                return train(results)

        def end_episode(self):
            end = getattr(self.delegate, "end_episode", None)
            if end is not None:
                return end()

        def set_initial_values(self, action_space, observation):
            initial = getattr(self.delegate, "set_initial_values", None)
            if initial is not None:
                return initial(action_space, observation)

    SingleActiveAgent.__name__ = f"SingleActive{delegate_class.__name__}"
    return SingleActiveAgent


__all__ = ["single_active_agent_class"]
