"""Nemesis evaluation stage — score a blue policy on CC4 and report metrics.

This package is the shared evaluation seam: the self-play league and the sandbox
stage reuse it so every blue policy (random baseline, scripted Nemesis team, or a
trained checkpoint) is scored the same way, on the metric vector defined in
:mod:`nemesis.platform.metrics`.

Typical use::

    from nemesis.platform import envs
    from nemesis.platform.metrics import summarize
    from nemesis.eval import EvalConfig, evaluate, scripted_blue

    cfg = EvalConfig(spec=envs.evaluation_spec(steps=500), episodes=10)
    episodes = evaluate(scripted_blue(), cfg)
    print(summarize(episodes).as_dict())

See ``docs/evaluation.md`` for how each metric is derived from CybORG CC4 state.
"""

from nemesis.eval.harness import (
    EvalConfig,
    MetricProbe,
    evaluate,
    evaluate_summary,
    run_episode,
)
from nemesis.eval.policies import (
    BluePolicy,
    RandomBluePolicy,
    ScriptedBluePolicy,
    TrainedBluePolicy,
    random_blue,
    scripted_blue,
    trained_blue,
)

__all__ = [
    "EvalConfig",
    "MetricProbe",
    "evaluate",
    "evaluate_summary",
    "run_episode",
    "BluePolicy",
    "RandomBluePolicy",
    "ScriptedBluePolicy",
    "TrainedBluePolicy",
    "random_blue",
    "scripted_blue",
    "trained_blue",
]
