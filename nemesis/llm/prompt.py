"""Turn a CC4 blue observation + its legal-action menu into a text prompt.

This module is **pure Python**: it imports nothing from CybORG, Ollama,
transformers or the network, so it imports and tests anywhere. It is the first
half of the constrained-decoding pipeline — :mod:`nemesis.llm.action_parser` is
the second half (parsing the model's reply back into a *legal* action index).

The design in one line: **we never let the model free-type an action.** Each
step we hand it a numbered menu of the actions the simulator currently says are
legal (built from the flat wrapper's ``action_labels`` + ``action_mask``), and
ask it to reply with one index. The parser then re-checks that index against the
same mask, so an illegal or malformed reply can never reach the environment.

The observation itself (a fixed-size vector of binary sensor flags produced by
``BlueFlatWrapper``) is rendered as a compact, honest summary. The host-level
context a defender actually needs to decide — *which* hosts it may analyse,
remove, restore, etc. — lives in the legal-action menu, where each entry already
names its target host or subnet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple, Union

#: CC4 mission phases, indexed by the first element of a blue observation vector.
MISSION_PHASES = {0: "Preplanning", 1: "Mission A", 2: "Mission B"}

#: The standing instructions handed to the model as the system prompt. Kept
#: terse on purpose: small instruct models follow a short, rule-shaped prompt
#: far better than a discursive one.
SYSTEM_PROMPT = (
    "You are BLUE, the defender in the CAGE Challenge 4 network-security "
    "simulation. Everything is simulated; there are no real hosts.\n"
    "Each turn you are given sensor readings and a numbered menu of the only "
    "actions that are legal right now. Your job is to keep attackers out while "
    "disrupting services as little as possible.\n"
    "Guidance: investigate suspicious hosts (Analyse) before acting; evict "
    "intruders with the cheap Remove before the disruptive Restore; use "
    "DeployDecoy/Misinform to slow attackers; block traffic only when needed; "
    "when the network looks clean, Monitor.\n"
    "You MUST pick exactly one action from the menu. Reply with a single line "
    "in the form:\nACTION: <index>\n"
    "where <index> is the number in brackets. Do not invent an index that is "
    "not in the menu and do not add any other text."
)

#: Appended to the user prompt, restating the output contract right before the
#: model answers (recency helps small models comply).
ANSWER_DIRECTIVE = "Reply with exactly one line: ACTION: <index>"


@dataclass
class ObservationView:
    """A human-readable, model-facing snapshot of one blue observation.

    Everything is optional so the view degrades gracefully: with only a raw
    vector we still report the mission phase and overall activity level; a
    runtime decoder may additionally fill ``alerts`` / ``blocked_subnets``.
    """

    mission_phase: Optional[str] = None
    step: Optional[int] = None
    #: Free-text lines describing suspicious activity, e.g.
    #: "malicious process on restricted_zone_a_subnet_server_host_0".
    alerts: List[str] = field(default_factory=list)
    #: Subnets blue currently has blocked.
    blocked_subnets: List[str] = field(default_factory=list)
    #: Count of active (non-zero) sensor flags — a crude "how noisy is it" cue.
    active_flags: Optional[int] = None
    notes: List[str] = field(default_factory=list)


def legal_actions(
    action_labels: Sequence[str], action_mask: Sequence[bool]
) -> List[Tuple[int, str]]:
    """Return ``[(index, label), ...]`` for every currently-legal action.

    ``index`` is the action's position in the *full* flat action list (the value
    that is fed back to ``env.step``), not its position in the filtered menu, so
    the parser can validate a reply against the full ``action_mask`` directly.

    Raises ``ValueError`` if the two sequences disagree in length.
    """
    if len(action_labels) != len(action_mask):
        raise ValueError(
            f"action_labels ({len(action_labels)}) and action_mask "
            f"({len(action_mask)}) must have the same length"
        )
    return [(i, str(action_labels[i])) for i, ok in enumerate(action_mask) if ok]


def mission_phase_name(value) -> str:
    """Map a mission-phase code (0/1/2) to a name, tolerant of junk input."""
    try:
        return MISSION_PHASES.get(int(value), f"phase {int(value)}")
    except (TypeError, ValueError):
        return str(value)


def _active_flag_count(vector: Sequence) -> int:
    n = 0
    for v in vector:
        try:
            if int(v) != 0:
                n += 1
        except (TypeError, ValueError):
            continue
    return n


def observation_view(
    observation: Union[ObservationView, Mapping, Sequence, None],
    *,
    step: Optional[int] = None,
    mission_phase: Optional[Union[int, str]] = None,
) -> ObservationView:
    """Normalise whatever the caller has into an :class:`ObservationView`.

    Accepts an existing view (returned as-is, with ``step`` filled if missing),
    a mapping of named fields, or a raw numeric vector (as produced by
    ``BlueFlatWrapper``) whose first element is the mission phase.
    """
    if isinstance(observation, ObservationView):
        if observation.step is None and step is not None:
            observation.step = step
        if observation.mission_phase is None and mission_phase is not None:
            observation.mission_phase = mission_phase_name(mission_phase)
        return observation

    view = ObservationView(step=step)
    if mission_phase is not None:
        view.mission_phase = mission_phase_name(mission_phase)

    if isinstance(observation, Mapping):
        if "mission_phase" in observation and view.mission_phase is None:
            view.mission_phase = mission_phase_name(observation["mission_phase"])
        view.alerts = list(observation.get("alerts", []) or [])
        view.blocked_subnets = list(observation.get("blocked_subnets", []) or [])
        view.notes = list(observation.get("notes", []) or [])
        if "active_flags" in observation:
            view.active_flags = observation["active_flags"]
        return view

    if observation is not None and _is_sequence(observation):
        vec = list(observation)
        if vec and view.mission_phase is None:
            view.mission_phase = mission_phase_name(vec[0])
        # Count non-zero flags after the leading mission-phase element.
        view.active_flags = _active_flag_count(vec[1:]) if len(vec) > 1 else 0
        return view

    return view


def _is_sequence(obj) -> bool:
    if isinstance(obj, (str, bytes, Mapping)):
        return False
    try:
        iter(obj)
        len(obj)
        return True
    except TypeError:
        return False


def render_observation(
    observation: Union[ObservationView, Mapping, Sequence, None],
    *,
    step: Optional[int] = None,
    mission_phase: Optional[Union[int, str]] = None,
) -> str:
    """Render an observation as a short, readable block for the prompt."""
    view = observation_view(observation, step=step, mission_phase=mission_phase)
    lines: List[str] = []
    if view.step is not None:
        lines.append(f"Step: {view.step}")
    if view.mission_phase is not None:
        lines.append(f"Mission phase: {view.mission_phase}")
    if view.blocked_subnets:
        lines.append("Blocked subnets: " + ", ".join(view.blocked_subnets))
    if view.alerts:
        lines.append("Alerts:")
        lines.extend(f"  - {a}" for a in view.alerts)
    elif view.active_flags is not None:
        if view.active_flags == 0:
            lines.append("Sensors: quiet (no active flags).")
        else:
            lines.append(
                f"Sensors: {view.active_flags} active flag(s); "
                "see the menu for hosts you can act on."
            )
    for note in view.notes:
        lines.append(note)
    if not lines:
        lines.append("Sensors: (no readings available)")
    return "\n".join(lines)


def format_action_menu(
    legal: Sequence[Tuple[int, str]], *, max_actions: Optional[int] = None
) -> str:
    """Render ``[(index, label), ...]`` as a numbered menu, one per line.

    ``max_actions`` truncates very long menus (keeping the first *n*); the
    truncation is announced so the model knows more legal actions exist. The
    always-safe ``Sleep``/``Monitor`` entries are kept even when truncating.
    """
    if not legal:
        return "(no legal actions)"
    items = list(legal)
    truncated = False
    if max_actions is not None and len(items) > max_actions:
        keep_safe = [p for p in items if p[1] in ("Sleep", "Monitor")]
        head = [p for p in items[:max_actions] if p not in keep_safe]
        items = keep_safe + head
        items = sorted(items, key=lambda p: p[0])[:max_actions]
        truncated = True
    lines = [f"  [{idx}] {label}" for idx, label in items]
    if truncated:
        lines.append("  ... (more legal actions exist; pick one shown above)")
    return "\n".join(lines)


def format_history(history: Optional[Sequence]) -> str:
    """Render a short action history as context. Each item is a mapping or a
    ``(step, label, reward)`` / ``(label, reward)`` / ``label`` entry."""
    if not history:
        return ""
    rows: List[str] = []
    for item in history:
        if isinstance(item, Mapping):
            label = item.get("label", item.get("action", "?"))
            reward = item.get("reward")
            step = item.get("step")
        elif isinstance(item, (list, tuple)):
            if len(item) == 3:
                step, label, reward = item
            elif len(item) == 2:
                step, (label, reward) = None, item
            else:
                step, label, reward = None, item[0], None
        else:
            step, label, reward = None, item, None
        prefix = f"  step {step}: " if step is not None else "  - "
        if reward is not None:
            rows.append(f"{prefix}{label} (reward {reward:+.2f})")
        else:
            rows.append(f"{prefix}{label}")
    return "Recent actions:\n" + "\n".join(rows)


def build_prompt(
    observation: Union[ObservationView, Mapping, Sequence, None],
    action_labels: Sequence[str],
    action_mask: Sequence[bool],
    *,
    step: Optional[int] = None,
    mission_phase: Optional[Union[int, str]] = None,
    history: Optional[Sequence] = None,
    retry_feedback: Optional[str] = None,
    max_actions: Optional[int] = None,
) -> str:
    """Assemble the full user prompt for one decision.

    Parameters
    ----------
    observation
        An :class:`ObservationView`, a mapping of named fields, or the raw flat
        observation vector.
    action_labels, action_mask
        The flat wrapper's ``action_labels(agent)`` and the per-step
        ``action_mask`` (``info[agent]["action_mask"]``). Must be equal length.
    step, mission_phase
        Optional extra context. ``mission_phase`` may be given explicitly; if
        omitted it is read from the observation vector's first element.
    history
        Optional recent-action context (see :func:`format_history`).
    retry_feedback
        If set, a line explaining why the previous reply was rejected — used by
        :func:`nemesis.llm.action_parser.decide` when it re-asks the model.
    max_actions
        Optional cap on how many menu entries to show.

    The returned string pairs with :data:`SYSTEM_PROMPT` as the user turn.
    """
    legal = legal_actions(action_labels, action_mask)
    parts = [
        "## Network status",
        render_observation(observation, step=step, mission_phase=mission_phase),
    ]
    hist = format_history(history)
    if hist:
        parts.append(hist)
    parts.append("## Legal actions (choose ONE by its index)")
    parts.append(format_action_menu(legal, max_actions=max_actions))
    if retry_feedback:
        parts.append(f"NOTE: {retry_feedback}")
    parts.append(ANSWER_DIRECTIVE)
    return "\n\n".join(parts)


__all__ = [
    "MISSION_PHASES",
    "SYSTEM_PROMPT",
    "ANSWER_DIRECTIVE",
    "ObservationView",
    "legal_actions",
    "mission_phase_name",
    "observation_view",
    "render_observation",
    "format_action_menu",
    "format_history",
    "build_prompt",
]
