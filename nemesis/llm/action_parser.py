"""Parse a model's text reply into a *legal* flat action index — safely.

This is the correctness core of the LLM agent. The guarantee it enforces:

    the index handed back to the environment is ALWAYS one the action mask
    marks legal — no matter what the model emits.

A reply is classified as:

* ``ok``           — a legal index was extracted.
* ``malformed``    — no index could be found at all.
* ``out_of_range`` — an index outside the action list was named.
* ``illegal``      — an in-range index that the mask marks *not* legal.

Only ``ok`` is accepted. :func:`decide` drives the full loop: ask, parse,
re-ask with feedback up to ``max_retries`` times, and finally fall back to the
always-legal ``Sleep`` action. The loop is pure — it takes a ``generate``
callable, so it is exercised in tests with a fake model and no server.

Pure Python: no CybORG, no network, no model libraries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

# Matches an explicit "ACTION: 3" / "action = 3" / "Action -> [3]" directive.
_ACTION_RE = re.compile(r"action\s*(?:[:=]|->|is)?\s*\[?\s*(-?\d+)\s*\]?", re.IGNORECASE)
# Matches a bracketed index like "[3]" anywhere in the text.
_BRACKET_RE = re.compile(r"\[\s*(-?\d+)\s*\]")
# A reply that is *only* a number (optionally fenced), e.g. "3" or "`3`".
_BARE_INT_RE = re.compile(r"^-?\d+$")

# status constants
OK = "ok"
MALFORMED = "malformed"
OUT_OF_RANGE = "out_of_range"
ILLEGAL = "illegal"


def _strip(text: str) -> str:
    """Remove code fences and <think>...</think> blocks, then trim."""
    if not text:
        return ""
    # Drop chain-of-thought blocks some models emit.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Drop Markdown code fences but keep their contents.
    text = text.replace("```", " ").replace("`", " ")
    return text.strip()


def extract_index(text: str) -> Optional[int]:
    """Pull the action index out of a model reply, or ``None`` if there is none.

    Resolution order, most explicit first (and the *last* match of each kind
    wins, since models often restate their final answer):

    1. an ``ACTION: <n>`` directive,
    2. a bracketed ``[n]``,
    3. a reply that is nothing but an integer.

    A bare integer buried in prose is deliberately **not** matched: host names
    such as ``...user_host_0`` contain digits, and grabbing one would risk a
    wrong-but-legal action. When in doubt we return ``None`` and let the caller
    re-ask or fall back to Sleep.
    """
    if text is None:
        return None
    cleaned = _strip(text)
    if not cleaned:
        return None

    matches = list(_ACTION_RE.finditer(cleaned))
    if matches:
        return int(matches[-1].group(1))

    brackets = list(_BRACKET_RE.finditer(cleaned))
    if brackets:
        return int(brackets[-1].group(1))

    # Only trust a bare integer if the whole reply (last non-empty line) is one.
    last_line = cleaned.splitlines()[-1].strip() if cleaned.splitlines() else cleaned
    if _BARE_INT_RE.match(last_line):
        return int(last_line)
    if _BARE_INT_RE.match(cleaned):
        return int(cleaned)
    return None


@dataclass
class ParseResult:
    """Outcome of parsing one reply against the action mask."""

    index: Optional[int]  # the index the model asked for (may be illegal/None)
    status: str  # one of OK / MALFORMED / OUT_OF_RANGE / ILLEGAL
    legal: bool  # True only when status == OK

    @property
    def reason(self) -> str:
        if self.status == OK:
            return "accepted"
        if self.status == MALFORMED:
            return "no action index found in the reply"
        if self.status == OUT_OF_RANGE:
            return f"index {self.index} is outside the menu"
        if self.status == ILLEGAL:
            return f"index {self.index} is not a legal action right now"
        return self.status


def resolve_action(text: str, action_mask: Sequence[bool]) -> ParseResult:
    """Parse ``text`` and validate the chosen index against ``action_mask``."""
    idx = extract_index(text)
    if idx is None:
        return ParseResult(None, MALFORMED, False)
    if idx < 0 or idx >= len(action_mask):
        return ParseResult(idx, OUT_OF_RANGE, False)
    if not bool(action_mask[idx]):
        return ParseResult(idx, ILLEGAL, False)
    return ParseResult(idx, OK, True)


def fallback_index(
    action_labels: Sequence[str],
    action_mask: Sequence[bool],
    *,
    fallback_label: str = "Sleep",
) -> int:
    """The index to use when the model cannot give a legal one.

    Prefers the action whose label is exactly ``fallback_label`` (``Sleep`` —
    the safe no-op, always legal in CC4). Falls back to the first legal index,
    then to 0 if, impossibly, nothing is legal.
    """
    for i, ok in enumerate(action_mask):
        if ok and i < len(action_labels) and action_labels[i] == fallback_label:
            return i
    for i, ok in enumerate(action_mask):
        if ok:
            return i
    return 0


@dataclass
class Decision:
    """The result of a full ask/parse/retry/fallback cycle.

    ``index`` is guaranteed legal. ``status`` is ``"ok"`` when the model picked
    a legal action within the retry budget, or ``"fallback"`` when we defaulted
    to the safe action instead.
    """

    index: int
    status: str  # "ok" | "fallback"
    attempts: int
    label: Optional[str] = None
    reason: str = ""
    raw_outputs: List[str] = field(default_factory=list)

    @property
    def used_fallback(self) -> bool:
        return self.status == "fallback"


def decide(
    generate: Callable[[str], str],
    prompt_builder: Callable[[Optional[str]], str],
    *,
    action_labels: Sequence[str],
    action_mask: Sequence[bool],
    max_retries: int = 2,
    fallback_label: str = "Sleep",
) -> Decision:
    """Run the constrained-decoding loop and return a guaranteed-legal action.

    Parameters
    ----------
    generate
        ``generate(prompt) -> str`` — calls the model. In production this is the
        Ollama client; in tests it is a stub.
    prompt_builder
        ``prompt_builder(feedback) -> str`` — builds the prompt. ``feedback`` is
        ``None`` on the first attempt and, on a retry, a short string explaining
        why the previous reply was rejected (so it can be shown to the model).
    action_labels, action_mask
        The current flat labels and legality mask.
    max_retries
        Extra attempts after the first (so ``max_retries=2`` means up to three
        model calls). After these are exhausted we fall back to Sleep.
    fallback_label
        Label of the safe action to fall back to.

    The returned :class:`Decision` always carries a legal ``index``.
    """
    outputs: List[str] = []
    last_reason = "no attempts made"
    attempts = 0
    feedback: Optional[str] = None

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        prompt = prompt_builder(feedback)
        try:
            raw = generate(prompt)
        except Exception as exc:  # a model/transport failure is just a bad attempt
            raw = ""
            last_reason = f"model call failed: {exc}"
            feedback = "Your previous reply did not arrive; answer with ACTION: <index>."
            outputs.append("")
            continue
        outputs.append(raw if raw is not None else "")
        result = resolve_action(raw, action_mask)
        if result.legal:
            label = (
                action_labels[result.index]
                if result.index < len(action_labels)
                else None
            )
            return Decision(
                index=result.index,
                status="ok",
                attempts=attempts,
                label=label,
                reason="accepted",
                raw_outputs=outputs,
            )
        last_reason = result.reason
        feedback = (
            f"Your last reply was rejected: {result.reason}. "
            "Choose one index shown in the menu and reply ACTION: <index>."
        )

    idx = fallback_index(action_labels, action_mask, fallback_label=fallback_label)
    label = action_labels[idx] if idx < len(action_labels) else None
    return Decision(
        index=idx,
        status="fallback",
        attempts=attempts,
        label=label,
        reason=last_reason,
        raw_outputs=outputs,
    )


def safe_index(
    text: str,
    action_labels: Sequence[str],
    action_mask: Sequence[bool],
    *,
    fallback_label: str = "Sleep",
) -> Tuple[int, bool]:
    """One-shot convenience: parse ``text`` once, returning a legal index.

    Returns ``(index, accepted)`` where ``index`` is legal either way and
    ``accepted`` is True when the model's own choice was used (False when the
    fallback was substituted). No retries — use :func:`decide` for those.
    """
    result = resolve_action(text, action_mask)
    if result.legal:
        return result.index, True
    return fallback_index(action_labels, action_mask, fallback_label=fallback_label), False


__all__ = [
    "OK",
    "MALFORMED",
    "OUT_OF_RANGE",
    "ILLEGAL",
    "ParseResult",
    "Decision",
    "extract_index",
    "resolve_action",
    "fallback_index",
    "decide",
    "safe_index",
]
