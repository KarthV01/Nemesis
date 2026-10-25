"""LLM agents for CC4 — a small open-weight model playing blue, with its output
*constrained* to legal simulator actions.

Three layers:

* :mod:`nemesis.llm.prompt` — render an observation + legal-action menu into a
  prompt (pure Python).
* :mod:`nemesis.llm.action_parser` — parse the reply back into a **legal** flat
  action index, with retry and a safe Sleep fallback (pure Python).
* :mod:`nemesis.llm.ollama_agent` — a NemesisAgent-compatible policy that runs
  the above against a locally-served Qwen model via the Ollama HTTP API.

The pure pieces import with no CybORG, no network and no model libraries. The
Ollama agent is imported lazily (via module ``__getattr__``) so that merely
importing :mod:`nemesis.llm` never reaches for CybORG, requests or a server.
"""

from __future__ import annotations

from nemesis.llm.action_parser import (
    Decision,
    ParseResult,
    decide,
    extract_index,
    fallback_index,
    resolve_action,
    safe_index,
)
from nemesis.llm.prompt import (
    SYSTEM_PROMPT,
    ObservationView,
    build_prompt,
    format_action_menu,
    legal_actions,
    render_observation,
)

_LAZY = {
    "OllamaAgent": "OllamaBlueAgent",
    "OllamaBlueAgent": "OllamaBlueAgent",
    "OllamaClient": "OllamaClient",
    "OllamaConfig": "OllamaConfig",
    "OllamaError": "OllamaError",
    "play_episode": "play_episode",
}


def __getattr__(name: str):  # PEP 562: defer the CybORG/requests import
    if name in _LAZY:
        from nemesis.llm import ollama_agent

        return getattr(ollama_agent, _LAZY[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # prompt
    "SYSTEM_PROMPT",
    "ObservationView",
    "build_prompt",
    "format_action_menu",
    "legal_actions",
    "render_observation",
    # parser
    "Decision",
    "ParseResult",
    "decide",
    "extract_index",
    "fallback_index",
    "resolve_action",
    "safe_index",
    # agent (lazy)
    "OllamaBlueAgent",
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
    "play_episode",
]
