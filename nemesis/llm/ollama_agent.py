"""A CC4 blue agent whose policy is a small Qwen model served by Ollama.

The agent asks a locally-running Ollama model to pick an action each step and
runs the reply through :mod:`nemesis.llm.action_parser`, so the choice is always
a legal flat action index — the exact thing ``BlueFlatWrapper.step`` expects,
and the same contract a PPO policy's predicted index satisfies. That is what
lets the LLM be benchmarked head-to-head against PPO in the eval harness.

Default model: ``qwen2.5:1.5b-instruct`` (``Qwen/Qwen2.5-1.5B-Instruct``) — small
enough to run on a laptop CPU/GPU through Ollama and the same family we
LoRA-fine-tune on Modal (see :mod:`modal_app.llm_lora`). Override with the
``model`` argument or the ``NEMESIS_LLM_MODEL`` environment variable.

Import safety
-------------
* Ollama is reached over HTTP; if the server is down, acting raises
  :class:`OllamaError` with a clear message (tests skip rather than fail).
* ``requests`` and CybORG's ``NemesisAgent`` are imported defensively so this
  module imports even where they are absent; only *using* the pieces that need
  them requires them.
"""

from __future__ import annotations

import json as _json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from nemesis.llm.action_parser import Decision, decide, fallback_index
from nemesis.llm.prompt import SYSTEM_PROMPT, build_prompt

try:  # requests is normally installed; keep the module importable if not.
    import requests as _requests
except Exception:  # pragma: no cover - environment without requests
    _requests = None

try:  # CybORG provides the agent base; fall back to a shim so imports work.
    from nemesis.agents.base import NemesisAgent as _AgentBase

    _HAS_NEMESIS_BASE = True
except Exception:  # pragma: no cover - environment without CybORG
    _HAS_NEMESIS_BASE = False

    class _AgentBase:  # minimal stand-in with the attributes we touch
        action_priority: List[str] = []

        def __init__(self, name=None, epsilon: float = 0.0, np_random=None):
            self.name = name
            self.epsilon = epsilon
            self.np_random = np_random

        def train(self, results):
            pass

        def end_episode(self):
            pass


DEFAULT_MODEL = os.environ.get("NEMESIS_LLM_MODEL", "qwen2.5:1.5b-instruct")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


@dataclass
class OllamaConfig:
    """Connection and sampling settings for the Ollama HTTP API."""

    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    temperature: float = 0.0  # deterministic defence by default
    top_p: float = 1.0
    num_predict: int = 24  # the reply is a single short line
    timeout: float = 60.0
    keep_alive: str = "5m"  # keep the model warm between steps

    def options(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
        }


class OllamaClient:
    """Thin wrapper over the Ollama ``/api/chat`` endpoint.

    No model libraries are pulled in; this is just HTTP + JSON. Any transport
    or server error surfaces as :class:`OllamaError`.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def _require_requests(self):
        if _requests is None:
            raise OllamaError(
                "the 'requests' package is required to talk to Ollama; install it"
            )
        return _requests

    def available(self) -> bool:
        """True if the Ollama server answers ``/api/tags`` (never raises)."""
        if _requests is None:
            return False
        try:
            resp = _requests.get(f"{self.config.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(self, system: str, user: str) -> str:
        """Send one system+user turn and return the model's text reply."""
        requests = self._require_requests()
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": self.config.options(),
            "keep_alive": self.config.keep_alive,
        }
        url = f"{self.config.host}/api/chat"
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout)
        except Exception as exc:
            raise OllamaError(
                f"could not reach Ollama at {self.config.host} "
                f"(is `ollama serve` running and `{self.config.model}` pulled?): {exc}"
            ) from exc
        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama returned HTTP {resp.status_code} for model "
                f"{self.config.model!r}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
        except _json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise OllamaError(f"Ollama returned non-JSON: {resp.text[:300]}") from exc
        return (data.get("message") or {}).get("content", "") or ""


class OllamaBlueAgent(_AgentBase):
    """A :class:`~nemesis.agents.base.NemesisAgent` whose actions come from an LLM.

    Primary (flat) interface — the one used to benchmark against PPO::

        agent = OllamaBlueAgent("blue_agent_0")
        idx = agent.act(obs_vector, env.action_labels(a), info[a]["action_mask"])
        obs, reward, term, trunc, info = env.step({a: idx, ...})

    ``idx`` is always a legal flat action index. ``agent.last_decision`` holds
    the :class:`~nemesis.llm.action_parser.Decision` for inspection/logging.
    """

    def __init__(
        self,
        name: str = "blue_agent_0",
        *,
        config: Optional[OllamaConfig] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        np_random=None,
        client: Optional[OllamaClient] = None,
    ):
        super().__init__(name=name, epsilon=0.0, np_random=np_random)
        if config is None:
            config = OllamaConfig(temperature=temperature)
            if model is not None:
                config.model = model
            if host is not None:
                config.host = host
        self.config = config
        self.client = client or OllamaClient(config)
        self.max_retries = max_retries
        self.last_decision: Optional[Decision] = None
        self._history: List[dict] = []

    # ------------------------------------------------------------------ #
    # Flat policy interface (benchmark / eval path)
    # ------------------------------------------------------------------ #
    def act(
        self,
        observation,
        action_labels: Sequence[str],
        action_mask: Sequence[bool],
        *,
        step: Optional[int] = None,
        mission_phase=None,
        use_history: bool = True,
    ) -> int:
        """Choose a legal flat action index for the current step."""

        def prompt_builder(feedback: Optional[str]) -> str:
            return build_prompt(
                observation,
                action_labels,
                action_mask,
                step=step,
                mission_phase=mission_phase,
                history=self._history if use_history else None,
                retry_feedback=feedback,
            )

        decision = decide(
            lambda prompt: self.client.chat(SYSTEM_PROMPT, prompt),
            prompt_builder,
            action_labels=action_labels,
            action_mask=action_mask,
            max_retries=self.max_retries,
        )
        self.last_decision = decision
        if use_history:
            self._history.append({"step": step, "label": decision.label})
            self._history = self._history[-5:]  # keep the tail short
        return decision.index

    def record_reward(self, reward: float) -> None:
        """Attach the realised reward to the most recent history entry (optional)."""
        if self._history:
            self._history[-1]["reward"] = float(reward)

    def end_episode(self):
        self._history = []
        self.last_decision = None

    # ------------------------------------------------------------------ #
    # NemesisAgent interface
    # ------------------------------------------------------------------ #
    def get_action(self, observation, action_space):
        """NemesisAgent entry point.

        The LLM agent reads a *labelled* action menu, which the bare flat
        ``Discrete`` space does not carry. Drive it through :meth:`act` (or
        :func:`play_episode`) with the wrapper's ``action_labels`` and
        ``action_mask`` instead. We raise a clear error rather than silently
        sampling a random action.
        """
        raise NotImplementedError(
            "OllamaBlueAgent needs the labelled action menu; call "
            "agent.act(observation, action_labels, action_mask) with the flat "
            "wrapper's labels/mask, or use nemesis.llm.ollama_agent.play_episode."
        )


def play_episode(
    env,
    agent_name: str,
    agent: OllamaBlueAgent,
    *,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Drive one episode of a ``BlueFlatWrapper`` with ``agent`` on ``agent_name``.

    Every other blue agent sleeps, mirroring ``SingleBlueEnv`` but exposing the
    per-step ``action_mask`` the LLM needs. Returns a summary dict with the
    total blue reward, step count, and how often the safe fallback was used.

    Imports CybORG's ``Sleep`` lazily, so this module still imports without the
    simulator; calling this without CybORG raises a clear error.
    """
    try:
        from CybORG.Simulator.Actions import Sleep
    except Exception as exc:  # pragma: no cover - environment without CybORG
        raise OllamaError(
            "CybORG is required to run an episode; it is not importable"
        ) from exc

    observations, info = env.reset(seed=seed)
    labels = env.action_labels(agent_name)
    total_reward = 0.0
    fallbacks = 0
    steps = 0
    agent.end_episode()

    while agent_name in getattr(env, "agents", [agent_name]):
        mask = info.get(agent_name, {}).get("action_mask")
        if mask is None:
            mask = env.action_mask(agent_name)
        idx = agent.act(
            observations[agent_name],
            labels,
            mask,
            step=steps,
            mission_phase=observations[agent_name][0]
            if len(observations[agent_name]) else None,
        )
        if agent.last_decision and agent.last_decision.used_fallback:
            fallbacks += 1
        actions = {a: Sleep() for a in env.agents if a != agent_name}
        actions[agent_name] = idx
        observations, rewards, terminated, truncated, info = env.step(actions)
        reward = float(rewards.get(agent_name, 0.0))
        agent.record_reward(reward)
        total_reward += reward
        steps += 1
        if terminated.get(agent_name) or truncated.get(agent_name):
            break
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "agent": agent_name,
        "steps": steps,
        "blue_reward": total_reward,
        "fallbacks": fallbacks,
        "fallback_rate": (fallbacks / steps) if steps else 0.0,
        "model": agent.config.model,
    }


__all__ = [
    "OllamaError",
    "OllamaConfig",
    "OllamaClient",
    "OllamaBlueAgent",
    "play_episode",
    "DEFAULT_MODEL",
    "DEFAULT_HOST",
    "fallback_index",
]
