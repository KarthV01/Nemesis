"""The allow-list adapter: trained agent's abstract decision -> benign test action.

A blue policy trained in CybORG emits abstract action objects (``Analyse``,
``Remove``, ``Restore``, ``BlockTrafficZone``, ``Monitor``, ...). Sandbox
evaluation must turn such a decision into *something to do against a real lab
container* — but only ever a **benign, defensive, pre-vetted** test action.

This module is the single chokepoint for that mapping and it is **deny-by-
default**: a decision is executed only if its action is explicitly on the
allow-list. Everything else — an offensive red action that somehow reached here,
a blue action with no vetted real-world equivalent, an unrecognised name — is
**refused and logged**, never executed. That is what keeps the sandbox safe by
construction: the set of things that can happen to a real container is finite,
small, and auditable, regardless of what the policy decides.

The module is pure Python with no CybORG or Docker import, so the allow-list
logic is fully unit-testable on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional

_log = logging.getLogger("nemesis.sandbox.action_adapter")


class TestAction(str, Enum):
    """The *only* actions the sandbox may take against a real lab target.

    Each is benign and defensive: a read-only observation, a service restart, or
    a network isolate/reconnect. There is no offensive action in this enum by
    design — the adapter cannot emit one because none exists to emit.
    """

    CHECK_HEALTH = "check_service_health"      # liveness/health probe (read-only)
    COLLECT_LOGS = "collect_logs"              # read recent logs (read-only)
    RESTART_SERVICE = "restart_service"        # restart the target service
    ISOLATE_CONTAINER = "isolate_container"    # detach from the lab network
    RECONNECT_CONTAINER = "reconnect_container"  # re-attach to the lab network
    NOOP = "noop"                              # do nothing this step

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


#: Benign test actions that *change* the target (as opposed to just observing
#: it). The verifier counts these as potential false positives when there was no
#: intrusion to justify them.
DISRUPTIVE_ACTIONS = frozenset(
    {TestAction.RESTART_SERVICE, TestAction.ISOLATE_CONTAINER}
)

#: The allow-list. Keys are CybORG blue action *class names*, lower-cased; values
#: are the benign test action each maps to. A blue decision is executed only if
#: its action name is a key here. Deliberately conservative:
#:   * ``Monitor``/``Analyse`` -> read-only observation,
#:   * ``Remove``/``Restore``  -> restart the service (a benign reset),
#:   * ``Block/AllowTrafficZone`` -> isolate/reconnect on the lab network,
#:   * ``Sleep`` -> no-op.
#: ``DeployDecoy``/``Misinform`` (deception) are intentionally *absent*: they
#: have no vetted benign equivalent against a real target, so they are refused.
DEFAULT_ALLOW_LIST: Dict[str, TestAction] = {
    "monitor": TestAction.CHECK_HEALTH,
    "analyse": TestAction.COLLECT_LOGS,
    "analyze": TestAction.COLLECT_LOGS,  # spelling tolerance
    "remove": TestAction.RESTART_SERVICE,
    "restore": TestAction.RESTART_SERVICE,
    "blocktrafficzone": TestAction.ISOLATE_CONTAINER,
    "allowtrafficzone": TestAction.RECONNECT_CONTAINER,
    "sleep": TestAction.NOOP,
}

#: Known offensive / state-changing actions that must NEVER touch a real target.
#: Being off the allow-list already refuses them; naming them here only makes the
#: refusal reason specific and lets tests pin the intent. This is not exhaustive
#: and is never consulted to *allow* anything.
KNOWN_OFFENSIVE = frozenset(
    {
        "exploitremoteservice",
        "exploitnetworkservice",
        "privilegeescalate",
        "escalate",
        "impact",
        "degradeservices",
        "discoverremotesystems",
        "discovernetworkservices",
        "aggressiveservicediscovery",
        "stealthservicediscovery",
        "ftpdirectorytraversal",
        "ssh_bruteforce",
        "ssh_login",
    }
)


@dataclass(frozen=True)
class AdaptResult:
    """Outcome of adapting one abstract decision.

    ``allowed`` is the thing to check. When ``True``, ``test_action`` is the
    benign action to run; when ``False``, ``test_action`` is ``None`` and the
    decision has been refused and logged.
    """

    allowed: bool
    abstract_action: str
    test_action: Optional[TestAction]
    reason: str
    #: Best-effort target string parsed from the decision (host/subnet), info only.
    target: Optional[str] = None


def action_name_of(decision: object) -> str:
    """Extract the canonical action name from many decision shapes.

    Accepts a plain string (``"Restore"`` or a label like ``"Restore host_3"``),
    an instantiated CybORG action object (uses its class name), an action class
    (uses its ``__name__``), or a mapping with an ``"action"`` key. Returns the
    leading action token; falls back to ``str(decision)``'s first token.
    """
    if isinstance(decision, Mapping):
        if "action" in decision:
            return action_name_of(decision["action"])
        return ""
    if isinstance(decision, str):
        return decision.strip().split()[0] if decision.strip() else ""
    if isinstance(decision, type):
        return decision.__name__
    # An instance of some object (e.g. a CybORG action) -> its class name.
    name = type(decision).__name__
    if name not in ("int", "float", "bool", "NoneType"):
        return name
    text = str(decision).strip()
    return text.split()[0] if text else ""


def target_of(decision: object) -> Optional[str]:
    """Best-effort extraction of a target host/subnet (informational only)."""
    if isinstance(decision, Mapping):
        for key in ("target", "hostname", "host", "subnet", "ip_address"):
            if decision.get(key):
                return str(decision[key])
        return None
    if isinstance(decision, str):
        parts = decision.strip().split(maxsplit=1)
        return parts[1] if len(parts) == 2 else None
    for attr in ("hostname", "host", "ip_address", "from_subnet", "subnet"):
        val = getattr(decision, attr, None)
        if val:
            return str(val)
    return None


class ActionAdapter:
    """Maps abstract decisions to benign test actions, deny-by-default.

    Parameters
    ----------
    allow_list : mapping, optional
        Override the default allow-list (keys lower-cased action names).
    logger : logging.Logger, optional
        Where refusals are logged (as warnings). Defaults to the module logger.
    """

    def __init__(
        self,
        allow_list: Optional[Mapping[str, TestAction]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        src = DEFAULT_ALLOW_LIST if allow_list is None else allow_list
        # Normalise keys to lower-case so lookups are case-insensitive.
        self.allow_list: Dict[str, TestAction] = {k.lower(): v for k, v in src.items()}
        self.logger = logger or _log
        #: Every refused decision, in order — an audit trail for the run.
        self.refusals: List[AdaptResult] = []

    def adapt(self, decision: object, *, target: Optional[str] = None) -> AdaptResult:
        """Adapt one decision. Refusals are recorded and logged as a side effect."""
        name = action_name_of(decision)
        key = name.lower()
        tgt = target if target is not None else target_of(decision)

        if key and key in self.allow_list:
            return AdaptResult(
                allowed=True,
                abstract_action=name,
                test_action=self.allow_list[key],
                reason="allow-listed benign defensive action",
                target=tgt,
            )

        if not key:
            reason = "refused: could not identify an action in the decision"
        elif key in KNOWN_OFFENSIVE:
            reason = "refused: offensive/unsafe action is never permitted against a real target"
        else:
            reason = "refused: action is not on the benign allow-list"

        result = AdaptResult(
            allowed=False,
            abstract_action=name,
            test_action=None,
            reason=reason,
            target=tgt,
        )
        self.refusals.append(result)
        self.logger.warning(
            "sandbox action adapter refused %r (target=%s): %s",
            name or "<unknown>",
            tgt,
            result.reason,
        )
        return result


#: A module-level default adapter for one-off calls.
_DEFAULT_ADAPTER = ActionAdapter()


def adapt(decision: object, *, target: Optional[str] = None) -> AdaptResult:
    """Adapt a decision with the shared default adapter."""
    return _DEFAULT_ADAPTER.adapt(decision, target=target)


__all__ = [
    "TestAction",
    "DISRUPTIVE_ACTIONS",
    "DEFAULT_ALLOW_LIST",
    "KNOWN_OFFENSIVE",
    "AdaptResult",
    "ActionAdapter",
    "action_name_of",
    "target_of",
    "adapt",
]
