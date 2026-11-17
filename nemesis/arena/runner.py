"""Run one arena match and persist its live/replay event stream."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

from nemesis.arena.models import MatchRequest
from nemesis.platform.events import EventKind, EventLog


def _resolve_agent(name: str):
    import nemesis.agents as agents

    try:
        return getattr(agents, name)
    except AttributeError as exc:
        raise ValueError(f"unknown arena agent class {name!r}") from exc


def _build_scripted_env(request: MatchRequest):
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from nemesis.arena.agents import single_active_agent_class

    red_name, blue_name = request.agent_classes()
    red_class = _resolve_agent(red_name)
    blue_class = _resolve_agent(blue_name)
    if request.format == "1v1":
        red_class = single_active_agent_class(red_class, "red_agent_0")
        blue_class = single_active_agent_class(blue_class, "blue_agent_0")
    scenario = EnterpriseScenarioGenerator(
        red_agent_class=red_class,
        blue_agent_class=blue_class,
        green_agent_class=EnterpriseGreenAgent,
        steps=request.steps,
    )
    return CybORG(scenario_generator=scenario, seed=request.seed)


def _describe(action) -> str:
    name = type(action).__name__
    target = getattr(action, "hostname", None) or getattr(action, "ip_address", None)
    return name if target is None else f"{name} -> {str(target).replace('_subnet_', '/')}"


def _state_snapshot(env) -> Dict[str, object]:
    controller = env.environment_controller
    red_hosts = set()
    privileged = set()
    for agent in controller.team.get("Red", []):
        for session in controller.state.sessions.get(agent, {}).values():
            red_hosts.add(session.hostname)
            if session.has_privileged_access():
                privileged.add(session.hostname)
    return {
        "red_footholds": len(red_hosts),
        "red_privileged": len(privileged),
        "red_hosts": sorted(red_hosts),
        "privileged_hosts": sorted(privileged),
    }


def run_match(request: MatchRequest, output_dir: Path) -> Dict[str, object]:
    """Run a finite match. The caller is responsible for process isolation."""
    request.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    result_path = output_dir / "result.json"

    with events_path.open("w", encoding="utf-8") as stream:
        log = EventLog(stream)
        log.log(
            EventKind.SYSTEM,
            f"Arena match started: {request.format}, {request.steps} steps",
            episode=1,
            phase="started",
            config=request.as_dict(),
        )
        env = _build_scripted_env(request)
        blue_total = 0.0
        red_total = 0.0
        previous_state = None
        completed_steps = 0
        try:
            env.reset(seed=request.seed)
            interfaces = env.environment_controller.agent_interfaces
            for step in range(1, request.steps + 1):
                env.step()
                completed_steps = step
                for name in env.active_agents:
                    side = name.split("_", 1)[0]
                    if side not in ("red", "blue"):
                        continue
                    action = env.get_last_action(name)
                    action = action[0] if isinstance(action, list) else action
                    if action is None or type(action).__name__ == "Sleep":
                        continue
                    role = getattr(interfaces[name].agent, "last_role", None)
                    if request.format == "1v1":
                        role = request.red_role if side == "red" else request.blue_role
                    success = str(env.get_observation(name).get("success", "UNKNOWN"))
                    log.log(
                        EventKind.RED if side == "red" else EventKind.BLUE,
                        _describe(action),
                        episode=1,
                        step=step,
                        agent=name,
                        role=role,
                        success=success,
                    )

                reward = float(env.environment_controller.reward.get("Blue", {}).get("BlueRewardMachine", 0.0))
                blue_total += reward
                red_total -= reward
                log.log(
                    EventKind.REWARD,
                    f"Step {step}: blue {reward:+.2f}, red {-reward:+.2f}",
                    episode=1,
                    step=step,
                    reward=reward,
                    blue_total=round(blue_total, 3),
                    red_total=round(red_total, 3),
                )
                state = _state_snapshot(env)
                if state != previous_state:
                    log.log(
                        EventKind.METRIC,
                        f"State changed: {state['red_footholds']} footholds, {state['red_privileged']} privileged",
                        episode=1,
                        step=step,
                        state=state,
                    )
                    previous_state = state
                if request.delay:
                    time.sleep(request.delay)
        finally:
            try:
                env.close()
            except Exception:
                pass

        result = {
            "status": "completed",
            "steps": completed_steps,
            "blue_reward": round(blue_total, 3),
            "red_reward": round(red_total, 3),
            "winner": "blue" if blue_total > red_total else "red" if red_total > blue_total else "draw",
            "final_state": previous_state or {},
            "config": request.as_dict(),
        }
        log.log(
            EventKind.SYSTEM,
            f"Match complete: {result['winner']} wins (blue {blue_total:+.2f}, red {red_total:+.2f})",
            episode=1,
            step=completed_steps,
            phase="completed",
            result=result,
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["run_match"]
