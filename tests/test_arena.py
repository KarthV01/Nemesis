"""Arena request and Docker job-controller tests (no Docker or CybORG required)."""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from nemesis.arena.controller import MatchManager
from nemesis.arena.models import MatchRequest


def wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return predicate()


def test_match_request_resolves_team_and_role_formats():
    team = MatchRequest(steps=10)
    assert team.agent_classes() == ("RedCaptain", "BlueCaptain")

    duel = MatchRequest(format="1v1", red_role="recon", blue_role="detect", steps=5)
    assert duel.agent_classes() == ("RedScout", "BlueSentinel")


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"format": "1v1"}, "red_role"),
        ({"steps": 0}, "steps"),
        ({"scenario": "unknown"}, "cc4"),
        ({"blue_checkpoint": "untrusted/path"}, "no arena-compatible checkpoint"),
        ({"extra": True}, "unknown fields"),
    ],
)
def test_match_request_rejects_invalid_input(payload, message):
    with pytest.raises((ValueError, TypeError), match=message):
        MatchRequest.from_dict(payload)


def test_manager_runs_hardened_container_and_saves_job(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        mount = command[command.index("-v") + 1]
        match_dir = mount.rsplit(":/arena", 1)[0]
        with open(f"{match_dir}/result.json", "w", encoding="utf-8") as fh:
            json.dump({"status": "completed", "winner": "draw"}, fh)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager = MatchManager(str(tmp_path), image="arena:test")
    submitted = manager.submit(MatchRequest(steps=2, delay=0))
    completed = wait_for(lambda: manager.get(submitted["id"])["status"] == "completed")

    assert completed
    command = calls[0]
    assert command[:2] == ["docker", "run"]
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert ["--cap-drop", "ALL"] == command[command.index("--cap-drop"):command.index("--cap-drop") + 2]
    assert command[-5:] == ["scripts/arena_match.py", "--config", "/arena/config.json", "--output", "/arena"]
    assert manager.get(submitted["id"])["result"]["winner"] == "draw"


def test_manager_reports_container_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "image missing"),
    )
    manager = MatchManager(str(tmp_path))
    job = manager.submit(MatchRequest(steps=2, delay=0))
    failed = wait_for(lambda: manager.get(job["id"])["status"] == "failed")
    assert failed
    assert "image missing" in manager.get(job["id"])["error"]
