"""Static contract tests for the default Docker runtime.

These do not require a running Docker daemon. Compose itself performs the full
schema/interpolation check in CI; these tests protect the security properties
that are easiest to accidentally remove during later edits.
"""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _compose():
    with (ROOT / "compose.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_trainer_is_networkless_and_hardened():
    config = _compose()
    shared = config["x-runtime-security"]
    trainer = config["services"]["trainer"]

    assert trainer["network_mode"] == "none"
    assert shared["read_only"] is True
    assert shared["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in shared["security_opt"]
    assert "./runs:/artifacts" in shared["volumes"]


def test_dashboard_is_loopback_only_on_an_internal_network():
    config = _compose()
    dashboard = config["services"]["dashboard"]

    assert config["networks"]["dashboard"]["internal"] is True
    assert dashboard["networks"] == ["dashboard"]
    assert dashboard["ports"] == [
        "127.0.0.1:${NEMESIS_DASHBOARD_PORT:-8001}:8001"
    ]


def test_build_context_excludes_credentials_and_artifacts():
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "runs" in ignored
    assert ".git" in ignored
