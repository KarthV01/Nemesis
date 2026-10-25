"""Tests for nemesis.sandbox.

Fully exercises the pure logic (action-adapter allow-list, verifier, the
backend's benign action bridge) with no Docker or CybORG. Tests that genuinely
need a container runtime detect its absence and SKIP with a clear reason.
"""

import json
import logging

import pytest

from nemesis.sandbox.action_adapter import (
    ActionAdapter,
    TestAction,
    action_name_of,
    adapt,
    target_of,
)
from nemesis.sandbox.backend import (
    ProbeResult,
    SandboxBackend,
    SandboxContainer,
    SandboxNetwork,
    SandboxUnavailable,
)
from nemesis.sandbox.docker_backend import DockerBackend
from nemesis.sandbox.verify import (
    StepOutcome,
    VerifyConfig,
    summary_event,
    verify,
)
from nemesis.platform.metrics import EpisodeMetrics


# --------------------------------------------------------------------------- #
# A Docker-free fake backend, for the backend's shared (non-abstract) helpers.
# --------------------------------------------------------------------------- #
class FakeBackend(SandboxBackend):
    def __init__(self, available=True):
        self._avail = available
        self.calls = []

    def is_available(self):
        return self._avail

    def create_network(self, name):
        self.calls.append(("create_network", name))
        return SandboxNetwork(name)

    def remove_network(self, network):
        self.calls.append(("remove_network", network.name))

    def start_target(self, image, name, network):
        self.calls.append(("start_target", image, name, network.name))
        return SandboxContainer(name, image, network.name)

    def stop_target(self, container):
        self.calls.append(("stop_target", container.name))

    def disconnect_target(self, container, network):
        self.calls.append(("disconnect", container.name, network.name))

    def connect_target(self, container, network):
        self.calls.append(("connect", container.name, network.name))

    def probe(self, container, probe):
        self.calls.append(("probe", container.name, probe))
        return ProbeResult(probe=probe, ok=True, raw={"service_up": True})


# --------------------------------------------------------------------------- #
# action adapter — the allow-list
# --------------------------------------------------------------------------- #
def test_allow_list_maps_defensive_actions():
    a = ActionAdapter()
    cases = {
        "Monitor": TestAction.CHECK_HEALTH,
        "Analyse": TestAction.COLLECT_LOGS,
        "Analyze": TestAction.COLLECT_LOGS,
        "Remove": TestAction.RESTART_SERVICE,
        "Restore": TestAction.RESTART_SERVICE,
        "BlockTrafficZone": TestAction.ISOLATE_CONTAINER,
        "AllowTrafficZone": TestAction.RECONNECT_CONTAINER,
        "Sleep": TestAction.NOOP,
    }
    for name, expected in cases.items():
        r = a.adapt(name)
        assert r.allowed, name
        assert r.test_action is expected, name
    assert a.refusals == []


def test_case_insensitive_and_label_parsing():
    a = ActionAdapter()
    assert a.adapt("MONITOR").allowed
    r = a.adapt("Restore restricted_zone_a_subnet_server_host_3")
    assert r.allowed
    assert r.test_action is TestAction.RESTART_SERVICE
    assert r.target == "restricted_zone_a_subnet_server_host_3"


def test_offensive_action_refused_and_logged(caplog):
    a = ActionAdapter()
    with caplog.at_level(logging.WARNING, logger="nemesis.sandbox.action_adapter"):
        r = a.adapt("ExploitRemoteService host_6")
    assert r.allowed is False
    assert r.test_action is None
    assert "offensive" in r.reason
    assert r.target == "host_6"
    assert [x.abstract_action for x in a.refusals] == ["ExploitRemoteService"]
    assert "refused" in caplog.text.lower()


def test_unmapped_blue_action_refused():
    a = ActionAdapter()
    for name in ("DeployDecoy", "Misinform"):
        r = a.adapt(name)
        assert not r.allowed
        assert "allow-list" in r.reason


def test_empty_decision_refused():
    a = ActionAdapter()
    r = a.adapt("")
    assert not r.allowed
    assert "identify" in r.reason


def test_action_name_and_target_extraction_variants():
    class Restore:  # stands in for a CybORG action object / class
        hostname = "host_9"

    assert action_name_of(Restore()) == "Restore"       # instance -> class name
    assert action_name_of(Restore) == "Restore"         # class -> __name__
    assert action_name_of({"action": "Remove", "hostname": "h1"}) == "Remove"
    assert action_name_of("Analyse host_2") == "Analyse"
    assert target_of(Restore()) == "host_9"
    assert target_of({"action": "Remove", "hostname": "h1"}) == "h1"
    assert target_of("Analyse host_2") == "host_2"


def test_custom_allow_list_and_module_level_adapt():
    # A restricted allow-list refuses things the default would allow.
    a = ActionAdapter(allow_list={"monitor": TestAction.CHECK_HEALTH})
    assert a.adapt("Monitor").allowed
    assert not a.adapt("Restore").allowed
    # Module-level convenience uses the default list.
    assert adapt("Restore").test_action is TestAction.RESTART_SERVICE


# --------------------------------------------------------------------------- #
# verify — independent outcome scoring
# --------------------------------------------------------------------------- #
def test_verify_clean_episode_is_a_win():
    steps = [StepOutcome(step=i, service_up=True) for i in range(10)]
    m = verify(steps, episode=3)
    assert isinstance(m, EpisodeMetrics)
    assert m.defender_availability == 1.0
    assert m.attack_success_rate == 0.0
    assert m.false_positives == 0
    assert m.blue_win is True
    assert m.steps == 10 and m.episode == 3
    assert m.extra["source"] == "sandbox"


def test_verify_detected_intrusion_earns_bonus_and_can_win():
    steps = [
        StepOutcome(0, True),
        StepOutcome(
            1, True, intrusion_present=True, intrusion_detected=True,
            action=TestAction.ISOLATE_CONTAINER,  # justified: intrusion present
        ),
        StepOutcome(2, True),
    ]
    m = verify(steps)
    assert m.extra["detection_rate"] == 1.0
    assert m.attack_success_rate == 0.0       # service stayed up
    assert m.false_positives == 0             # isolate during an intrusion is not a FP
    assert m.blue_win is True


def test_verify_counts_false_positives_and_blocks_win():
    steps = [
        StepOutcome(0, True, action=TestAction.RESTART_SERVICE),  # no intrusion -> FP
        StepOutcome(1, True, action="isolate_container"),         # string form -> FP
        StepOutcome(2, True, action=TestAction.CHECK_HEALTH),     # read-only -> not FP
    ]
    m = verify(steps)
    assert m.false_positives == 2
    assert m.blue_win is False


def test_verify_attack_success_when_service_down_during_intrusion():
    steps = [
        StepOutcome(0, True, intrusion_present=True, intrusion_detected=True),
        StepOutcome(1, False, intrusion_present=True, intrusion_detected=True),
    ]
    m = verify(steps)
    assert m.attack_success_rate == 0.5
    assert m.defender_availability == 0.5
    assert m.blue_win is False  # availability below target


def test_verify_undetected_intrusion_no_win():
    steps = [StepOutcome(0, True, intrusion_present=True, intrusion_detected=False)]
    m = verify(steps)
    assert m.extra["detection_rate"] == 0.0
    assert m.blue_win is False


def test_verify_custom_config_threshold():
    steps = [StepOutcome(0, True), StepOutcome(1, False)]  # availability 0.5
    strict = verify(steps, config=VerifyConfig(availability_target=0.5))
    assert strict.blue_win is True
    lax = verify(steps, config=VerifyConfig(availability_target=0.9))
    assert lax.blue_win is False


def test_verify_empty_raises():
    with pytest.raises(ValueError):
        verify([])


def test_verify_metrics_json_roundtrip():
    m = verify([StepOutcome(0, True)], episode=2)
    back = EpisodeMetrics.from_dict(json.loads(m.to_json()))
    assert back.episode == 2
    assert back.extra["source"] == "sandbox"


def test_summary_event_shape():
    m = verify([StepOutcome(0, True)], episode=1)
    ev = summary_event(m)
    assert str(ev.kind) == "metric"
    d = ev.to_dict()
    assert d["episode"] == 1
    assert d["source"] == "sandbox"


# --------------------------------------------------------------------------- #
# backend bridge (Docker-free, via the fake)
# --------------------------------------------------------------------------- #
def test_apply_test_action_bridges_to_primitives():
    b = FakeBackend()
    net = b.create_network("labnet")
    c = b.start_target("img", "t", net)

    assert b.apply_test_action(TestAction.CHECK_HEALTH, c, net).probe == "service_health"
    assert b.apply_test_action(TestAction.COLLECT_LOGS, c, net).probe == "collect_logs"

    assert b.apply_test_action(TestAction.RESTART_SERVICE, c, net) is None
    assert ("stop_target", "t") in b.calls
    assert b.calls.count(("start_target", "img", "t", "labnet")) == 2  # initial + restart

    assert b.apply_test_action(TestAction.ISOLATE_CONTAINER, c, net) is None
    assert ("disconnect", "t", "labnet") in b.calls
    assert b.apply_test_action(TestAction.RECONNECT_CONTAINER, c, net) is None
    assert ("connect", "t", "labnet") in b.calls
    assert b.apply_test_action(TestAction.NOOP, c, net) is None


def test_apply_test_action_refuses_non_benign():
    b = FakeBackend()
    net = SandboxNetwork("n")
    c = SandboxContainer("t", "img", "n")
    with pytest.raises(ValueError):
        b.apply_test_action("rm -rf /", c, net)


def test_require_available_raises_when_unavailable():
    with pytest.raises(SandboxUnavailable):
        FakeBackend(available=False).require_available()


# --------------------------------------------------------------------------- #
# DockerBackend — no daemon needed for these
# --------------------------------------------------------------------------- #
def test_is_available_returns_bool_and_caches():
    b = DockerBackend()
    v = b.is_available()
    assert isinstance(v, bool)
    assert b.is_available() is v  # cached, second call does not reprobe


def test_docker_methods_raise_when_unavailable():
    b = DockerBackend()
    b._available = False  # force "no daemon" without touching Docker
    with pytest.raises(SandboxUnavailable):
        b.create_network("nemesis-sbx-test")
    with pytest.raises(SandboxUnavailable):
        b.start_target("img", "t", SandboxNetwork("n"))
    with pytest.raises(SandboxUnavailable):
        b.probe(SandboxContainer("t", "img", "n"), "service_health")


def test_docker_refuses_non_internal_network():
    b = DockerBackend()
    b._available = True  # pretend the daemon is up; the check is before any call
    with pytest.raises(SandboxUnavailable):
        b.start_target("img", "t", SandboxNetwork("n", internal=False))


def test_docker_probe_rejects_unknown_probe_name():
    b = DockerBackend()
    b._available = True
    with pytest.raises(ValueError):
        b.probe(SandboxContainer("t", "img", "n"), "exec_shell")


# --------------------------------------------------------------------------- #
# Real-container integration — SKIPS cleanly when no Docker daemon is available.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not DockerBackend().is_available(),
    reason="Docker daemon not available; skipping real-container sandbox test",
)
def test_docker_network_create_and_remove_roundtrip():
    # Benign: creates and removes an isolated network only (no containers).
    b = DockerBackend()
    net = b.create_network("nemesis-sbx-pytest-net")
    try:
        assert net.internal is True
    finally:
        b.remove_network(net)
