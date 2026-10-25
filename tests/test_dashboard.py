"""Tests for the observability dashboard (no browser, no CybORG).

Covers the log-reading + metric-time-series computation, the event tailer and
file-feed fan-out, and the HTTP request handlers against a server bound to an
ephemeral port and torn down in each test.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time

import pytest

from nemesis.dashboard.server import (
    EventTailer,
    FileFeed,
    build_server,
    metrics_timeseries,
    read_metrics,
    rolling_mean,
)
from nemesis.platform.events import EventKind, EventLog
from nemesis.platform.metrics import EpisodeMetrics, MetricsLog


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def write_metrics(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        log = MetricsLog(fh)
        for r in rows:
            log.write(r)


def write_events(path, specs):
    with open(path, "w", encoding="utf-8") as fh:
        log = EventLog(fh)
        for kind, text, extra in specs:
            log.log(kind, text, **extra)


def wait_for(predicate, timeout=4.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def run_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


# --------------------------------------------------------------------------- #
# rolling mean / metric time-series
# --------------------------------------------------------------------------- #
def test_rolling_mean_skips_none_gaps():
    out = rolling_mean([1.0, None, 3.0], window=2)
    assert out[0] == 1.0
    assert out[1] == 1.0          # window [1, None] -> mean of present = 1
    assert out[2] == 3.0          # window [None, 3] -> 3


def test_read_metrics_roundtrip_and_tolerates_partial_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [
        EpisodeMetrics(episode=1, steps=10, blue_reward=-2.0, blue_win=False),
        EpisodeMetrics(episode=2, steps=10, blue_reward=3.0, blue_win=True),
    ])
    # Simulate a live writer mid-flush: a trailing line with no newline.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"episode": 3, "steps": 10, "blue_reward"')  # incomplete
    rows = read_metrics(str(path))
    assert [r.episode for r in rows] == [1, 2]   # partial row skipped


def test_read_metrics_missing_file_is_empty(tmp_path):
    assert read_metrics(str(tmp_path / "nope.jsonl")) == []
    assert read_metrics(None) == []


def test_metrics_timeseries_series_and_cumulative_winrate(tmp_path):
    rows = [
        EpisodeMetrics(episode=1, steps=10, blue_reward=-1.0, red_reward=1.0,
                       attack_success_rate=0.8, defender_availability=0.5,
                       false_positives=2, blue_win=False),
        EpisodeMetrics(episode=2, steps=10, blue_reward=0.0, red_reward=0.5,
                       attack_success_rate=0.6, defender_availability=0.7,
                       false_positives=1, blue_win=True),
        EpisodeMetrics(episode=3, steps=10, blue_reward=2.0, red_reward=-0.5,
                       attack_success_rate=0.4, defender_availability=0.9,
                       false_positives=0, blue_win=True),
    ]
    ts = metrics_timeseries(rows, window=2)

    assert ts["count"] == 3 and ts["window"] == 2
    # raw series carry one point per episode.
    assert [p["x"] for p in ts["raw"]["blue_reward"]] == [1, 2, 3]
    assert [p["y"] for p in ts["raw"]["blue_reward"]] == [-1.0, 0.0, 2.0]
    # rolling blue_reward window 2: [-1, mean(-1,0)=-0.5, mean(0,2)=1.0]
    assert [round(p["y"], 3) for p in ts["rolling"]["blue_reward"]] == [-1.0, -0.5, 1.0]
    # cumulative win rate: 0/1, 1/2, 2/3
    cum = [round(p["y"], 3) for p in ts["cumulative"]["blue_win_rate"]]
    assert cum == [0.0, 0.5, 0.667]
    # win raw is 0/1 per episode.
    assert [p["y"] for p in ts["raw"]["blue_win"]] == [0.0, 1.0, 1.0]
    # summary comes from platform.summarize.
    assert ts["summary"]["episodes"] == 3
    assert round(ts["summary"]["blue_win_rate"], 3) == 0.667


def test_metrics_timeseries_empty_has_no_summary():
    ts = metrics_timeseries([], window=5)
    assert ts["count"] == 0
    assert ts["summary"] is None
    assert ts["raw"]["blue_reward"] == []


def test_metrics_timeseries_skips_none_metric_points():
    rows = [
        EpisodeMetrics(episode=1, steps=10, blue_reward=1.0, attack_success_rate=None),
        EpisodeMetrics(episode=2, steps=10, blue_reward=2.0, attack_success_rate=0.5),
    ]
    ts = metrics_timeseries(rows)
    # the None attack_success_rate on episode 1 produces no point.
    assert [p["x"] for p in ts["raw"]["attack_success_rate"]] == [2]


# --------------------------------------------------------------------------- #
# event tailer + file feed
# --------------------------------------------------------------------------- #
def test_event_tailer_incremental_and_partial_line(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(path, [
        (EventKind.SYSTEM, "Episode 1 started", {"episode": 1}),
        (EventKind.RED, "Exploit host_6", {"episode": 1}),
    ])
    tailer = EventTailer(str(path))
    first = tailer.read_new()
    assert [str(e.kind) for e in first] == ["system", "red"]
    assert tailer.read_new() == []           # nothing new

    # Append a complete line and a half-written one.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": 2, "t": 1.0, "kind": "reward", "text": "blue +1"}) + "\n")
        fh.write('{"id": 3, "kind": "blue"')   # partial, no newline
    nxt = tailer.read_new()
    assert [str(e.kind) for e in nxt] == ["reward"]   # partial line withheld

    # Finish the partial line.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(', "text": "Remove host_2"}\n')
    done = tailer.read_new()
    assert [e.text for e in done] == ["Remove host_2"]


def test_event_tailer_resets_on_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(path, [(EventKind.SYSTEM, "old run", {})])
    tailer = EventTailer(str(path))
    assert len(tailer.read_new()) == 1
    # A following tailer observes the file shrink when a new run truncates it
    # (the frequent poll catches the moment it drops below the old offset).
    with open(path, "w"):
        pass                              # truncated to empty
    assert tailer.read_new() == []        # shrink detected -> offset reset to 0
    write_events(path, [(EventKind.SYSTEM, "new run A", {}), (EventKind.RED, "new run B", {})])
    again = tailer.read_new()
    assert [e.text for e in again] == ["new run A", "new run B"]


def test_file_feed_follow_picks_up_appends(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(path, [(EventKind.SYSTEM, "Episode 1 started", {"episode": 1})])
    feed = FileFeed(str(path), follow=True, poll=0.05).start()
    try:
        assert wait_for(lambda: feed.since(-1, timeout=0.1))
        got = feed.since(-1, timeout=0.1)
        assert got[0]["text"] == "Episode 1 started"
        last = got[-1]["id"]
        # append while the feed is following
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": 1, "t": 2.0, "kind": "red", "text": "Exploit"}) + "\n")
        fresh = wait_for(lambda: feed.since(last, timeout=0.1))
        assert fresh and fresh[-1]["text"] == "Exploit"
    finally:
        feed.stop()


# --------------------------------------------------------------------------- #
# HTTP handlers
# --------------------------------------------------------------------------- #
@pytest.fixture
def running_server(tmp_path):
    events = tmp_path / "events.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    write_events(events, [
        (EventKind.SYSTEM, "Episode 1 started", {"episode": 1}),
        (EventKind.RED, "Exploit host_6", {"episode": 1, "step": 3}),
        (EventKind.REWARD, "Blue reward +1", {"episode": 1, "reward": 1.0}),
    ])
    write_metrics(metrics, [
        EpisodeMetrics(episode=1, steps=10, blue_reward=-1.0, red_reward=1.0,
                       attack_success_rate=0.8, defender_availability=0.6,
                       false_positives=2, blue_win=False),
        EpisodeMetrics(episode=2, steps=10, blue_reward=1.5, red_reward=0.2,
                       attack_success_rate=0.5, defender_availability=0.8,
                       false_positives=0, blue_win=True),
    ])
    server, feed = build_server(str(events), str(metrics), host="127.0.0.1", port=0,
                                follow=False, poll=0.05, window=5)
    port = server.server_address[1]
    run_server(server)
    try:
        yield port
    finally:
        feed.stop()
        server.shutdown()
        server.server_close()


def test_metrics_endpoint_returns_timeseries(running_server):
    conn = http.client.HTTPConnection("127.0.0.1", running_server, timeout=3)
    conn.request("GET", "/metrics")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "application/json"
    payload = json.loads(resp.read())
    conn.close()
    assert payload["count"] == 2
    assert payload["summary"]["blue_win_rate"] == 0.5
    assert [p["y"] for p in payload["raw"]["blue_reward"]] == [-1.0, 1.5]
    assert payload["cumulative"]["blue_win_rate"][-1]["y"] == 0.5


def test_health_endpoint(running_server):
    conn = http.client.HTTPConnection("127.0.0.1", running_server, timeout=3)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    payload = json.loads(resp.read())
    conn.close()
    assert payload["ok"] is True
    assert payload["metrics_count"] == 2


def test_serves_dashboard_index(running_server):
    conn = http.client.HTTPConnection("127.0.0.1", running_server, timeout=3)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    assert resp.status == 200
    assert "NEMESIS / DASHBOARD" in body


def test_events_endpoint_streams_logged_events(running_server):
    # /events never closes, so read raw with a socket timeout.
    sock = socket.create_connection(("127.0.0.1", running_server), timeout=3)
    sock.settimeout(3)
    sock.sendall(b"GET /events HTTP/1.1\r\nHost: localhost\r\n\r\n")
    chunks = []
    deadline = time.time() + 3
    try:
        while time.time() < deadline:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
            if b"Episode 1 started" in b"".join(chunks):
                break
    except socket.timeout:
        pass
    finally:
        sock.close()
    blob = b"".join(chunks).decode("utf-8", "replace")
    assert "text/event-stream" in blob
    assert "Episode 1 started" in blob
    assert "Exploit host_6" in blob
