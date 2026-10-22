"""Live feed — stream red-vs-blue self-play to a terminal-style web page.

Runs CybORG Enterprise episodes back to back with the Nemesis captains on both
sides and pushes every red and blue move to the browser as Server-Sent Events.
The page itself lives in ``web/index.html``. Run it from the repo root:

    PYTHONPATH=external/cage-challenge-4 python -m nemesis.live

then open http://localhost:8000. Pass ``--delay`` to slow the feed down or
speed it up, and ``--steps`` to change the episode length.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import warnings
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

warnings.filterwarnings("ignore")  # silence CybORG's gym-deprecation noise

from CybORG import CybORG
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from nemesis.agents import BlueCaptain, RedCaptain

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


class Feed:
    """Fan-out of simulation events to every connected browser.

    Keeps a short backlog so a page that connects mid-episode still sees what
    just happened instead of a blank screen.
    """

    def __init__(self, backlog: int = 200):
        self._cond = threading.Condition()
        self._events: deque = deque(maxlen=backlog)
        self._next_id = 0

    def publish(self, kind: str, text: str, **extra) -> None:
        with self._cond:
            self._events.append(
                {"id": self._next_id, "t": time.time(), "kind": kind, "text": text, **extra}
            )
            self._next_id += 1
            self._cond.notify_all()

    def since(self, last_id: int, timeout: float = 15.0) -> list:
        """Events newer than ``last_id``; blocks up to ``timeout`` for new ones."""
        with self._cond:
            self._cond.wait_for(lambda: self._next_id - 1 > last_id, timeout=timeout)
            return [e for e in self._events if e["id"] > last_id]


def _describe(action) -> str:
    """``Analyse restricted_zone_a_subnet_user_host_6`` -> ``Analyse -> restricted_zone_a/user_host_6``."""
    name = type(action).__name__
    target = getattr(action, "hostname", None) or getattr(action, "ip_address", None)
    if target is None:
        return name
    return f"{name} -> {str(target).replace('_subnet_', '/')}"


def simulate(feed: Feed, steps: int, delay: float) -> None:
    """Run episodes forever, publishing each red/blue move as it happens."""
    episode = 0
    while True:
        episode += 1
        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=BlueCaptain,
            red_agent_class=RedCaptain,
            steps=steps,
        )
        env = CybORG(scenario_generator=scenario)
        env.reset()
        agents = env.environment_controller.agent_interfaces
        feed.publish("system", f"Episode {episode} started. {steps} steps, network topology randomized.",
                     episode=episode)

        blue_total = 0.0
        for step in range(1, steps + 1):
            env.step()
            for name in env.active_agents:
                side = name.split("_")[0]
                if side not in ("red", "blue"):
                    continue  # green users are background noise
                action = env.get_last_action(name)
                action = action[0] if isinstance(action, list) else action
                if action is None or isinstance(action, Sleep):
                    continue
                role = getattr(agents[name].agent, "last_role", None)
                success = env.get_observation(name).get("success")
                feed.publish(
                    "attack" if side == "red" else "defend",
                    _describe(action),
                    episode=episode, step=step, agent=name, role=role,
                    success=str(success).split(".")[-1] if success is not None else None,
                )

            reward = sum(env.get_rewards().get("Blue", {}).values())
            blue_total += reward
            if reward:
                feed.publish("reward", f"Blue reward {reward:+g} (episode total {blue_total:+g})",
                             episode=episode, step=step, reward=reward, total=blue_total)
            time.sleep(delay)

        feed.publish("system", f"Episode {episode} finished. Blue total reward {blue_total:+g}.",
                     episode=episode, total=blue_total)


def make_handler(feed: Feed):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def log_message(self, *args):
            pass  # keep the console quiet; the feed is the interesting part

        def do_GET(self):
            if self.path.split("?")[0] != "/events":
                return super().do_GET()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_id = int(self.headers.get("Last-Event-ID", -1))
            try:
                while True:
                    events = feed.since(last_id)
                    if not events:
                        self.wfile.write(b": keepalive\n\n")
                    for event in events:
                        self.wfile.write(f"id: {event['id']}\ndata: {json.dumps(event)}\n\n".encode())
                        last_id = event["id"]
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # browser closed the tab

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=100, help="steps per episode")
    parser.add_argument("--delay", type=float, default=0.75, help="seconds between steps")
    args = parser.parse_args()

    feed = Feed()
    threading.Thread(target=simulate, args=(feed, args.steps, args.delay), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(feed))
    server.daemon_threads = True
    print(f"Nemesis live feed on http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
