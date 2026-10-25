"""Single-agent Gymnasium view of CC4 from the RED (attacker) side.

CC4 ships flat Gym wrappers (``BlueFlatWrapper`` / ``EnterpriseMAE``) only for
the *blue* defenders: both hardcode ``[a for a in env.agents if "blue" in a]``
and build blue-specific observation/action vectors, so neither can expose a red
learner. The red agents are driven the "scripted" way instead -- CybORG calls
``red_agent.get_action(raw_obs_dict, raw_action_space_dict)`` with a nested
observation dict and a *parameterised* action space (an ``action`` sub-dict of
10 red action classes, plus ``ip_address`` / ``subnet`` / ``hostname`` /
``session`` ... parameter sub-dicts). Nothing in CC4 turns that into the fixed
flat vectors Stable-Baselines3 needs.

``SingleRedEnv`` builds that flattening itself, for one red agent, analogous to
``nemesis.blue_env.SingleBlueEnv``:

* **Action space** -- ``MultiDiscrete([n_action_types, n_host_slots])``. The
  first dimension picks one of the 10 CC4 red actions; the second picks a target
  host *slot* from a fixed, scenario-independent universe of hostnames
  (``<subnet>_user_host_<i>`` / ``<subnet>_server_host_<i>`` for the eight
  enterprise subnets). Slots are a fixed, sorted list so an index always means
  the same host across episodes; the hostname is translated to that episode's IP
  at step time. Actions whose target host does not exist this episode, or that
  CC4 rejects, become no-ops (CybORG's own ``InvalidAction`` replacement), so the
  policy simply learns to avoid them -- the same "masking-free" contract SB3 uses
  for blue.

* **Observation** -- a fixed ``Box(0, 1)`` belief vector: three bits per host slot
  (``known`` -- ever seen in red's own observations; ``session`` -- red currently
  holds a shell there; ``root`` -- that shell is privileged), plus the mission
  phase (one-hot, 3), the last action's success (one-hot, 3) and two scalar
  summaries (fraction of slots known / sessioned). ``known`` is accumulated from
  red's observations across the episode (a persistent belief, like the scripted
  FSM red tracks); ``session``/``root`` are read from red's own sessions, which a
  red agent legitimately knows.

* **Reward** -- CC4 scores only the *blue team* (red uses an
  ``EmptyRewardCalculator`` and always sees 0). The attacker's objective is the
  mirror image of the defender's, so the red reward here is the **negated blue
  team reward**: red is paid for exactly the damage blue is penalised for.

The other red agents and the blue agents are *scripted* (both ``SleepAgent`` by
default), so this single learner is the sole attacker and gets clean credit for
the blue-negative reward it causes. Pass ``red_agent_class=FiniteStateRedAgent``
to run the learner inside a full scripted red campaign instead (see the credit
caveat in ``docs/red-training.md``), or a trained/scripted blue to make the
defender adversarial (what the self-play stage will do).

Everything runs entirely inside CybORG's simulation: actions are the simulator's
abstract action objects; there are no real hosts, services or exploits.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

EPISODE_LENGTH = 500

#: The eight CC4 enterprise subnets a red learner can meaningfully target
#: (``internet_subnet`` holds only a root relay host and is excluded). These
#: names are fixed by ``EnterpriseScenarioGenerator`` and never change, which is
#: what lets the action space stay fixed across episodes.
ENTERPRISE_SUBNETS = (
    "contractor_network_subnet",
    "restricted_zone_a_subnet",
    "operational_zone_a_subnet",
    "restricted_zone_b_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "admin_network_subnet",
    "office_network_subnet",
)

#: Host-count caps from ``EnterpriseScenarioGenerator`` (MAX_USER_HOSTS /
#: MAX_SERVER_HOSTS). The per-episode host count is random up to these, so
#: enumerating up to the cap gives a fixed superset of every host that can exist.
MAX_USER_HOSTS = 10
MAX_SERVER_HOSTS = 6

#: Red action classes in a fixed order. Index 0 is the first MultiDiscrete
#: dimension. ``_PARAM`` records how each action is targeted:
#:   "none"     -> no target (Sleep)
#:   "subnet"   -> target the subnet of the chosen host slot
#:   "ip"       -> target the chosen host's IP this episode
#:   "hostname" -> target the chosen host's (stable) hostname
#:   "withdraw" -> needs both IP and hostname
_ACTION_SPECS = (
    ("Sleep", "none"),
    ("DiscoverRemoteSystems", "subnet"),
    ("AggressiveServiceDiscovery", "ip"),
    ("StealthServiceDiscovery", "ip"),
    ("DiscoverDeception", "ip"),
    ("ExploitRemoteService", "ip"),
    ("PrivilegeEscalate", "hostname"),
    ("Impact", "hostname"),
    ("DegradeServices", "hostname"),
    ("Withdraw", "withdraw"),
)


class SingleRedEnv(gym.Env):
    """Expose one CC4 red agent as a standard Gymnasium environment.

    Parameters
    ----------
    agent : str
        Which red agent the policy controls. ``red_agent_0`` (the contractor
        network foothold) is the only red agent active at episode start, so it
        is the sensible learner.
    steps : int
        Episode length. CC4 episodes only end on this limit, so the episode end
        is reported as a truncation, not a termination (mirrors ``SingleBlueEnv``).
    red_agent_class : type
        Scripted class for the *other* red agents. ``SleepAgent`` (default) makes
        the learner the sole attacker (clean reward credit); ``FiniteStateRedAgent``
        runs a full scripted red campaign around the learner.
    blue_agent_class : type
        Scripted blue defender (``SleepAgent`` by default -- a passive bootstrap
        opponent; self-play swaps in a trained blue).
    green_agent_class : type
        Background user-traffic generator (``EnterpriseGreenAgent`` by default).
    reward_scale : float
        Multiplies the (negated blue) reward; handy for keeping PPO returns in a
        comfortable range.
    subnets : tuple[str], optional
        Override the target-host universe (defaults to :data:`ENTERPRISE_SUBNETS`).
        Fewer subnets => smaller action space => faster iteration/tests.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent: str = "red_agent_0",
        steps: int = EPISODE_LENGTH,
        red_agent_class: Optional[type] = None,
        blue_agent_class: Optional[type] = None,
        green_agent_class: Optional[type] = None,
        reward_scale: float = 1.0,
        subnets: Optional[tuple] = None,
    ):
        super().__init__()
        # Imported lazily so merely importing this module does not require CybORG.
        from CybORG.Agents import EnterpriseGreenAgent, SleepAgent

        self.agent = agent
        self.steps = steps
        self.reward_scale = reward_scale
        self._subnets = tuple(subnets) if subnets else ENTERPRISE_SUBNETS
        self._red_agent_class = red_agent_class or SleepAgent
        self._blue_agent_class = blue_agent_class or SleepAgent
        self._green_agent_class = green_agent_class or EnterpriseGreenAgent

        # Fixed, sorted host-slot universe. Index -> hostname is stable across
        # episodes, which is what makes the action space fixed.
        hosts: List[str] = []
        for subnet in self._subnets:
            for i in range(MAX_USER_HOSTS):
                hosts.append(f"{subnet}_user_host_{i}")
            for i in range(MAX_SERVER_HOSTS):
                hosts.append(f"{subnet}_server_host_{i}")
        self.host_slots: List[str] = sorted(hosts)
        self._slot_of = {h: i for i, h in enumerate(self.host_slots)}
        self._slot_subnet = [self._subnet_of_host(h) for h in self.host_slots]

        n_hosts = len(self.host_slots)
        self.action_space = gym.spaces.MultiDiscrete([len(_ACTION_SPECS), n_hosts])
        self._obs_len = n_hosts * 3 + 3 + 3 + 2
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_len,), dtype=np.float32
        )

        self._cyborg = None  # built on first reset
        self._known = np.zeros(n_hosts, dtype=bool)
        self._last_success = 2  # UNKNOWN

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._cyborg is None or seed is not None:
            self._build_cyborg(seed)
        else:
            self._cyborg.reset()
        if self.agent not in self._cyborg.agents:
            raise ValueError(
                f"{self.agent!r} is not one of {[a for a in self._cyborg.agents if 'red' in a]}"
            )
        self._known[:] = False
        self._last_success = 2
        obs = self._cyborg.get_observation(self.agent)
        self._ingest_observation(obs)
        return self._encode_observation(), {}

    def step(self, action):
        cyborg_action = self._decode_action(action)
        self._cyborg.parallel_step({self.agent: cyborg_action})

        blue_reward = sum(self._cyborg.get_rewards().get("Blue", {}).values())
        reward = -float(blue_reward) * self.reward_scale

        truncated = bool(self._cyborg.environment_controller.done)
        obs = self._cyborg.get_observation(self.agent)
        self._ingest_observation(obs)
        info = {"blue_reward": float(blue_reward), "last_action": str(cyborg_action)}
        return self._encode_observation(), reward, False, truncated, info

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    def action_type_labels(self) -> List[str]:
        """Human-readable name of each action-type index (first action dim)."""
        return [name for name, _ in _ACTION_SPECS]

    def describe_action(self, action) -> str:
        """Human-readable meaning of a concrete ``[type, host_slot]`` action."""
        a_idx, h_idx = int(action[0]), int(action[1])
        name, param = _ACTION_SPECS[a_idx]
        if param == "none":
            return name
        if param == "subnet":
            return f"{name} {self._slot_subnet[h_idx]}"
        return f"{name} {self.host_slots[h_idx]}"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_cyborg(self, seed):
        from CybORG import CybORG
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=self._blue_agent_class,
            green_agent_class=self._green_agent_class,
            red_agent_class=self._red_agent_class,
            steps=self.steps,
        )
        self._cyborg = CybORG(scenario_generator=scenario, seed=seed)
        self._cyborg.reset()

    @staticmethod
    def _subnet_of_host(hostname: str) -> str:
        return hostname.rsplit("_user_host_", 1)[0].rsplit("_server_host_", 1)[0]

    def _ip_map(self) -> Dict[str, "object"]:
        """hostname -> IPv4Address for the current episode (only hosts that exist)."""
        return self._cyborg.get_ip_map()

    def _ingest_observation(self, obs: dict) -> None:
        """Fold a raw red observation into the persistent belief + last-success."""
        if not isinstance(obs, dict):
            return
        success = obs.get("success")
        if success is not None:
            name = getattr(success, "name", "")
            self._last_success = {"TRUE": 0, "FALSE": 1}.get(name, 2)

        ip_to_host = {str(ip): h for h, ip in self._ip_map().items()}
        for key, val in obs.items():
            if key in ("success", "action", "message"):
                continue
            hostname = None
            if key in self._slot_of:
                hostname = key
            elif key in ip_to_host:
                hostname = ip_to_host[key]
            elif isinstance(val, dict):
                sysinfo = val.get("System info", {})
                cand = sysinfo.get("Hostname") if isinstance(sysinfo, dict) else None
                if cand in self._slot_of:
                    hostname = cand
            if hostname is not None and hostname in self._slot_of:
                self._known[self._slot_of[hostname]] = True

    def _red_sessions(self):
        """Current sessions held by the controlled red agent: {hostname: is_root}."""
        state = self._cyborg.environment_controller.state
        result: Dict[str, bool] = {}
        for sess in state.sessions.get(self.agent, {}).values():
            host = sess.hostname
            is_root = str(getattr(sess, "username", "")).lower() == "root"
            result[host] = result.get(host, False) or is_root
        return result

    def _encode_observation(self) -> np.ndarray:
        n = len(self.host_slots)
        vec = np.zeros(self._obs_len, dtype=np.float32)
        vec[:n] = self._known

        sessions = self._red_sessions()
        for host, is_root in sessions.items():
            slot = self._slot_of.get(host)
            if slot is not None:
                vec[n + slot] = 1.0
                self._known[slot] = True
                if is_root:
                    vec[2 * n + slot] = 1.0
        vec[:n] = self._known  # refresh after sessions may have added knowledge

        base = 3 * n
        phase = int(getattr(self._cyborg.environment_controller.state, "mission_phase", 0))
        vec[base + min(max(phase, 0), 2)] = 1.0
        vec[base + 3 + self._last_success] = 1.0
        vec[base + 6] = float(self._known.mean())
        vec[base + 7] = float(min(len(sessions), n) / n) if n else 0.0
        return vec

    def _decode_action(self, action):
        from CybORG.Simulator.Actions import Sleep
        from CybORG.Simulator.Actions.AbstractActions import (
            AggressiveServiceDiscovery,
            DegradeServices,
            DiscoverDeception,
            DiscoverRemoteSystems,
            ExploitRemoteService,
            Impact,
            PrivilegeEscalate,
            StealthServiceDiscovery,
        )
        from CybORG.Simulator.Actions.ConcreteActions.Withdraw import Withdraw

        a_idx, h_idx = int(action[0]), int(action[1])
        name, param = _ACTION_SPECS[a_idx]
        if name == "Sleep":
            return Sleep()

        hostname = self.host_slots[h_idx]
        ip_map = self._ip_map()
        session, agent = 0, self.agent

        if param == "subnet":
            state = self._cyborg.environment_controller.state
            subnet_name = self._slot_subnet[h_idx]
            cidr = None
            for k, v in state.subnet_name_to_cidr.items():
                if getattr(k, "value", str(k)) == subnet_name:
                    cidr = v
                    break
            if cidr is None:
                return Sleep()
            return DiscoverRemoteSystems(subnet=cidr, session=session, agent=agent)

        # Host-targeted actions need the host to exist this episode.
        ip = ip_map.get(hostname)
        if param in ("ip", "withdraw") and ip is None:
            return Sleep()
        if param == "hostname" and hostname not in ip_map:
            return Sleep()

        classes = {
            "AggressiveServiceDiscovery": AggressiveServiceDiscovery,
            "StealthServiceDiscovery": StealthServiceDiscovery,
            "DiscoverDeception": DiscoverDeception,
            "ExploitRemoteService": ExploitRemoteService,
            "PrivilegeEscalate": PrivilegeEscalate,
            "Impact": Impact,
            "DegradeServices": DegradeServices,
        }
        if param == "ip":
            return classes[name](session=session, agent=agent, ip_address=ip)
        if param == "hostname":
            return classes[name](hostname=hostname, session=session, agent=agent)
        if param == "withdraw":
            return Withdraw(session=session, agent=agent, ip_address=ip, hostname=hostname)
        return Sleep()
