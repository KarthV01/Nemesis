"""Docker-backed arena job controller used by the dashboard API."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from nemesis.arena.models import MatchRequest, ROLE_AGENTS, TEAM_AGENTS


MATCH_ID = re.compile(r"^[a-f0-9]{12}$")


class ArenaUnavailable(RuntimeError):
    pass


class MatchManager:
    """Starts one locked-down Docker container per match and tracks its artifacts."""

    def __init__(
        self,
        root: str = "runs/arena",
        artifact_root: Optional[str] = None,
        image: str = "nemesis-runtime:local",
        docker_bin: str = "docker",
        max_concurrent: int = 2,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else self.root.parent
        self.image = image
        self.docker_bin = docker_bin
        self.max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict] = {}

    def catalog(self) -> Dict[str, object]:
        return {
            "teams": sorted(TEAM_AGENTS),
            "scenarios": ["cc4"],
            "formats": ["5v5", "1v1"],
            "roles": {side: sorted(roles) for side, roles in ROLE_AGENTS.items()},
            "checkpoints": {
                "red": self._checkpoint_refs("red"),
                "blue": self._checkpoint_refs("blue"),
            },
            "limits": {"steps": {"min": 1, "max": 500}, "max_concurrent": self.max_concurrent},
        }

    def submit(self, request: MatchRequest) -> Dict:
        request.validate()
        available = {
            "red": set(self._checkpoint_refs("red")),
            "blue": set(self._checkpoint_refs("blue")),
        }
        for side, reference in (
            ("red", request.red_checkpoint),
            ("blue", request.blue_checkpoint),
        ):
            if reference and reference not in available[side]:
                raise ValueError(f"unknown {side} checkpoint {reference!r}")
        with self._lock:
            active = sum(j["status"] in ("queued", "running") for j in self._jobs.values())
            if active >= self.max_concurrent:
                raise ArenaUnavailable("arena capacity reached; wait for a running match to finish")
            match_id = uuid.uuid4().hex[:12]
            match_dir = self.root / match_id
            match_dir.mkdir()
            (match_dir / "config.json").write_text(
                json.dumps(request.as_dict(), indent=2), encoding="utf-8"
            )
            job = {
                "id": match_id,
                "status": "queued",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "config": request.as_dict(),
            }
            self._jobs[match_id] = job
        threading.Thread(target=self._run, args=(match_id,), daemon=True).start()
        return dict(job)

    def _run(self, match_id: str) -> None:
        match_dir = self.match_dir(match_id)
        container_name = f"nemesis-arena-{match_id}"
        command = [
            self.docker_bin, "run", "--rm", "--name", container_name,
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", "256", "--memory", "4g", "--cpus", "2",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m",
            "-v", f"{self.artifact_root}:/artifacts:ro",
            "-v", f"{match_dir}:/arena", self.image,
            "scripts/arena_match.py", "--config", "/arena/config.json", "--output", "/arena",
            "--artifacts", "/artifacts",
        ]
        self._update(match_id, status="running", started_at=time.time())
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "container exited unsuccessfully").strip()
                raise ArenaUnavailable(detail[-2000:])
            if not (match_dir / "result.json").is_file():
                raise ArenaUnavailable("match container produced no result")
            self._update(match_id, status="completed", finished_at=time.time())
        except subprocess.TimeoutExpired:
            subprocess.run([self.docker_bin, "rm", "--force", container_name], capture_output=True)
            self._update(match_id, status="failed", error="match exceeded 15 minute limit", finished_at=time.time())
        except (OSError, ArenaUnavailable) as exc:
            self._update(match_id, status="failed", error=str(exc), finished_at=time.time())

    def _update(self, match_id: str, **changes) -> None:
        with self._lock:
            self._jobs[match_id].update(changes)

    def match_dir(self, match_id: str) -> Path:
        if not MATCH_ID.fullmatch(match_id):
            raise KeyError(match_id)
        path = self.root / match_id
        if not path.is_dir():
            raise KeyError(match_id)
        return path

    def get(self, match_id: str) -> Dict:
        with self._lock:
            if match_id not in self._jobs:
                path = self.match_dir(match_id)
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
                result_path = path / "result.json"
                self._jobs[match_id] = {
                    "id": match_id, "status": "completed" if result_path.is_file() else "failed",
                    "created_at": path.stat().st_mtime, "started_at": None, "finished_at": None,
                    "error": None if result_path.is_file() else "incomplete saved match", "config": config,
                }
            job = dict(self._jobs[match_id])
        result_path = self.match_dir(match_id) / "result.json"
        if result_path.is_file():
            job["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        return job

    def list(self) -> List[Dict]:
        ids = {p.name for p in self.root.iterdir() if p.is_dir() and MATCH_ID.fullmatch(p.name)}
        with self._lock:
            ids.update(self._jobs)
        return sorted((self.get(i) for i in ids), key=lambda j: j["created_at"], reverse=True)

    def events_path(self, match_id: str) -> Path:
        return self.match_dir(match_id) / "events.jsonl"

    def _checkpoint_refs(self, side: str) -> List[str]:
        """Return complete, loadable checkpoint directories under the artifact root."""
        def complete(path: Path) -> bool:
            prefix = f"{side}_agent_"
            return any(
                agent.is_dir()
                and agent.name.startswith(prefix)
                and (agent / "latest.json").is_file()
                for agent in path.iterdir()
            )

        refs = []
        side_root = self.artifact_root / side
        if side_root.is_dir():
            best = side_root / "best_policy"
            if best.is_dir() and complete(best):
                refs.append(best.relative_to(self.artifact_root).as_posix())
            for run in side_root.iterdir():
                if not run.is_dir() or run.name == "best_policy":
                    continue
                for checkpoint in run.iterdir():
                    if not checkpoint.is_dir() or not checkpoint.name.startswith("checkpoint_"):
                        continue
                    if complete(checkpoint):
                        refs.append(checkpoint.relative_to(self.artifact_root).as_posix())
        selfplay_root = self.artifact_root / "selfplay"
        if selfplay_root.is_dir():
            for generation in selfplay_root.glob("generation_*"):
                checkpoint = generation / side
                if checkpoint.is_dir() and complete(checkpoint):
                    refs.append(checkpoint.relative_to(self.artifact_root).as_posix())
        if side == "blue":
            legacy = self.artifact_root / "ppo_team"
            if legacy.is_dir() and complete(legacy):
                refs.append("ppo_team/default")
        return sorted(refs)


__all__ = ["ArenaUnavailable", "MatchManager"]
