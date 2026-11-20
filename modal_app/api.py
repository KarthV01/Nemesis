"""Persistent Modal API for Vercel's static Nemesis dashboard.

Vercel serves only browser assets. Match simulation and checkpoint access stay
in isolated Modal containers with the shared artifact Volume mounted. Launches
require a bearer token from the ``nemesis-api`` Modal Secret; read-only catalog,
status, replay, and event endpoints may be viewed publicly.
"""

import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Dict

from modal_app.images import (
    ARTIFACTS_PATH,
    MODAL_AVAILABLE,
    VOLUME_MOUNT_PATH,
    app,
    image,
    volume,
)
from nemesis.arena.controller import MatchManager
from nemesis.arena.models import MatchRequest


ARENA_PATH = Path(ARTIFACTS_PATH) / "arena"
MATCH_ID_LENGTH = 12
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _match_dir(match_id: str) -> Path:
    if len(match_id) != MATCH_ID_LENGTH or any(c not in "0123456789abcdef" for c in match_id):
        raise KeyError(match_id)
    path = ARENA_PATH / match_id
    if not path.is_dir():
        raise KeyError(match_id)
    return path


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _job(match_id: str) -> Dict:
    match_dir = _match_dir(match_id)
    job_path = match_dir / "job.json"
    if job_path.is_file():
        job = _read_json(job_path)
    else:
        job = {
            "id": match_id,
            "status": "completed" if (match_dir / "result.json").is_file() else "failed",
            "error": None,
            "config": _read_json(match_dir / "config.json"),
        }
    result_path = match_dir / "result.json"
    if result_path.is_file():
        job["result"] = _read_json(result_path)
    return job


def _catalog() -> Dict:
    manager = MatchManager(
        str(ARENA_PATH), artifact_root=ARTIFACTS_PATH, max_concurrent=4
    )
    return manager.catalog()


def _empty_metrics() -> Dict:
    fields = (
        "blue_reward",
        "red_reward",
        "attack_success_rate",
        "defender_availability",
        "false_positives",
        "blue_win",
    )
    return {
        "count": 0,
        "window": 20,
        "summary": None,
        "raw": {field: [] for field in fields},
        "rolling": {**{field: [] for field in fields[:-1]}, "blue_win_rate": []},
        "cumulative": {"blue_win_rate": []},
    }


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise KeyError(run_id)
    path = Path(ARTIFACTS_PATH) / "experiments" / run_id
    if not path.is_dir():
        raise KeyError(run_id)
    return path


def _runs() -> list[Dict]:
    root = Path(ARTIFACTS_PATH) / "experiments"
    if not root.is_dir():
        return []
    runs = []
    for path in root.iterdir():
        if not path.is_dir() or not RUN_ID.fullmatch(path.name):
            continue
        events_path = path / "events.jsonl"
        completed = False
        if events_path.is_file():
            lines = events_path.read_text(encoding="utf-8").splitlines()
            if lines:
                try:
                    completed = "complete" in json.loads(lines[-1]).get("text", "").lower()
                except json.JSONDecodeError:
                    pass
        runs.append(
            {
                "id": path.name,
                "status": "completed" if completed else "running",
                "updated_at": path.stat().st_mtime,
                "has_events": events_path.is_file(),
                "has_metrics": (path / "evaluations.jsonl").is_file(),
            }
        )
    return sorted(runs, key=lambda value: value["updated_at"], reverse=True)


if MODAL_AVAILABLE:  # pragma: no cover - requires a deployed Modal app
    import modal
    # Importing registers the training functions on the same App, so deploying
    # this module publishes the API and all cloud compute entrypoints together.
    from modal_app.train import train_blue, train_red, train_selfplay  # noqa: F401

    api_secret = modal.Secret.from_name("nemesis-api")

    @app.function(
        image=image,
        volumes={VOLUME_MOUNT_PATH: volume},
        cpu=2.0,
        memory=4096,
        timeout=15 * 60,
        max_containers=4,
        block_network=True,
    )
    def run_arena_match(payload: Dict, match_id: str):
        """Run one untrusted-policy match in its own network-blocked container."""
        from nemesis.arena.runner import run_match

        match_dir = _match_dir(match_id)
        job_path = match_dir / "job.json"
        job = _read_json(job_path)
        job.update(status="running", started_at=time.time())
        _write_json(job_path, job)
        volume.commit()
        try:
            request = MatchRequest.from_dict(payload)
            result = run_match(
                request,
                match_dir,
                artifact_root=Path(ARTIFACTS_PATH),
                sync=volume.commit,
            )
            job.update(
                status="completed",
                finished_at=time.time(),
                result=result,
            )
        except Exception as exc:
            job.update(
                status="failed",
                finished_at=time.time(),
                error=str(exc)[-2000:],
            )
        _write_json(job_path, job)
        volume.commit()
        return job

    @app.function(
        image=image,
        volumes={VOLUME_MOUNT_PATH: volume},
        secrets=[api_secret],
        cpu=0.5,
        memory=512,
        timeout=10 * 60,
        min_containers=0,
        max_containers=4,
    )
    @modal.asgi_app()
    def dashboard_api():
        import asyncio

        from fastapi import FastAPI, Header, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse

        api = FastAPI(title="Nemesis Arena API", version="1")
        configured_origins = os.environ.get("NEMESIS_ALLOWED_ORIGINS", "*")
        origins = [value.strip() for value in configured_origins.split(",") if value.strip()]
        api.add_middleware(
            CORSMiddleware,
            allow_origins=origins or ["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

        def reload_volume() -> None:
            try:
                volume.reload()
            except RuntimeError:
                pass

        def require_token(authorization: str | None) -> None:
            expected = os.environ.get("NEMESIS_API_TOKEN")
            supplied = authorization.removeprefix("Bearer ") if authorization else ""
            if not expected or not secrets.compare_digest(supplied, expected):
                raise HTTPException(status_code=401, detail="valid launch token required")

        @api.get("/health")
        def health():
            return {"ok": True, "compute": "modal", "policy_runtime": "pytorch"}

        @api.get("/api/arena")
        def catalog():
            reload_volume()
            return _catalog()

        @api.get("/api/matches")
        def matches():
            reload_volume()
            ARENA_PATH.mkdir(parents=True, exist_ok=True)
            jobs = []
            for path in ARENA_PATH.iterdir():
                if path.is_dir():
                    try:
                        jobs.append(_job(path.name))
                    except (KeyError, OSError, ValueError, json.JSONDecodeError):
                        continue
            jobs.sort(key=lambda value: value.get("created_at", 0), reverse=True)
            return {"matches": jobs}

        @api.post("/api/matches", status_code=202)
        async def submit(request: Request, authorization: str | None = Header(default=None)):
            require_token(authorization)
            try:
                parsed = MatchRequest.from_dict(await request.json())
                available = _catalog()["checkpoints"]
                for side, reference in (
                    ("red", parsed.red_checkpoint),
                    ("blue", parsed.blue_checkpoint),
                ):
                    if reference and reference not in available[side]:
                        raise ValueError(f"unknown {side} checkpoint {reference!r}")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            match_id = uuid.uuid4().hex[:MATCH_ID_LENGTH]
            match_dir = ARENA_PATH / match_id
            match_dir.mkdir(parents=True)
            payload = parsed.as_dict()
            _write_json(match_dir / "config.json", payload)
            job = {
                "id": match_id,
                "status": "queued",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "config": payload,
            }
            _write_json(match_dir / "job.json", job)
            volume.commit()
            run_arena_match.spawn(payload, match_id)
            return job

        @api.get("/api/matches/{match_id}")
        def match(match_id: str):
            reload_volume()
            try:
                return _job(match_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="unknown match") from exc

        async def match_events(match_id: str):
            last_id = -1
            while True:
                reload_volume()
                try:
                    job = _job(match_id)
                except KeyError:
                    return
                events_path = _match_dir(match_id) / "events.jsonl"
                emitted = False
                if events_path.is_file():
                    for line in events_path.read_text(encoding="utf-8").splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        event_id = int(event.get("id", -1))
                        if event_id > last_id:
                            yield f"id: {event_id}\ndata: {json.dumps(event)}\n\n"
                            last_id = event_id
                            emitted = True
                if job["status"] in ("completed", "failed") and not emitted:
                    return
                if not emitted:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        @api.get("/api/matches/{match_id}/events")
        def events(match_id: str):
            reload_volume()
            try:
                _job(match_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="unknown match") from exc
            return StreamingResponse(match_events(match_id), media_type="text/event-stream")

        @api.get("/api/runs")
        def runs():
            reload_volume()
            return {"runs": _runs()}

        async def run_events(run_id: str):
            last_id = -1
            while True:
                reload_volume()
                try:
                    run_dir = _run_dir(run_id)
                except KeyError:
                    return
                events_path = run_dir / "events.jsonl"
                emitted = False
                completed = False
                if events_path.is_file():
                    for line in events_path.read_text(encoding="utf-8").splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        event_id = int(event.get("id", -1))
                        completed = "complete" in event.get("text", "").lower()
                        if event_id > last_id:
                            yield f"id: {event_id}\ndata: {json.dumps(event)}\n\n"
                            last_id = event_id
                            emitted = True
                if completed and not emitted:
                    return
                if not emitted:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1.0)

        @api.get("/api/runs/{run_id}/events")
        def training_events(run_id: str):
            reload_volume()
            try:
                _run_dir(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="unknown run") from exc
            return StreamingResponse(run_events(run_id), media_type="text/event-stream")

        @api.get("/api/runs/{run_id}/metrics")
        def training_metrics(run_id: str):
            from nemesis.dashboard.server import metrics_timeseries, read_metrics

            reload_volume()
            try:
                metrics_path = _run_dir(run_id) / "evaluations.jsonl"
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="unknown run") from exc
            return metrics_timeseries(read_metrics(str(metrics_path)))

        @api.get("/events")
        def idle_events():
            async def idle():
                event = {
                    "id": 0,
                    "t": time.time(),
                    "kind": "system",
                    "text": "Select or launch an arena match to begin streaming.",
                }
                yield f"id: 0\ndata: {json.dumps(event)}\n\n"

            return StreamingResponse(idle(), media_type="text/event-stream")

        @api.get("/metrics")
        def metrics():
            return _empty_metrics()

        return api


__all__ = ["ARENA_PATH", "dashboard_api", "run_arena_match"] if MODAL_AVAILABLE else ["ARENA_PATH"]
