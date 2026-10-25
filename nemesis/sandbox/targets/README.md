# Sandbox lab targets — ISOLATED, AUTHORIZED use only

These files define the **lab** that Nemesis sandbox evaluation runs a trained
blue policy against. They are intentionally-vulnerable, so treat them with care.

> ⚠️ **Read this before building anything.**
>
> - These images are **intentionally vulnerable** (reflected XSS, a leaked
>   hard-coded weak secret). They exist **only** as evaluation targets.
> - Run them **only** on the `internal` network defined here. An internal Docker
>   network has **no gateway**, so the containers **cannot reach the host or the
>   internet**, and nothing here publishes a port, so they are **not reachable
>   from the host** either.
> - **Never** expose them: do not add `ports:`, do not attach a non-internal
>   network, do not bind the Docker socket, do not deploy them anywhere real.
> - Only run this against infrastructure you **own and are authorized** to test.
>   The whole point of the sandbox stage is authorized, isolated lab evaluation —
>   not touching any real or third-party system.

## What's here

| Path | Role | Notes |
| --- | --- | --- |
| `vulnerable_web/` | **the vulnerable target** | stdlib Python web app on `:8080`; deliberate reflected-XSS + info-disclosure. |
| `defender/` | **the defensive service** | watches the target over the internal network; reports availability + a simulated intrusion signal to its logs. |
| `docker-compose.yml` | wires both onto one `internal: true` network | no published ports; `cap_drop: ALL`, `no-new-privileges`, read-only rootfs, memory/pids caps. |

The target and defender talk to each other by **service name** over the internal
network. The Nemesis harness never talks to them over a published port — it
observes them with read-only `docker inspect` (health) and `docker logs`
(the defender's `SERVICE-UP` / `SERVICE-DOWN` / `INTRUSION-DETECTED` markers).

## How the lab produces a signal

1. The `defender` polls the target's `/health` and prints `SERVICE-UP` /
   `SERVICE-DOWN` — this is the availability signal.
2. A request to the target's `/greet?name=<attack marker>` (e.g. containing
   `<script>`) is the lab's **simulated** attack trigger: it bumps the target's
   `suspicious_requests` counter. No code runs; nothing is actually exploited.
3. The `defender` sees that counter over `/metrics` and prints
   `INTRUSION-DETECTED` — this is the detection signal.

The Nemesis verifier reads those markers and turns them into
`EpisodeMetrics` (`defender_availability`, `attack_success_rate`, detection).

## Build & run (manual, optional)

Only when you have a running Docker daemon and want to exercise the lab:

```bash
docker compose -f nemesis/sandbox/targets/docker-compose.yml up --build
# ... in another shell, confirm there is NO host-reachable port:
docker compose -f nemesis/sandbox/targets/docker-compose.yml ps
# tear down:
docker compose -f nemesis/sandbox/targets/docker-compose.yml down
```

The Nemesis `DockerBackend` builds/starts these the same way but always with the
`--internal` network and the safety flags above; see `docs/sandbox-eval.md`.
