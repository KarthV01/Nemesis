#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

command_name="${1:-up}"
if [[ $# -gt 0 ]]; then shift; fi
export NEMESIS_RUN_NAME="${NEMESIS_RUN_NAME:-live}"
export NEMESIS_DASHBOARD_PORT="${NEMESIS_DASHBOARD_PORT:-8001}"

case "$command_name" in
  build)
    docker compose build
    ;;
  dashboard)
    docker compose up --build -d dashboard
    echo "Dashboard: http://localhost:${NEMESIS_DASHBOARD_PORT}"
    ;;
  train|up)
    docker compose up --build -d dashboard
    echo "Dashboard: http://localhost:${NEMESIS_DASHBOARD_PORT}"
    docker compose --profile train run --build --rm trainer \
      scripts/train_selfplay.py --name "$NEMESIS_RUN_NAME" \
      --artifacts /artifacts/selfplay "$@"
    ;;
  smoke)
    docker compose up --build -d dashboard
    echo "Dashboard: http://localhost:${NEMESIS_DASHBOARD_PORT}"
    docker compose --profile train run --build --rm trainer \
      scripts/train_selfplay.py --name "$NEMESIS_RUN_NAME" \
      --artifacts /artifacts/selfplay --smoke "$@"
    ;;
  logs)
    docker compose logs -f dashboard
    ;;
  down)
    docker compose down --remove-orphans
    ;;
  *)
    echo "usage: scripts/nemesis.sh {up|train|smoke|dashboard|down|logs|build} [training args...]" >&2
    exit 2
    ;;
esac
