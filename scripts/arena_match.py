"""Container entry point for one isolated arena match."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemesis.arena.models import MatchRequest
from nemesis.arena.runner import run_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_match(MatchRequest.from_dict(payload), Path(args.output))


if __name__ == "__main__":
    main()
