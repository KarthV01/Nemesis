#!/usr/bin/env bash
# Launch Claude Code inside a sandbox on Sonnet 5.5, billed to a Claude subscription.
#
#   bash scripts/sandbox_claude.sh [claude args...]
#
# Auth, in order of preference:
#   1. CLAUDE_CODE_OAUTH_TOKEN (subscription). Generate it on your own machine with
#      `claude setup-token` and add it to the sandbox as a custom secret with that env var name.
#      Any ANTHROPIC_API_KEY is dropped so it can't take precedence and bill the API instead.
#   2. Otherwise Claude Code starts unauthenticated: run /login and choose your subscription.
#
# Override the model with CLAUDE_MODEL=<model-id>.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${CLAUDE_MODEL:-claude-sonnet-5-5}"

if ! command -v claude >/dev/null 2>&1; then
    curl -fsSL https://claude.ai/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi

# Persist the model in the sandbox's *user* settings (not the repo's .claude/), so it also
# applies if Claude Code is started some other way, without changing anyone's local setup.
mkdir -p "$HOME/.claude"
PYTHON="$(command -v python3 || echo .venv/bin/python)"
"$PYTHON" - "$HOME/.claude/settings.json" "$MODEL" <<'EOF'
import json, os, sys
path, model = sys.argv[1], sys.argv[2]
settings = {}
if os.path.exists(path):
    with open(path) as f:
        settings = json.load(f)
settings["model"] = model
with open(path, "w") as f:
    json.dump(settings, f, indent=2)
EOF

if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "Using Claude subscription token (CLAUDE_CODE_OAUTH_TOKEN), model $MODEL"
    unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "WARNING: no CLAUDE_CODE_OAUTH_TOKEN; ANTHROPIC_API_KEY is set, so usage bills the API, not your subscription."
else
    echo "No credentials found. Once Claude Code starts, run /login and pick your Claude subscription."
fi

exec claude --model "$MODEL" "$@"
