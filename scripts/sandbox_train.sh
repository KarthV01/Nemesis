#!/usr/bin/env bash
# One-shot training run for a (cloud) sandbox: set up if needed, train within a time budget,
# then push the checkpoint + logs to a `runs/<run-id>` branch so nothing is lost when the
# sandbox expires.
#
#   bash scripts/sandbox_train.sh                        # 50 min budget, defaults
#   RUN_MINUTES=230 bash scripts/sandbox_train.sh --iterations 2000 --steps 500
#
# Env vars:
#   RUN_MINUTES   training time budget in minutes (default 50; keep it ~10 min under the
#                 sandbox's "Run for" limit to leave time for the final checkpoint and push)
#   PUSH_RESULTS  1 (default) to push results to GitHub; 0 to keep them local only
# Any extra arguments are passed through to scripts/train_ppo_rllib.py.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_MINUTES="${RUN_MINUTES:-50}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)-ppo"
RUN_DIR="runs/$RUN_ID"
MAX_PUSH_MB=90  # GitHub rejects files over 100 MB

if [ ! -x .venv/bin/python ]; then
    bash setup.sh
fi

mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
echo "Run $RUN_ID: budget ${RUN_MINUTES} min, $(nproc) CPUs, args: $*" | tee "$RUN_DIR/train.log"

set +e
.venv/bin/python scripts/train_ppo_rllib.py \
    --checkpoint-dir "$RUN_DIR/checkpoint" \
    --time-budget-min "$RUN_MINUTES" \
    --iterations 100000 \
    "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
echo "Training exited with status $TRAIN_STATUS" | tee -a "$RUN_DIR/train.log"

copy_hint() {
    echo "Copy the results out before the sandbox expires, from your own machine:"
    echo "  sbx --cloud cp <sandbox-name>:$(pwd)/$RUN_DIR ./$RUN_DIR"
}

if [ "$PUSH_RESULTS" != "1" ]; then
    echo "Results saved in $RUN_DIR (PUSH_RESULTS=0, not pushed)."
    copy_hint
    exit "$TRAIN_STATUS"
fi

SIZE_MB=$(du -sm "$RUN_DIR" | cut -f1)
if [ "$SIZE_MB" -gt "$MAX_PUSH_MB" ]; then
    echo "Results are ${SIZE_MB} MB, too large to push to GitHub."
    copy_hint
    exit "$TRAIN_STATUS"
fi

# Commit the results on a fresh branch in a throwaway worktree so the current checkout
# (and whatever branch it is on) is left untouched.
BRANCH="runs/$RUN_ID"
PUSH_TMP="$(mktemp -d)"
PUSH_WT="$PUSH_TMP/results"

cleanup() {
    if git worktree list --porcelain | grep -Fqx "worktree $PUSH_WT"; then
        git worktree remove --force "$PUSH_WT" || true
    fi
    rm -f "$PUSH_TMP/git-askpass.sh"
    rmdir "$PUSH_TMP" 2>/dev/null || true
}
trap cleanup EXIT

git worktree add -q -b "$BRANCH" "$PUSH_WT" HEAD
mkdir -p "$PUSH_WT/runs"
cp -r "$RUN_DIR" "$PUSH_WT/runs/"
git -C "$PUSH_WT" add -f "runs/$RUN_ID"
git -C "$PUSH_WT" \
    -c user.name="${GIT_AUTHOR_NAME:-$(git config user.name || echo 'Nemesis sandbox')}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-$(git config user.email || echo 'sandbox@nemesis.local')}" \
    commit -q -m "Training results for run $RUN_ID"

# Keep the token out of command-line arguments, where process listings could expose it.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    ASKPASS="$PUSH_TMP/git-askpass.sh"
    cat > "$ASKPASS" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    *Username*) printf '%s\n' 'x-access-token' ;;
    *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
    *) exit 1 ;;
esac
EOF
    chmod 700 "$ASKPASS"
fi

PUSH_STATUS=0
if [ -n "${GITHUB_TOKEN:-}" ]; then
    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
        git -C "$PUSH_WT" -c credential.helper= push -q origin "$BRANCH" || PUSH_STATUS=$?
else
    git -C "$PUSH_WT" push -q origin "$BRANCH" || PUSH_STATUS=$?
fi

if [ "$PUSH_STATUS" -eq 0 ]; then
    echo "Pushed results to branch $BRANCH"
else
    echo "Push failed (is a GITHUB_TOKEN secret configured for this sandbox?)."
    copy_hint
fi
git worktree remove --force "$PUSH_WT"
rm -f "$PUSH_TMP/git-askpass.sh"
rmdir "$PUSH_TMP" 2>/dev/null || true
trap - EXIT

if [ "$TRAIN_STATUS" -ne 0 ]; then
    exit "$TRAIN_STATUS"
fi
exit "$PUSH_STATUS"
