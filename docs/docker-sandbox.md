# Running Nemesis on Docker Agentic Platform

Docker Agentic Platform sandboxes give you a cloud Linux machine billed per second against
your Docker credits. Nemesis uses them in two ways:

- **RL training** (`scripts/sandbox_train.sh`). Needs no LLM, API key, or Claude account.
- **Claude Code in the sandbox** (`scripts/sandbox_claude.sh`), on Sonnet 5.5 and billed to
  your Claude subscription, for when you want an agent working on the code there.

Docker credits pay only for sandbox compute. Claude usage comes out of your subscription.

## 1. One-time: create the secrets

On your own machine:

1. **Claude subscription token.** Run `claude setup-token`, sign in with your Claude
   subscription, and copy the long-lived token it prints.
2. **GitHub token.** Create a fine-grained personal access token at
   <https://github.com/settings/personal-access-tokens>. Give it access to `KarthV01/Nemesis`
   only, with **Contents: Read and write**. `sandbox_train.sh` uses it to push results.

In Docker Agentic Platform, go to **Secrets** and add:

| Secret | Environment variable name | Value |
| --- | --- | --- |
| Custom secret | `CLAUDE_CODE_OAUTH_TOKEN` | the `claude setup-token` output |
| GitHub | `GITHUB_TOKEN` | the GitHub token |

Don't add an Anthropic API key unless you want API billing: Claude Code uses
`ANTHROPIC_API_KEY` ahead of the subscription token. `sandbox_claude.sh` drops it whenever
`CLAUDE_CODE_OAUTH_TOKEN` is present, as a safeguard.

## 2. Launch a sandbox

Choose **New**, then:

- **Kit:** `shell`. The Claude Code kit asks for an Anthropic API key, while the shell kit
  works with the subscription token, and `sandbox_claude.sh` installs Claude Code itself.
- **Secrets:** attach `CLAUDE_CODE_OAUTH_TOKEN` and `GITHUB_TOKEN`.
- **Compute:** CybORG training is CPU-bound, so more CPUs mean more rollout workers. The
  script uses every core except one.
- **Run for:** at least the training budget plus about 10 minutes. The default 1 hour fits
  the default 50-minute budget. You can't change the timer after launch, and files that
  exist only in the sandbox are lost when it's deleted.

## 3. In the sandbox shell

```bash
git clone --recurse-submodules -b setup/docker-sandbox https://github.com/KarthV01/Nemesis
cd Nemesis
bash setup.sh          # ~5 min: Python 3.10 venv, CPU-only PyTorch, CybORG, smoke test
```

(Use `-b main` once this branch has been merged.)

### Train

```bash
# Detached, so it survives the browser terminal disconnecting
RUN_MINUTES=50 nohup bash scripts/sandbox_train.sh > train.out 2>&1 &
tail -f train.out
```

Arguments after the script name pass through to `scripts/train_ppo_rllib.py`, for example
`--steps 500 --batch-size 8000 --workers 7`. When the budget runs out, the script saves a
final checkpoint and pushes `runs/<run-id>/` (checkpoint, `metrics.csv`, `train.log`, and
the commit it trained from) to a new GitHub branch, `runs/<run-id>`. To get it locally:

```bash
git fetch origin runs/<run-id> && git checkout origin/runs/<run-id> -- runs/<run-id>
```

If the push fails, the script prints an `sbx --cloud cp` command that copies the results
to your machine. That needs the `sbx` CLI (0.45.0 or later) on your machine.

### Claude Code on Sonnet 5.5

```bash
bash scripts/sandbox_claude.sh
```

This installs Claude Code if needed, sets `claude-sonnet-5-5` in the sandbox's
`~/.claude/settings.json`, and starts Claude using your subscription token. Without the
token, run `/login` inside Claude and choose your subscription. To use another model, set
`CLAUDE_MODEL=<model-id>` in front of the command.

## Not yet verified on the platform

These scripts were tested in a local `ubuntu:24.04` container, not on Docker Agentic
Platform itself. Two details of how the platform handles secrets aren't covered by
Docker's documentation:

- **Subscription auth in cloud sandboxes.** Docker documents `/login` for *local*
  sandboxes and an API key for cloud ones. Docker's secret proxy may give the sandbox a
  placeholder for the custom `CLAUDE_CODE_OAUTH_TOKEN` and swap in the real token only on
  outgoing requests. If Claude reports an auth error, try `/login`.
- **`git push` with the GitHub secret.** If the push in `sandbox_train.sh` fails, copy the
  results out with `sbx --cloud cp` instead.
