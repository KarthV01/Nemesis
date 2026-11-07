[CmdletBinding()]
param(
    [ValidateSet("up", "train", "train-team", "smoke", "dashboard", "down", "logs", "build")]
    [string]$Command = "up",
    [string]$RunName = "live",
    [int]$Port = 8001,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainingArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:NEMESIS_RUN_NAME = $RunName
$env:NEMESIS_DASHBOARD_PORT = "$Port"

function Invoke-Compose {
    param([string[]]$Arguments)
    & docker compose @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is required. Install Docker Desktop and start its Linux container engine."
}

switch ($Command) {
    "build" {
        Invoke-Compose @("build")
    }
    "dashboard" {
        Invoke-Compose @("up", "--build", "-d", "dashboard")
        Write-Host "Dashboard: http://localhost:$Port"
    }
    "train" {
        Invoke-Compose @("up", "--build", "-d", "dashboard")
        Write-Host "Dashboard: http://localhost:$Port"
        $args = @(
            "--profile", "train", "run", "--build", "--rm", "trainer",
            "scripts/train_selfplay.py", "--name", $RunName,
            "--artifacts", "/artifacts/selfplay"
        )
        if ($TrainingArgs) { $args += $TrainingArgs }
        Invoke-Compose $args
    }
    "train-team" {
        $args = @(
            "--profile", "train", "run", "--build", "--rm", "trainer",
            "scripts/train_ppo_team.py", "--checkpoint-dir", "/artifacts/ppo_team"
        )
        if ($TrainingArgs) { $args += $TrainingArgs }
        Invoke-Compose $args
    }
    "smoke" {
        Invoke-Compose @("up", "--build", "-d", "dashboard")
        Write-Host "Dashboard: http://localhost:$Port"
        Invoke-Compose @(
            "--profile", "train", "run", "--build", "--rm", "trainer",
            "scripts/train_selfplay.py", "--name", $RunName,
            "--artifacts", "/artifacts/selfplay", "--smoke"
        )
    }
    "up" {
        Invoke-Compose @("up", "--build", "-d", "dashboard")
        Write-Host "Dashboard: http://localhost:$Port"
        Invoke-Compose @("--profile", "train", "run", "--build", "--rm", "trainer")
    }
    "logs" {
        Invoke-Compose @("logs", "-f", "dashboard")
    }
    "down" {
        Invoke-Compose @("down", "--remove-orphans")
    }
}
