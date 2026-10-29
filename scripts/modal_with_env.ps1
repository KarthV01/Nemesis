param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ModalArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot ".env"

if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing $envFile. Copy .env.example to .env and add your Modal token."
}

foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        continue
    }

    $name = $Matches[1]
    $value = $Matches[2].Trim()
    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
        ($value.StartsWith("'") -and $value.EndsWith("'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    switch ($name) {
        { $_ -in @('token_id', 'MODAL_TOKEN_ID') } {
            Set-Item -Path Env:MODAL_TOKEN_ID -Value $value
        }
        { $_ -in @('token_secret', 'MODAL_TOKEN_SECRET') } {
            Set-Item -Path Env:MODAL_TOKEN_SECRET -Value $value
        }
        default {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

if (-not $env:MODAL_TOKEN_ID -or -not $env:MODAL_TOKEN_SECRET) {
    throw "The .env file must define token_id/token_secret or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET."
}

# Modal's CLI uses Unicode status symbols; force UTF-8 on Windows consoles.
$env:PYTHONUTF8 = "1"

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython -c "import modal" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & $venvPython -m modal @ModalArgs
        exit $LASTEXITCODE
    }
}

& modal @ModalArgs
exit $LASTEXITCODE
