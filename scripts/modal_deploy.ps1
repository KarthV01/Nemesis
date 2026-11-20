param(
    [ValidateSet("rolling", "recreate")]
    [string] $Strategy = "rolling"
)

$runner = Join-Path $PSScriptRoot "modal_with_env.ps1"
& $runner -ModalArgs @("deploy", "-m", "modal_app.api", "--strategy", $Strategy)
exit $LASTEXITCODE
