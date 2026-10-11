# Run offline project tests with a fresh, dedicated pytest temporary directory.
# Never reuse a data directory: pytest clears the directory passed to --basetemp.
[CmdletBinding()]
param(
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $PSScriptRoot
$testsDirectory = Join-Path $projectDirectory "tests"
$pytestTempDirectory = Join-Path ([System.IO.Path]::GetTempPath()) (
    "learning-os-pytest-" + [Guid]::NewGuid().ToString("N")
)

Write-Host "Pytest temporary directory: $pytestTempDirectory"

Push-Location -LiteralPath $projectDirectory
try {
    & $PythonExecutable -B -m pytest $testsDirectory -q --tb=short `
        -p no:cacheprovider --basetemp $pytestTempDirectory
    $testExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $testExitCode
