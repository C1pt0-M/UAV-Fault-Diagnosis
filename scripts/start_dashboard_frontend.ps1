$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $repoRoot "dashboard\frontend"

if (-not (Test-Path -LiteralPath $frontendDir -PathType Container)) {
    Write-Error "frontend directory not found: $frontendDir"
    exit 1
}

$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if ($null -eq $npmCmd) {
    Write-Error "npm not found. Install Node.js 18+ and ensure npm is on PATH."
    exit 1
}

$commandText = "npm run frontend"
Write-Host "[INFO] Starting Dashboard Frontend"
Set-Location -LiteralPath $frontendDir
& $npmCmd.Source run frontend
