$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WindowsPath
    )

    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $matches[1].ToLower()
        $rest = $matches[2] -replace "\\", "/"
        return "/mnt/$drive/$rest"
    }

    throw "Path cannot be converted to a WSL path: $WindowsPath"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultWslRepo = Convert-ToWslPath -WindowsPath $repoRoot
$wslRepo = if ($env:UAV_WSL_REPO) { $env:UAV_WSL_REPO } else { $defaultWslRepo }
$distroArgs = @()

if ($env:UAV_WSL_DISTRO) {
    $distroArgs = @("-d", $env:UAV_WSL_DISTRO)
}

$bashCommand = @"
cd '$wslRepo'
if [ -f scripts/start_dashboard.sh ]; then
  bash scripts/start_dashboard.sh
else
  (
    cd dashboard/backend || exit 1
    if [ -x .venv/bin/python ]; then
      .venv/bin/python run.py
    elif command -v python3 >/dev/null 2>&1; then
      python3 run.py
    else
      python run.py
    fi
  ) &
  (
    cd dashboard/frontend || exit 1
    npm run frontend
  ) &
  wait
fi
"@

Write-Host "[INFO] Starting Dashboard from WSL: $wslRepo"
& wsl.exe @distroArgs bash -lc $bashCommand
