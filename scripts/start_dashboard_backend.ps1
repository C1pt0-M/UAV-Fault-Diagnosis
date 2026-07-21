$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $repoRoot "apps\\dashboard\\backend"
$uavFaultPackageDir = Join-Path $repoRoot "packages\\uav_fault"
$rflyBridgePackageDir = Join-Path $repoRoot "packages\\rfly_bridge"

if (-not (Test-Path -LiteralPath $backendDir -PathType Container)) {
    Write-Error "backend directory not found: $backendDir"
    exit 1
}

foreach ($packageDir in @($uavFaultPackageDir, $rflyBridgePackageDir)) {
    if (-not (Test-Path -LiteralPath $packageDir -PathType Container)) {
        Write-Error "local package directory not found: $packageDir"
        exit 1
    }
}

if ($env:DASHBOARD_BACKEND_PYTHON) {
    $pythonBin = $env:DASHBOARD_BACKEND_PYTHON
} else {
    $venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $pythonBin = $venvPython
    } else {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCmd) {
            Write-Error "Python not found. Set DASHBOARD_BACKEND_PYTHON or create apps/dashboard/backend/.venv."
            exit 1
        }
        $pythonBin = $pythonCmd.Source
    }
}

$pythonPathEntries = @($uavFaultPackageDir, $rflyBridgePackageDir)
if ($env:PYTHONPATH) {
    $pythonPathEntries += $env:PYTHONPATH.Split([IO.Path]::PathSeparator)
}
$env:PYTHONPATH = ($pythonPathEntries | Where-Object { $_ } | Select-Object -Unique) -join [IO.Path]::PathSeparator

Write-Host "[INFO] Starting Dashboard Backend"
Set-Location -LiteralPath $backendDir
& $pythonBin -m uav_dashboard_api
