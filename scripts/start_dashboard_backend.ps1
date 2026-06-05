$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $repoRoot "apps\\dashboard\\backend"

if (-not (Test-Path -LiteralPath $backendDir -PathType Container)) {
    Write-Error "backend directory not found: $backendDir"
    exit 1
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

Write-Host "[INFO] Backend Python: $pythonBin"
& $pythonBin -c "import reportlab" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] reportlab is not installed in the backend Python environment."
    Write-Host "[WARN] PDF reports will fail until you run: $pythonBin -m pip install -r requirements.txt"
}

Write-Host "[INFO] Starting Dashboard Backend"
Set-Location -LiteralPath $backendDir
& $pythonBin "run.py"
