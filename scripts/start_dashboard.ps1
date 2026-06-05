$ErrorActionPreference = "Stop"

$backendScript = Join-Path $PSScriptRoot "start_dashboard_backend.ps1"
$frontendScript = Join-Path $PSScriptRoot "start_dashboard_frontend.ps1"

foreach ($scriptPath in @($backendScript, $frontendScript)) {
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        Write-Error "startup script not found: $scriptPath"
        exit 1
    }
}

Write-Host "[INFO] Launching Dashboard backend and frontend in separate PowerShell windows..."
Start-Process powershell.exe -ArgumentList @("-ExecutionPolicy", "Bypass", "-NoExit", "-File", $backendScript)
Start-Process powershell.exe -ArgumentList @("-ExecutionPolicy", "Bypass", "-NoExit", "-File", $frontendScript)
