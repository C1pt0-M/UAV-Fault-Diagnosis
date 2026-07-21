param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')]
    [string]$SessionName,
    [Parameter(Mandatory = $true)]
    [string]$Maneuver,
    [double]$TargetAltitudeM = 0,
    [double]$TargetSpeedMps = 0,
    [string]$OutputDir = "",
    [int]$ForwardPort = 15560,
    [int]$PymavPort = 14031,
    [switch]$AttachOnly,
    [switch]$KeepSimulationRunning
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "runtime\extra_normals"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$pythonBin = "D:\PX4PSP\Python38\python.exe"
if (-not (Test-Path -LiteralPath $pythonBin -PathType Leaf)) {
    $pythonBin = Join-Path $repoRoot "apps\dashboard\backend\.venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $pythonBin -PathType Leaf)) {
    throw "Python interpreter not found."
}

$px4MavCtrl = "D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py"
if (-not (Test-Path -LiteralPath $px4MavCtrl -PathType Leaf)) {
    throw "PX4MavCtrlV4.py not found: $px4MavCtrl"
}

$csvPath = Join-Path $OutputDir "$SessionName.csv"
$receiverLog = Join-Path $OutputDir "$SessionName.receiver.log"
$receiverErrorLog = Join-Path $OutputDir "$SessionName.receiver.error.log"
$bridgeLog = Join-Path $OutputDir "$SessionName.bridge.log"
$bridgeErrorLog = Join-Path $OutputDir "$SessionName.bridge.error.log"
if (Test-Path -LiteralPath $csvPath) {
    throw "Output already exists: $csvPath"
}
foreach ($path in @($receiverLog, $receiverErrorLog, $bridgeLog, $bridgeErrorLog)) {
    Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
}

foreach ($port in @($ForwardPort, $PymavPort)) {
    $owner = Get-NetUDPEndpoint -LocalPort $port -ErrorAction SilentlyContinue
    if ($owner) {
        throw "UDP port $port is already in use by PID $($owner.OwningProcess)."
    }
}

$oldPythonPath = $env:PYTHONPATH
$pythonPathEntries = @(
    (Join-Path $repoRoot "packages\uav_fault"),
    (Join-Path $repoRoot "packages\rfly_bridge"),
    $repoRoot,
    $oldPythonPath
) | Where-Object { $_ } | Select-Object -Unique
$env:PYTHONPATH = $pythonPathEntries -join [IO.Path]::PathSeparator

$receiver = $null
$bridge = $null
$simulationStarted = $false
$collectionSucceeded = $false
try {
    if (-not $AttachOnly) {
        & (Join-Path $PSScriptRoot "start_rflysim_sitl_qgc.ps1") `
            -VehicleNum 1 -Frame iris -Map LowGPU -UdpSimMode 2 `
            -DashboardPymavUdpPort $PymavPort
        $simulationStarted = $true
        Write-Host "[INFO] Waiting for RflySim/PX4 startup..."
        Start-Sleep -Seconds 12
    }

    $receiverArgs = @(
        "-m", "uav_fault.telemetry.receiver",
        "--port", "$ForwardPort",
        "--csv", $csvPath,
        "--window", "150",
        "--auto-stop", "5",
        "--recv-timeout", "180",
        "--require-ready-field", "pymav_highres_imu_seen",
        "--require-ready-field", "pymav_actuator_control_seen"
    )
    $receiver = Start-Process -FilePath $pythonBin -ArgumentList $receiverArgs `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $receiverLog `
        -RedirectStandardError $receiverErrorLog -PassThru -WindowStyle Hidden

    $bridgeArgs = @(
        (Join-Path $repoRoot "packages\rfly_bridge\rfly_bridge\run_rfly_live_bridge.py"),
        "--px4mavctrl", $px4MavCtrl,
        "--host", "127.0.0.1", "--port", "$ForwardPort",
        "--emit-rate", "50", "--rpm-scale", "10000",
        "--test-id", "1", "--udp-port", "20100", "--udp-mode", "2",
        "--mav-ip", "127.0.0.1",
        "--pymav-address", "udpin:0.0.0.0:$PymavPort",
        "--duration", "0"
    )
    $bridge = Start-Process -FilePath $pythonBin -ArgumentList $bridgeArgs `
        -WorkingDirectory $repoRoot -RedirectStandardOutput $bridgeLog `
        -RedirectStandardError $bridgeErrorLog -PassThru -WindowStyle Hidden

    Write-Host ""
    Write-Host "[READY] Recording normal RflySim telemetry"
    Write-Host "[READY] CSV: $csvPath"
    Write-Host "[ACTION] Wait for QGC Connected, then fly the planned maneuver."
    Write-Host "[ACTION] Finish with Land and Disarm. Recording stops after motors remain idle for 5 seconds."
    Wait-Process -Id $receiver.Id
    $receiver.Refresh()
    if ($receiver.ExitCode -ne 0) {
        throw "Receiver exited with code $($receiver.ExitCode). See $receiverErrorLog"
    }
    $collectionSucceeded = $true
}
finally {
    if ($bridge -and -not $bridge.HasExited) {
        Stop-Process -Id $bridge.Id -Force -ErrorAction SilentlyContinue
    }
    if ($receiver -and -not $receiver.HasExited) {
        Stop-Process -Id $receiver.Id -Force -ErrorAction SilentlyContinue
    }
    $env:PYTHONPATH = $oldPythonPath
    if (-not $collectionSucceeded -and $simulationStarted -and -not $KeepSimulationRunning) {
        & (Join-Path $PSScriptRoot "stop_rflysim_sitl.ps1")
    }
}

if (-not (Test-Path -LiteralPath $csvPath -PathType Leaf)) {
    throw "Collection failed; CSV was not created. See $receiverLog and $bridgeLog"
}

& $pythonBin (Join-Path $PSScriptRoot "data\trim_rfly_prearm.py") `
    --csv $csvPath `
    --max-prearm-seconds 30 `
    --keep-prearm-seconds 20
if ($LASTEXITCODE -ne 0) {
    throw "Pre-Armed normalization failed: $csvPath"
}

& $pythonBin (Join-Path $PSScriptRoot "data\validate_rfly_normal.py") `
    --csv $csvPath `
    --session-name $SessionName `
    --maneuver $Maneuver `
    --target-altitude-m $TargetAltitudeM `
    --target-speed-m-s $TargetSpeedMps
$qualityExit = $LASTEXITCODE

if (-not $KeepSimulationRunning) {
    & (Join-Path $PSScriptRoot "stop_rflysim_sitl.ps1")
}

if ($qualityExit -ne 0) {
    throw "Collection finished but quality validation failed: $csvPath"
}
Write-Host "[OK] Normal session passed quality validation: $csvPath"
