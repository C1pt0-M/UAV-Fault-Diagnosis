param(
    [int]$CaseId = 4521,
    [switch]$PrepareOnly,
    [string]$PlatformRoot = "D:\BuaaAutoTestPlatForm",
    [string]$OutputRoot = "D:\UAV-Data\rfly_sensor_fault_validation",
    [int]$TimeoutMinutes = 10,
    [ValidateRange(1, 5)]
    [int]$MaxAttempts = 3
)

$ErrorActionPreference = "Stop"

function Stop-ProcessTree {
    param([int]$ProcessId)
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId $child.ProcessId
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

$pythonBin = "D:\PX4PSP\Python38\python.exe"
if (-not (Test-Path -LiteralPath $pythonBin -PathType Leaf)) {
    throw "RflySim Python not found: $pythonBin"
}
if ($CaseId -notin (4521..4532)) {
    throw "Invalid CaseId $CaseId. Valid sensor case IDs are 4521..4532."
}

& $pythonBin (Join-Path $PSScriptRoot "data\prepare_buaa_sensor_fault_cases.py") `
    --platform-root $PlatformRoot --output-root $OutputRoot --case-id $CaseId
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare sensor fault case $CaseId."
}

$manifestPath = Join-Path $OutputRoot "sensor_fault_case_manifest.json"
if ($PrepareOnly) {
    Write-Host "[READY] Sensor case $CaseId prepared; simulation was not started."
    exit 0
}

$apiDir = Join-Path $PlatformRoot "AutoTestAPI"
Write-Host "[START] Running sensor fault CaseID $CaseId"
Write-Host "[INFO] Do not operate QGC; the simulator stops after the 15-second fault window."
$completed = $false
$failureMessage = ""
$runLog = ""
for ($attempt = 1; $attempt -le $MaxAttempts -and -not $completed; $attempt++) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
    $runLog = Join-Path $OutputRoot "autotest_$CaseId`_$stamp`_attempt$attempt.log"
    $errorLog = Join-Path $OutputRoot "autotest_$CaseId`_$stamp`_attempt$attempt.error.log"
    Write-Host "[INFO] Attempt $attempt/$MaxAttempts"
    $process = Start-Process -FilePath $pythonBin `
        -ArgumentList @("-u", (Join-Path $apiDir "AutoTestAPI.py")) `
        -WorkingDirectory $apiDir `
        -RedirectStandardOutput $runLog -RedirectStandardError $errorLog `
        -PassThru -WindowStyle Hidden

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    $lastLength = 0
    $lastProgress = Get-Date
    $failureMessage = ""
    try {
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 2
            if (Test-Path -LiteralPath $runLog) {
                $length = (Get-Item -LiteralPath $runLog).Length
                if ($length -gt $lastLength) {
                    $lastLength = $length
                    $lastProgress = Get-Date
                }
                $text = Get-Content -Raw -LiteralPath $runLog -ErrorAction SilentlyContinue
                if ($text -match "All case test finish!") {
                    $completed = $true
                    break
                }
                if (($text -match "\d+s,Crash!" -or $text -match "PX4 DisArmed!") -and $text -notmatch "Start Inject Fault") {
                    $failureMessage = "failed before fault injection"
                    break
                }
                if (((Get-Date) - $lastProgress).TotalSeconds -ge 90) {
                    $failureMessage = "produced no new output for 90 seconds"
                    break
                }
            }
            $process.Refresh()
            if ($process.HasExited) {
                $failureMessage = "AutoTestAPI exited early with code $($process.ExitCode)"
                break
            }
        }
        if (-not $completed -and -not $failureMessage) {
            $failureMessage = "timed out after $TimeoutMinutes minutes"
        }
    } finally {
        $process.Refresh()
        if (-not $process.HasExited) {
            Stop-ProcessTree -ProcessId $process.Id
            Wait-Process -Id $process.Id -Timeout 10 -ErrorAction SilentlyContinue
        }
        if (-not $completed) {
            & (Join-Path $PSScriptRoot "stop_rflysim_sitl.ps1")
        }
    }

    if (-not $completed -and $attempt -lt $MaxAttempts) {
        Write-Warning "Case $CaseId $failureMessage on attempt $attempt. Retrying with a clean PX4 instance."
        Start-Sleep -Seconds 3
    }
}
if (-not $completed) {
    throw "Case $CaseId failed after $MaxAttempts attempts: $failureMessage. See $runLog"
}

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$case = $manifest.cases | Where-Object { [int]$_.CaseID -eq $CaseId }
$caseDir = Join-Path (Join-Path $PlatformRoot "Data") "TestCase_$($case.CaseID)_$($case.UniqueIndex)"
$bias = ([string]$case.bias).Replace('.', 'p')
$out = Join-Path $OutputRoot "sensor_case_$CaseId`_$($case.sensor)_$($case.axis)_bias$bias.csv"
& $pythonBin (Join-Path $PSScriptRoot "data\validate_buaa_sensor_fault.py") `
    --case-dir $caseDir --manifest $manifestPath --case-id $CaseId --out $out
if ($LASTEXITCODE -ne 0) {
    throw "Sensor fault case $CaseId failed quality validation."
}
Write-Host "[OK] Sensor fault case $CaseId passed: $out"
