param(
    [int[]]$CaseId = @(4501),
    [switch]$All,
    [switch]$PrepareOnly,
    [string]$PlatformRoot = "D:\BuaaAutoTestPlatForm",
    [string]$OutputRoot = "D:\UAV-Data\rfly_motor_fault_validation",
    [int]$TimeoutMinutes = 10
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonBin = "D:\PX4PSP\Python38\python.exe"
if (-not (Test-Path -LiteralPath $pythonBin -PathType Leaf)) {
    throw "RflySim Python not found: $pythonBin"
}

$validCaseIds = 4501..4512
if ($All) {
    $selectedCaseIds = $validCaseIds
} else {
    $selectedCaseIds = @($CaseId)
}
$invalidCaseIds = @($selectedCaseIds | Where-Object { $_ -notin $validCaseIds })
if ($invalidCaseIds.Count -gt 0) {
    throw "Invalid CaseId: $($invalidCaseIds -join ', '). Valid IDs are 4501..4512."
}

$prepareArgs = @(
    (Join-Path $PSScriptRoot "data\prepare_buaa_motor_fault_cases.py"),
    "--platform-root", $PlatformRoot,
    "--output-root", $OutputRoot
)
if ($All) {
    $prepareArgs += "--all"
} else {
    foreach ($id in $selectedCaseIds) {
        $prepareArgs += @("--case-id", "$id")
    }
}
& $pythonBin @prepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare motor fault cases."
}

$manifestPath = Join-Path $OutputRoot "motor_fault_case_manifest.json"
if ($PrepareOnly) {
    Write-Host "[READY] Motor cases prepared; simulation was not started."
    Write-Host "[INFO] Manifest: $manifestPath"
    exit 0
}

Write-Host ""
Write-Host "[START] Running Buaa motor fault cases: $($selectedCaseIds -join ', ')"
Write-Host "[INFO] Fault model: FaultModel450d_new"
Write-Host "[INFO] Do not operate QGC; AutoTestAPI controls takeoff and injection."
Write-Host "[INFO] The simulator is stopped automatically after the 15-second fault window."

$apiDir = Join-Path $PlatformRoot "AutoTestAPI"
$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runLog = Join-Path $OutputRoot "autotest_$($selectedCaseIds -join '-')_$runStamp.log"
$runErrorLog = Join-Path $OutputRoot "autotest_$($selectedCaseIds -join '-')_$runStamp.error.log"
$autoTest = Start-Process -FilePath $pythonBin `
    -ArgumentList @("-u", (Join-Path $apiDir "AutoTestAPI.py")) `
    -WorkingDirectory $apiDir `
    -RedirectStandardOutput $runLog `
    -RedirectStandardError $runErrorLog `
    -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$completed = $false
$lastLogLength = 0
$lastProgressAt = Get-Date
try {
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if (Test-Path -LiteralPath $runLog) {
            $logLength = (Get-Item -LiteralPath $runLog).Length
            if ($logLength -gt $lastLogLength) {
                $lastLogLength = $logLength
                $lastProgressAt = Get-Date
            }
            $runText = Get-Content -Raw -LiteralPath $runLog -ErrorAction SilentlyContinue
            if ($runText -match "All case test finish!") {
                $completed = $true
                break
            }
            if ($runText -match "\d+s,Crash!" -and $runText -notmatch "Start Inject Fault") {
                throw "Case crashed before fault injection; this run is invalid. See $runLog"
            }
            if ($runText -match "PX4 DisArmed!" -and $runText -notmatch "Start Inject Fault") {
                throw "PX4 disarmed before fault injection; this run is invalid. See $runLog"
            }
            if (((Get-Date) - $lastProgressAt).TotalSeconds -ge 90) {
                throw "Buaa AutoTestAPI produced no new log output for 90 seconds. See $runLog and $runErrorLog"
            }
        }
        $autoTest.Refresh()
        if ($autoTest.HasExited) {
            throw "Buaa AutoTestAPI exited before completion with code $($autoTest.ExitCode). See $runLog and $runErrorLog"
        }
    }
    if (-not $completed) {
        throw "Buaa AutoTestAPI timed out after $TimeoutMinutes minutes. See $runLog and $runErrorLog"
    }
} finally {
    $autoTest.Refresh()
    if (-not $autoTest.HasExited) {
        Stop-Process -Id $autoTest.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $autoTest.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
    if (-not $completed) {
        & (Join-Path $PSScriptRoot "stop_rflysim_sitl.ps1")
    }
}
Write-Host "[INFO] AutoTestAPI completed; cleaned up its lingering receiver threads."
Write-Host "[INFO] Run log: $runLog"

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
foreach ($id in $selectedCaseIds) {
    $case = $manifest.cases | Where-Object { [int]$_.CaseID -eq $id }
    if (-not $case) {
        throw "CaseId $id is missing from manifest."
    }
    $caseDir = Join-Path (Join-Path $PlatformRoot "Data") "TestCase_$($case.CaseID)_$($case.UniqueIndex)"
    $outPath = Join-Path $OutputRoot ("motor_case_{0}_m{1}_eff{2}.csv" -f $id, $case.motor_index, ([string]$case.efficiency_coefficient).Replace('.', 'p'))
    & $pythonBin (Join-Path $PSScriptRoot "data\validate_buaa_motor_fault.py") `
        --case-dir $caseDir `
        --manifest $manifestPath `
        --case-id $id `
        --out $outPath
    if ($LASTEXITCODE -ne 0) {
        throw "Motor fault case $id failed quality validation. See $($outPath -replace '\.csv$', '.quality.json')"
    }
}

Write-Host "[OK] Motor fault collection passed: $($selectedCaseIds -join ', ')"
Write-Host "[OK] Dataset output: $OutputRoot"
