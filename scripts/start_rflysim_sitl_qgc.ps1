param(
    [int]$VehicleNum = 1,
    [int]$StartIndex = 1,
    [string]$Frame = "iris",
    [string]$Map = "LowGPU",
    [string]$UdpSimMode = "2",
    [int]$QgcUdpPort = 14600,
    [int]$DashboardPymavUdpPort = 14031,
    [switch]$NoDirectQgcLink,
    [switch]$NoDashboardPymavLink,
    [switch]$NoQgcConfigPatch,
    [switch]$SkipQGC,
    [switch]$Skip3D
)

$ErrorActionPreference = "Stop"

# LowGPU non-interactive wrapper for D:\PX4PSP\RflySimAPIs\SITLRunLowGPU.bat.
# It keeps the dashboard/QGC direct MAVLink patches while using the LowGPU map.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PspPath = "D:\PX4PSP"
$PspPathLinux = "/mnt/d/PX4PSP"
$LowGpuBatPath = Join-Path $PspPath "RflySimAPIs\SITLRunLowGPU.bat"
$RuntimeDir = Join-Path $RepoRoot "runtime\rflysim"
$RelayScript = Join-Path $PSScriptRoot "rfly_wsl_udp_relay.py"
$Px4BuildDir = Join-Path $PspPath "Firmware\build\px4_sitl_default"
$QgcIniPath = Join-Path $env:APPDATA "QGroundControl.org\QGroundControl.ini"

$env:PSP_PATH = $PspPath
$env:PSP_PATH_LINUX = $PspPathLinux

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

function Convert-ToWslPath {
    param([Parameter(Mandatory=$true)][string]$WindowsPath)
    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    $drive = $resolved.Substring(0, 1).ToLowerInvariant()
    $rest = $resolved.Substring(2).Replace('\', '/')
    return "/mnt/$drive$rest"
}

function Stop-ExistingPx4Launcher {
    $existingLauncher = Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*uav_rflysim_px4_start.sh*" -or $_.CommandLine -like "*rflysim_px4_start.sh*" }

    if ($existingLauncher) {
        Write-Host "[INFO] Stopping existing RflySim PX4 WSL launcher"
        wsl -d RflySim-20.04 -e bash -lc "pkill -9 -f '[p]x4' || true" | Out-Null
        $existingLauncher | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 2
    }
}

function Start-AppIfMissing {
    param(
        [string]$ProcessName,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )

    if (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue) {
        Write-Host "[INFO] $ProcessName already running"
        return
    }

    Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory
    Write-Host "[INFO] Started $ProcessName"
}

function Test-TcpListenPort {
    param([Parameter(Mandatory=$true)][int]$Port)

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Any,
        $Port
    )

    try {
        $listener.Start()
    } catch {
        $socketError = $_.Exception.InnerException
        if ($null -eq $socketError) {
            $socketError = $_.Exception
        }
        throw "CopterSim requires TCP port $Port, but Windows cannot bind it ($($socketError.Message)). Run scripts\repair_rflysim_tcp_4560.ps1 from an Administrator PowerShell, then restart collection."
    } finally {
        try { $listener.Stop() } catch {}
    }
}

function Set-QgcUdpLink {
    param(
        [string]$IniPath,
        [int]$Port,
        [string]$BackupDir
    )

    $iniDir = Split-Path -Parent $IniPath
    New-Item -ItemType Directory -Force -Path $iniDir | Out-Null

    if (Test-Path -LiteralPath $IniPath) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupPath = Join-Path $BackupDir "QGroundControl.ini.$stamp.bak"
        Copy-Item -LiteralPath $IniPath -Destination $backupPath -Force
        Write-Host "[INFO] Backed up QGC config: $backupPath"
        $lines = [System.IO.File]::ReadAllLines($IniPath)
    } else {
        $lines = @("[General]", "SettingsVersion=9")
    }

    $linkLines = @(
        "[LinkConfigurations]",
        "Link0\name=UAV-Dashboard-PX4-$Port",
        "Link0\type=1",
        "Link0\auto=true",
        "Link0\high_latency=false",
        "Link0\port=$Port",
        "Link0\hostCount=0",
        "count=1"
    )

    $out = New-Object 'System.Collections.Generic.List[string]'
    $i = 0
    $replaced = $false

    while ($i -lt $lines.Count) {
        if ($lines[$i] -eq "[LinkConfigurations]") {
            foreach ($line in $linkLines) {
                $out.Add($line)
            }
            $replaced = $true
            $i++
            while ($i -lt $lines.Count -and -not ($lines[$i] -match '^\[.+\]$')) {
                $i++
            }
            continue
        }

        $out.Add($lines[$i])
        $i++
    }

    if (-not $replaced) {
        if ($out.Count -gt 0 -and $out[$out.Count - 1] -ne "") {
            $out.Add("")
        }
        foreach ($line in $linkLines) {
            $out.Add($line)
        }
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($IniPath, [string[]]$out, $utf8NoBom)
    Write-Host "[INFO] Configured QGC UDP auto link on port $Port"
}

if (-not (Test-Path -LiteralPath $PspPath -PathType Container)) {
    throw "RflySim path not found: $PspPath"
}

if (-not (Test-Path -LiteralPath $LowGpuBatPath -PathType Leaf)) {
    throw "RflySim LowGPU launcher not found: $LowGpuBatPath"
}

Stop-ExistingPx4Launcher

Get-Process -Name "CopterSim","CopterSimNoUI" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

for ($i = $StartIndex; $i -lt ($StartIndex + $VehicleNum); $i++) {
    Test-TcpListenPort -Port (4559 + $i)
}

if (-not $SkipQGC -and -not $NoDirectQgcLink -and -not $NoQgcConfigPatch) {
    Get-Process -Name "QGroundControl" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Set-QgcUdpLink -IniPath $QgcIniPath -Port $QgcUdpPort -BackupDir $RuntimeDir
}

if (Test-Path -LiteralPath $Px4BuildDir -PathType Container) {
    $resolvedBuildDir = (Resolve-Path -LiteralPath $Px4BuildDir).Path
    for ($i = $StartIndex; $i -lt ($StartIndex + $VehicleNum); $i++) {
        $instancePath = Join-Path $resolvedBuildDir "instance_$i"
        if (Test-Path -LiteralPath $instancePath) {
            $resolvedInstancePath = (Resolve-Path -LiteralPath $instancePath).Path
            if (-not $resolvedInstancePath.StartsWith($resolvedBuildDir, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove unexpected path: $resolvedInstancePath"
            }
            Remove-Item -LiteralPath $resolvedInstancePath -Recurse -Force
            Write-Host "[INFO] Removed stale PX4 runtime directory: $resolvedInstancePath"
        }
    }
}

if (-not $SkipQGC) {
    Start-AppIfMissing `
        -ProcessName "QGroundControl" `
        -FilePath (Join-Path $PspPath "QGroundControl\QGroundControl.exe") `
        -ArgumentList @("-noComPix") `
        -WorkingDirectory (Join-Path $PspPath "QGroundControl")
}

if (-not $Skip3D) {
    Get-Process -Name "RflySim3D" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Start-Process `
        -FilePath (Join-Path $PspPath "RflySim3D\RflySim3D.exe") `
        -ArgumentList @("-cmd=RflyChangeMapbyName-$Map") `
        -WorkingDirectory (Join-Path $PspPath "RflySim3D")
    Write-Host "[INFO] Started RflySim3D with map=$Map"
    Start-Sleep -Seconds 5
}

$copterArgs = @(
    "1",
    "$StartIndex",
    "-1",
    "0",
    "2",
    "$Map",
    "0",
    "0", "0", "0",
    "1",
    "$UdpSimMode"
)

Start-Process `
    -FilePath (Join-Path $PspPath "CopterSim\CopterSim.exe") `
    -ArgumentList $copterArgs `
    -WorkingDirectory (Join-Path $PspPath "CopterSim")

Write-Host "[INFO] Started CopterSim"
Start-Sleep -Seconds 2

Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*rfly_wsl_udp_relay.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$wslConfigPath = Join-Path $env:USERPROFILE ".wslconfig"
$wslConfigText = if (Test-Path -LiteralPath $wslConfigPath) {
    [System.IO.File]::ReadAllText($wslConfigPath)
} else {
    ""
}
$isMirroredNetworking = $wslConfigText -match '(?im)^\s*networkingMode\s*=\s*mirrored\s*$'
$wslKernelRelease = ((& wsl -d RflySim-20.04 -e uname -r) -join "").Trim()
$isWsl1 = $wslKernelRelease -notmatch '(?i)WSL2|microsoft-standard'
$usesSharedLoopback = $isWsl1 -or $isMirroredNetworking

if ($usesSharedLoopback) {
    $hostIp = "127.0.0.1"
    $wslIp = "127.0.0.1"
    $simForwardPort = 16540
} else {
    $routeOutput = & wsl -d RflySim-20.04 -e bash -lc "/sbin/ip route show default"
    $hostIp = $null
    foreach ($line in $routeOutput) {
        if ($line -match "default via ([0-9.]+)") {
            $hostIp = $Matches[1]
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($hostIp)) {
        throw "Failed to detect WSL gateway IP from: $routeOutput"
    }

    $wslIpOutput = & wsl -d RflySim-20.04 -e bash -lc "hostname -I"
    $wslIp = (($wslIpOutput -join " ").Trim() -split '\s+')[0]
    if ([string]::IsNullOrWhiteSpace($wslIp)) {
        throw "Failed to detect RflySim WSL IP from: $wslIpOutput"
    }
    $simForwardPort = 16539

    $relayPython = Join-Path $RepoRoot "apps\dashboard\backend\.venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $relayPython -PathType Leaf)) {
        $relayPython = (Get-Command python -ErrorAction Stop).Source
    }
    $relayStdout = Join-Path $RuntimeDir "rfly_wsl_udp_relay.log"
    $relayStderr = Join-Path $RuntimeDir "rfly_wsl_udp_relay.error.log"
    $relayProcess = Start-Process `
        -FilePath $relayPython `
        -ArgumentList @(
            $RelayScript,
            "--listen-port", "16539",
            "--copter-port", "16540",
            "--copter-source-port", "17540",
            "--wsl-host", $wslIp,
            "--wsl-port", "17540"
        ) `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $relayStdout `
        -RedirectStandardError $relayStderr `
        -WindowStyle Hidden `
        -PassThru
    Start-Sleep -Milliseconds 500
    if ($relayProcess.HasExited) {
        throw "Rfly WSL UDP relay exited early. See $relayStdout and $relayStderr"
    }
}

$repoRootLinux = Convert-ToWslPath $RepoRoot
$sourceQgcScript = Join-Path $PSScriptRoot "rflysim_sitl_qgc.sh"
$runtimeQgcScript = Join-Path $RuntimeDir "rflysim_sitl_qgc.sh"
$sourceText = [System.IO.File]::ReadAllText($sourceQgcScript)
$sourceText = $sourceText -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($runtimeQgcScript, $sourceText, [System.Text.Encoding]::ASCII)
$runtimeQgcScriptLinux = Convert-ToWslPath $runtimeQgcScript

$bashScript = @"
#!/usr/bin/env bash
set -e
cd $PspPathLinux/Firmware
./BkFile/EnvOri.sh
export PATH="`$HOME/ninja:`$HOME/gcc-arm-none-eabi-7-2017-q4-major/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export RFLY_SIM_HOST_IP="$hostIp"
export RFLY_SIM_FORWARD_PORT="$simForwardPort"
export RFLY_QGC_HOST_IP="$hostIp"
export RFLY_QGC_UDP_PORT="$QgcUdpPort"
export RFLY_DASHBOARD_HOST_IP="$hostIp"
export RFLY_DASHBOARD_PYMAV_UDP_PORT="$DashboardPymavUdpPort"
export RFLY_PX4_FIRMWARE_DIR="$PspPathLinux/Firmware"
echo "RFLY_SIM_HOST_IP=`$RFLY_SIM_HOST_IP"
echo "RFLY_SIM_FORWARD_PORT=`$RFLY_SIM_FORWARD_PORT"
echo "RFLY_QGC_HOST_IP=`$RFLY_QGC_HOST_IP"
echo "RFLY_QGC_UDP_PORT=`$RFLY_QGC_UDP_PORT"
echo "RFLY_DASHBOARD_HOST_IP=`$RFLY_DASHBOARD_HOST_IP"
echo "RFLY_DASHBOARD_PYMAV_UDP_PORT=`$RFLY_DASHBOARD_PYMAV_UDP_PORT"
make px4_sitl_default

"@

if ($NoDirectQgcLink) {
    $bashScript += @"
unset RFLY_QGC_HOST_IP

"@
}

if ($NoDashboardPymavLink) {
    $bashScript += @"
unset RFLY_DASHBOARD_HOST_IP

"@
}

$bashScript += @"
bash $runtimeQgcScriptLinux $VehicleNum $StartIndex $Frame
echo "PX4 SITL launched; keeping WSL session alive."
while true; do sleep 3600; done
"@

$bashFile = Join-Path $RuntimeDir "uav_rflysim_px4_start.sh"
$bashScript = $bashScript -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($bashFile, $bashScript, [System.Text.Encoding]::ASCII)
$bashFileLinux = Convert-ToWslPath $bashFile

$stdoutLog = Join-Path $RuntimeDir "rflysim_px4_stdout.log"
$stderrLog = Join-Path $RuntimeDir "rflysim_px4_stderr.log"

Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList @("-d", "RflySim-20.04", "-e", "bash", $bashFileLinux) `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden

Write-Host "[INFO] Started PX4 SITL in WSL"
Write-Host "[INFO] PSP_PATH=$env:PSP_PATH"
Write-Host "[INFO] PSP_PATH_LINUX=$env:PSP_PATH_LINUX"
Write-Host "[INFO] LowGPU reference launcher=$LowGpuBatPath"
Write-Host "[INFO] RflySim3D map=$Map"
Write-Host "[INFO] CopterSim UDPSIMMODE=$UdpSimMode"
Write-Host "[INFO] WSL networking mode=$(if ($isWsl1) { 'wsl1-shared' } elseif ($isMirroredNetworking) { 'mirrored' } else { 'nat' })"
Write-Host "[INFO] PX4 Windows target IP=$hostIp"
Write-Host "[INFO] PX4 Rfly MAVLink target=$hostIp`:$simForwardPort"
if (-not $usesSharedLoopback) {
    Write-Host "[INFO] Rfly WSL UDP relay=Windows 16539/17540 -> WSL $wslIp`:17540"
}
if (-not $NoDirectQgcLink) {
    Write-Host "[INFO] PX4 direct QGC UDP port=$QgcUdpPort"
}
if (-not $NoDashboardPymavLink) {
    Write-Host "[INFO] PX4 direct Dashboard pymav UDP port=$DashboardPymavUdpPort"
}
Write-Host "[INFO] PX4 stdout log: $stdoutLog"
Write-Host "[INFO] PX4 stderr log: $stderrLog"
Write-Host "[INFO] PX4 instance logs: $PspPath\Firmware\build\px4_sitl_default\instance_$StartIndex"
Write-Host "[INFO] Start Dashboard collection after QGC shows connected."
