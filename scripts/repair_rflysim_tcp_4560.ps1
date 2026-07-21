param(
    [int]$Port = 4560
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-ExcludedTcpRanges {
    param(
        [Parameter(Mandatory=$true)][ValidateSet("ipv4", "ipv6")][string]$Family
    )

    $output = & netsh interface $Family show excludedportrange protocol=tcp
    foreach ($line in $output) {
        if ($line -match '^\s*(\d+)\s+(\d+)\s*\*?\s*$') {
            [pscustomobject]@{
                Family = $Family
                Start = [int]$Matches[1]
                End = [int]$Matches[2]
            }
        }
    }
}

function Test-TcpBind {
    param([Parameter(Mandatory=$true)][int]$ListenPort)

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Any,
        $ListenPort
    )
    try {
        $listener.Start()
        return $true
    } catch {
        Write-Host "[ERROR] TCP $ListenPort bind failed: $($_.Exception.InnerException.Message)" -ForegroundColor Red
        return $false
    } finally {
        try { $listener.Stop() } catch {}
    }
}

if (-not (Test-IsAdministrator)) {
    throw "Administrator privileges are required. Open PowerShell 7 as Administrator and run this script again."
}

$stopScript = Join-Path $PSScriptRoot "stop_rflysim_sitl.ps1"
if (Test-Path -LiteralPath $stopScript) {
    & $stopScript
}

& wsl --shutdown

$ranges = @(
    Get-ExcludedTcpRanges -Family ipv4
    Get-ExcludedTcpRanges -Family ipv6
) | Where-Object { $_.Start -le $Port -and $_.End -ge $Port }

if ($ranges.Count -eq 0 -and (Test-TcpBind -ListenPort $Port)) {
    Write-Host "[OK] TCP $Port is already available."
    exit 0
}

Write-Host "[INFO] Restoring the Windows TCP dynamic port range to 49152-65535."
& netsh interface ipv4 set dynamicport tcp start=49152 num=16384 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Failed to set the IPv4 dynamic TCP port range." }
& netsh interface ipv6 set dynamicport tcp start=49152 num=16384 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Failed to set the IPv6 dynamic TCP port range." }

$serviceState = @{}
foreach ($name in @("WinNAT", "hns")) {
    $service = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($null -ne $service) {
        $serviceState[$name] = $service.Status
        if ($service.Status -ne "Stopped") {
            Write-Host "[INFO] Stopping $name"
            Stop-Service -Name $name -Force
        }
    }
}

try {
    foreach ($range in $ranges) {
        $count = $range.End - $range.Start + 1
        Write-Host "[INFO] Removing $($range.Family) TCP exclusion $($range.Start)-$($range.End)"
        & netsh interface $range.Family delete excludedportrange `
            protocol=tcp startport=$($range.Start) numberofports=$count | Out-Host
    }
} finally {
    foreach ($name in @("hns", "WinNAT")) {
        if ($serviceState[$name] -eq "Running") {
            Write-Host "[INFO] Starting $name"
            Start-Service -Name $name
        }
    }
}

Start-Sleep -Seconds 2

$remaining = @(
    Get-ExcludedTcpRanges -Family ipv4
    Get-ExcludedTcpRanges -Family ipv6
) | Where-Object { $_.Start -le $Port -and $_.End -ge $Port }

if ($remaining.Count -gt 0 -or -not (Test-TcpBind -ListenPort $Port)) {
    throw "TCP $Port is still unavailable. Restart Windows once, then rerun this script before starting RflySim."
}

Write-Host "[OK] TCP $Port is available. RflySim/CopterSim can now use the PX4 simulator channel." -ForegroundColor Green
