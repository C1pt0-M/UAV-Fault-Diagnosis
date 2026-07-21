$ErrorActionPreference = "Continue"

Get-Process -Name "QGroundControl","CopterSim","CopterSimNoUI","RflySim3D" -ErrorAction SilentlyContinue |
    Stop-Process -Force

Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*rfly_wsl_udp_relay.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

wsl -d RflySim-20.04 -e bash -lc "pkill -9 -f '[p]x4' || true" | Out-Null

Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*uav_rflysim_px4_start.sh*" -or $_.CommandLine -like "*rflysim_px4_start.sh*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Write-Host "Stopped RflySim/QGC/CopterSim/PX4 SITL processes started by the wrapper where possible."
