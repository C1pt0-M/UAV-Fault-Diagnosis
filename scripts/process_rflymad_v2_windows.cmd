@echo off
setlocal
set TOOL_ROOT=D:\UAV-Fault-Diagnosis\vendor\Data_processing_tools_windows
set PYTHON_BIN=D:\Anaconda\python.exe
if not exist "%PYTHON_BIN%" set PYTHON_BIN=python
set ORIGINAL=D:\RflyMAD
set OUTPUT=D:\UAV-Data\processed_official_50hz
set LOG_ROOT=D:\UAV-Data\logs
set TARGET_PLATFORM=%~1
if "%TARGET_PLATFORM%"=="" set TARGET_PLATFORM=HIL
set FAULT_FILTER=%~2
set FLIGHT_FILTER=%~3
if "%FLIGHT_FILTER%"=="" set FLIGHT_FILTER=0
if not "%~4"=="" set OUTPUT=%~4
if not exist "%OUTPUT%" mkdir "%OUTPUT%"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"
set CASE_MAPPING_PATH=%LOG_ROOT%\case_mapping_%TARGET_PLATFORM%.csv
set BUILD_BATCH_ID=%TARGET_PLATFORM%_50hz
if /I "%TARGET_PLATFORM%"=="SIL" set SUB_DATASET=1
if /I "%TARGET_PLATFORM%"=="HIL" set SUB_DATASET=2
if /I "%TARGET_PLATFORM%"=="Real" set SUB_DATASET=3
if not defined SUB_DATASET (
    echo Usage: process_rflymad_v2_windows.cmd [SIL|HIL|Real] [fault_list] [flight_status] [output_dir]
    exit /b 1
)
if "%FAULT_FILTER%"=="" (
    if /I "%TARGET_PLATFORM%"=="Real" (
        set FAULT_FILTER=1 6 7 8 9 10 11
    ) else (
        set FAULT_FILTER=0
    )
)
echo platform=%TARGET_PLATFORM% fault=%FAULT_FILTER% flight_status=%FLIGHT_FILTER% freq=50Hz
echo output=%OUTPUT%
echo case_mapping=%CASE_MAPPING_PATH%
"%PYTHON_BIN%" "%TOOL_ROOT%\Rflytool_main.py" --original_path "%ORIGINAL%" --restore_path "%OUTPUT%" --sub_dataset %SUB_DATASET% --fault_type %FAULT_FILTER% --flight_status %FLIGHT_FILTER% --trans_freq 50 --case_mapping_path "%CASE_MAPPING_PATH%" --build_batch_id "%BUILD_BATCH_ID%"
