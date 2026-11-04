@echo off
REM ==============================================================================
REM Unreal Engine 5 Automated Multi-Modal Data Collection Script
REM Runs headless UnrealEditor-Cmd to capture RGB, Depth, Normal maps & Metadata
REM ==============================================================================

set UE_CMD="C:\Program Files\Epic Games\UE_5.4\Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
set PROJECT="C:\UnrealProjects\GenerativeDiffusionSim\GenerativeDiffusionSim.uproject"
set SCRIPT_PATH="%~dp0..\src\unreal\ue_python_capture.py"

echo ==================================================================
echo Starting Unreal Engine 5 Automated Multi-Modal Data Capture...
echo ==================================================================
echo Editor: %UE_CMD%
echo Project: %PROJECT%
echo Python Script: %SCRIPT_PATH%
echo ==================================================================

if not exist %UE_CMD% (
    echo [ERROR] UnrealEditor-Cmd.exe not found at %UE_CMD%.
    echo Please install Unreal Engine 5 or update UE_CMD path in this script.
    exit /b 1
)

%UE_CMD% %PROJECT% -run=pythonscript -script=%SCRIPT_PATH% -unattended -nopause -nosplash -stdout

echo Data collection job finished with exit code %ERRORLEVEL%.

