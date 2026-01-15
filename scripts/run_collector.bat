@echo off

chcp 65001 > nul

echo ========================================
echo Tourism Data Collector Service
echo ========================================
echo Started at: %date% %time%
echo.

cd /d "%~dp0.."

set PYTHONIOENCODING=utf-8

call .venv\Scripts\activate.bat

echo Running collector...
python scripts\run_collector_service.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Collection completed successfully
) else (
    echo.
    echo [FAILED] Collection failed with exit code %ERRORLEVEL%
)

echo.
echo Finished at: %date% %time%
echo ========================================
