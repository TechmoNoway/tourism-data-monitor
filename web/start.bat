@echo off
echo ========================================
echo Tourism Data Monitor Frontend
echo ========================================
echo.

REM Check if node_modules exists
if not exist "node_modules\" (
    echo [1/2] Installing dependencies...
    echo.
    call yarn install
    echo.
) else (
    echo [INFO] Dependencies already installed
    echo.
)

echo [2/2] Starting development server...
echo.
echo Backend should be running on http://localhost:8080
echo Frontend will start on http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================

call yarn dev
