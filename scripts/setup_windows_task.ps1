# Tourism Data Collector - Windows Task Scheduler Setup
# This script creates a scheduled task to run the collector every hour

param(
    [Parameter(Mandatory=$false)]
    [string]$TaskName = "TourismDataCollector",
    
    [Parameter(Mandatory=$false)]
    [int]$IntervalHours = 1
)

# Get project root directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Path to batch file
$BatchFile = Join-Path $ProjectRoot "scripts\run_collector.bat"

# Check if batch file exists
if (-not (Test-Path $BatchFile)) {
    Write-Error "Batch file not found: $BatchFile"
    exit 1
}

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Tourism Data Collector - Task Scheduler Setup" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Task Name: $TaskName" -ForegroundColor Yellow
Write-Host "Interval: Every $IntervalHours hour(s)" -ForegroundColor Yellow
Write-Host "Script: $BatchFile" -ForegroundColor Yellow
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[WARNING] This script requires Administrator privileges" -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "[INFO] Task '$TaskName' already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to update it? (Y/N)"
    
    if ($response -ne 'Y' -and $response -ne 'y') {
        Write-Host "Setup cancelled" -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "[INFO] Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create task action
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BatchFile`"" -WorkingDirectory $ProjectRoot

# Create trigger - repeat every X hours
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Hours $IntervalHours)

# Create task settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Create principal (run as current user)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Automated tourism data collection service - collects comments from social platforms every $IntervalHours hour(s)" | Out-Null
    
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "[SUCCESS] Task created successfully!" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Cyan
    Write-Host "- Name: $TaskName" -ForegroundColor White
    Write-Host "- First Run: In 5 minutes" -ForegroundColor White
    Write-Host "- Repeat: Every $IntervalHours hour(s)" -ForegroundColor White
    Write-Host "- Timeout: 2 hours per run" -ForegroundColor White
    Write-Host ""
    Write-Host "To manage the task:" -ForegroundColor Yellow
    Write-Host "1. Open Task Scheduler (taskschd.msc)" -ForegroundColor White
    Write-Host "2. Navigate to: Task Scheduler Library" -ForegroundColor White
    Write-Host "3. Find: $TaskName" -ForegroundColor White
    Write-Host ""
    Write-Host "To run manually:" -ForegroundColor Yellow
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor White
    Write-Host ""
    Write-Host "To disable:" -ForegroundColor Yellow
    Write-Host "  Disable-ScheduledTask -TaskName '$TaskName'" -ForegroundColor White
    Write-Host ""
    Write-Host "To remove:" -ForegroundColor Yellow
    Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName'" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "[ERROR] Failed to create scheduled task" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Ask if user wants to run immediately
Write-Host "Do you want to test the collector now? (Y/N): " -ForegroundColor Yellow -NoNewline
$runNow = Read-Host

if ($runNow -eq 'Y' -or $runNow -eq 'y') {
    Write-Host ""
    Write-Host "[INFO] Running collector immediately..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "[INFO] Check Task Scheduler for execution status" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
