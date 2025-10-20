@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

:: Ensure Python is available
where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not on PATH.
    echo Install Python 3.10+ and rerun this script.
    goto :END
)

:: Create the virtual environment on first run
if not exist ".venv\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
    goto :END
    )
)

:: Activate the virtual environment
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    goto :END
)

:: Install or update dependencies
if exist requirements.txt (
    echo Installing required Python packages...
    python -m pip install --upgrade pip >nul
    if errorlevel 1 (
        echo Failed to upgrade pip.
    goto :END
    )
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
    goto :END
    )
)

set "PYTHONPATH=%SCRIPT_DIR%src;%PYTHONPATH%"

echo.
echo Next steps: Open the GUI (or run "start.bat --cli --colmap-bin C:\Path\to\colmap.exe").
echo Point the app at your COLMAP executable so the reconstruction stage can run.

:: Allow explicit CLI execution
if /I "%~1"=="--cli" (
    shift
    python -m videoto3d %*
    goto :END
)

:: Run the GUI by default (passes any extra args through)
python -m videoto3d.gui %*

:END
popd >nul
endlocal
