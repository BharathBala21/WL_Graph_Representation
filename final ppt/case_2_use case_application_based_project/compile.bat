@echo off
setlocal

cd /d "%~dp0"

:: Locate tectonic executable
set "TECTONIC="
if exist "%~dp0tectonic.exe" set "TECTONIC=%~dp0tectonic.exe"
if not defined TECTONIC if exist "%~dp0..\tectonic.exe" set "TECTONIC=%~dp0..\tectonic.exe"
if not defined TECTONIC if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\tectonic.exe" set "TECTONIC=%LOCALAPPDATA%\Microsoft\WindowsApps\tectonic.exe"
if not defined TECTONIC if exist "%USERPROFILE%\.gemini\antigravity-ide\bin\tectonic.exe" set "TECTONIC=%USERPROFILE%\.gemini\antigravity-ide\bin\tectonic.exe"
if not defined TECTONIC set "TECTONIC=tectonic"

echo Compiling 2_presentation.tex using %TECTONIC%...
"%TECTONIC%" 2_presentation.tex

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================================
    echo Successfully generated 2_presentation.pdf!
    echo ======================================================================
) else (
    echo.
    echo Compilation failed with error code %ERRORLEVEL%.
)

pause
