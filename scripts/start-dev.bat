@echo off
REM Windows cmd wrapper for scripts\start-dev.ps1
powershell -ExecutionPolicy Bypass -File "%~dp0start-dev.ps1"
