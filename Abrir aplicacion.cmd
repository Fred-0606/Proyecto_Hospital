@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_app.ps1"
if errorlevel 1 (
  echo.
  echo No fue posible abrir la aplicacion. Revisa el mensaje anterior.
  pause
)
endlocal
