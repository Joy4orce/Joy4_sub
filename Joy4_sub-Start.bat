@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv-vatsg\Scripts\python.exe"
set "APP=%ROOT%Joy4_sub\Joy4_sub.py"

if not exist "%PYTHON%" (
  echo.
  echo [Joy4_sub] Virtual environment not found:
  echo %PYTHON%
  echo.
  pause
  exit /b 1
)

if not exist "%APP%" (
  echo.
  echo [Joy4_sub] App entry file not found:
  echo %APP%
  echo.
  pause
  exit /b 1
)

pushd "%ROOT%"
start "" "%PYTHON%" "%APP%"
popd

exit /b 0
