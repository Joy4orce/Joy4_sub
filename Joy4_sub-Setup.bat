@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv-vatsg"
set "REQ=%ROOT%requirements-venv.txt"
set "APP=%ROOT%Joy4_sub-Start.bat"
set "PYTHON_CMD="
set "PYTHON_EXE="
set "PYTHON_VERSION_OK="

call :resolve_python
if not defined PYTHON_CMD (
  echo.
  echo [Joy4_sub] Python 3.10 was not found.
  echo.
  choice /C YN /M "Install Python 3.10 automatically with winget"
  if errorlevel 2 goto :python_missing

  winget --version >nul 2>&1
  if errorlevel 1 goto :winget_missing

  echo.
  echo [Joy4_sub] Installing Python 3.10 with winget...
  winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :fail

  call :resolve_python
  if not defined PYTHON_CMD goto :python_missing
)

if not exist "%REQ%" (
  echo.
  echo [Joy4_sub] requirements file not found:
  echo %REQ%
  echo.
  pause
  exit /b 1
)

echo.
echo [Joy4_sub] Creating virtual environment...
%PYTHON_CMD% -m venv "%VENV%"
if errorlevel 1 goto :fail

set "VPYTHON=%VENV%\Scripts\python.exe"
if not exist "%VPYTHON%" (
  echo.
  echo [Joy4_sub] Virtual environment Python not found:
  echo %VPYTHON%
  echo.
  pause
  exit /b 1
)

echo.
echo [Joy4_sub] Upgrading pip...
"%VPYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo [Joy4_sub] Installing packages...
"%VPYTHON%" -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r "%REQ%"
if errorlevel 1 goto :fail

echo.
echo [Joy4_sub] Setup complete.
echo Use Joy4_sub-Start.bat to launch the program.
echo.
choice /C YN /M "Start Joy4_sub now"
if errorlevel 2 goto :done
call "%APP%"
goto :done

:fail
echo.
echo [Joy4_sub] Setup failed.
echo Check the messages above, then run this setup again.
echo.
pause
exit /b 1

:winget_missing
echo.
echo [Joy4_sub] winget is not available on this PC.
echo Install Python 3.10 manually, then run Joy4_sub-Setup.bat again.
echo Download: https://www.python.org/downloads/release/python-31011/
echo.
pause
exit /b 1

:python_missing
echo.
echo [Joy4_sub] Python 3.10 installation could not be confirmed.
echo If Python was just installed, close this window and run Joy4_sub-Setup.bat again.
echo Download: https://www.python.org/downloads/release/python-31011/
echo.
pause
exit /b 1

:resolve_python
set "PYTHON_CMD="
set "PYTHON_EXE="
set "PYTHON_VERSION_OK="

py -3.10 -c "import sys; print(sys.executable)" > "%TEMP%\joy4sub_python_path.txt" 2>nul
if %errorlevel%==0 (
  set /p PYTHON_EXE=<"%TEMP%\joy4sub_python_path.txt"
  if defined PYTHON_EXE (
    set "PYTHON_CMD=py -3.10"
    goto :resolve_done
  )
)

python -c "import sys; assert sys.version_info[:2] == (3, 10); print(sys.executable)" > "%TEMP%\joy4sub_python_path.txt" 2>nul
if %errorlevel%==0 (
  set /p PYTHON_EXE=<"%TEMP%\joy4sub_python_path.txt"
  if defined PYTHON_EXE (
    set "PYTHON_CMD=python"
    goto :resolve_done
  )
)

:resolve_done
del "%TEMP%\joy4sub_python_path.txt" >nul 2>&1
exit /b 0

:done
exit /b 0
