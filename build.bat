@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Joy4_sub portable build script
REM  Bundles source, assets, ffmpeg, and the existing venv into
REM  dist\Joy4_sub-Portable so it can be copied to another PC and
REM  launched with Joy4_sub-Start.bat.
REM ============================================================

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv-vatsg"
set "VPYTHON=%VENV%\Scripts\python.exe"
set "DIST=%ROOT%dist"
set "OUT=%DIST%\Joy4_sub-Portable"
set "ZIP_PATH=%DIST%\Joy4_sub-Portable.zip"

echo.
echo === Joy4_sub portable build ===
echo Source root : %ROOT%
echo Output dir  : %OUT%
echo.
echo Note: the bundled venv (.venv-vatsg) includes torch + CUDA libs and
echo       can easily reach 4-6 GB. Make sure you have free disk space.
echo.

REM ---------- Sanity checks ----------
if not exist "%VPYTHON%" (
  echo [BUILD] Virtual environment not found at:
  echo   %VENV%
  echo [BUILD] Run Joy4_sub-Setup.bat first to create the venv.
  goto :fail
)

if not exist "%ROOT%Joy4_sub\Joy4_sub.py" (
  echo [BUILD] Joy4_sub\Joy4_sub.py not found.
  goto :fail
)

REM ---------- Confirm before wiping ----------
if exist "%OUT%" (
  echo [BUILD] Output directory already exists. It will be deleted.
  choice /C YN /M "Continue"
  if errorlevel 2 goto :cancelled
)

REM ---------- Clean previous build ----------
echo.
echo [BUILD] Cleaning previous build...
if exist "%OUT%" rmdir /s /q "%OUT%"
if exist "%ZIP_PATH%" del /q "%ZIP_PATH%"
mkdir "%OUT%" 2>nul
mkdir "%DIST%" 2>nul

REM ---------- Strip __pycache__ from source/venv before copying ----------
echo [BUILD] Removing __pycache__ from source...
for /d /r "%ROOT%Joy4_sub" %%D in (__pycache__) do if exist "%%D" rmdir /s /q "%%D"
echo [BUILD] Removing __pycache__ from venv (this can take a while)...
for /d /r "%VENV%" %%D in (__pycache__) do if exist "%%D" rmdir /s /q "%%D"

REM ---------- Copy source files ----------
echo.
echo [BUILD] Copying source files...
xcopy /E /I /Y /Q "%ROOT%Joy4_sub" "%OUT%\Joy4_sub" >nul
if errorlevel 1 goto :fail

echo [BUILD] Copying Asset directory...
xcopy /E /I /Y /Q "%ROOT%Asset" "%OUT%\Asset" >nul
if errorlevel 1 goto :fail

REM ---------- Copy individual support files ----------
echo [BUILD] Copying ffmpeg, hooks, configs, docs...
copy /Y "%ROOT%ffmpeg.exe"            "%OUT%\" >nul
copy /Y "%ROOT%hook-tkinterdnd2.py"   "%OUT%\" >nul
copy /Y "%ROOT%srclangcode.txt"       "%OUT%\" >nul
copy /Y "%ROOT%targetlangcode.txt"    "%OUT%\" >nul
copy /Y "%ROOT%requirements.txt"      "%OUT%\" >nul
copy /Y "%ROOT%requirements-venv.txt" "%OUT%\" >nul
copy /Y "%ROOT%LICENSE.txt"           "%OUT%\" >nul
copy /Y "%ROOT%licenses.txt"          "%OUT%\" >nul
copy /Y "%ROOT%README.md"             "%OUT%\" >nul
copy /Y "%ROOT%Joy4_sub-Setup.bat"    "%OUT%\" >nul
copy /Y "%ROOT%Joy4_sub-Start.bat"    "%OUT%\" >nul

REM ---------- Copy venv ----------
echo.
echo [BUILD] Copying virtual environment (this is the slow part)...
xcopy /E /I /Y /Q "%VENV%" "%OUT%\.venv-vatsg" >nul
if errorlevel 1 goto :fail

REM ---------- Drop a portable launcher inside the dist ----------
REM On the target PC the bundled venv only works if Python 3.10 is also
REM installed (the venv's python.exe relies on the base interpreter's DLLs).
REM We probe the bundled python.exe first; on any failure, fall back to
REM Joy4_sub-Setup.bat which auto-installs Python 3.10 via winget and rebuilds
REM the venv in place.
echo [BUILD] Writing portable Joy4_sub-Start.bat...
> "%OUT%\Joy4_sub-Start.bat" (
  echo @echo off
  echo setlocal
  echo set "ROOT=%%~dp0"
  echo set "PYTHON=%%ROOT%%.venv-vatsg\Scripts\python.exe"
  echo set "APP=%%ROOT%%Joy4_sub\Joy4_sub.py"
  echo.
  echo if not exist "%%PYTHON%%" goto :do_setup
  echo "%%PYTHON%%" -c "import sys" ^>nul 2^>^&1
  echo if errorlevel 1 goto :do_setup
  echo.
  echo if not exist "%%APP%%" ^(
  echo   echo.
  echo   echo [Joy4_sub] App entry not found: %%APP%%
  echo   pause
  echo   exit /b 1
  echo ^)
  echo pushd "%%ROOT%%"
  echo start "" "%%PYTHON%%" "%%APP%%"
  echo popd
  echo exit /b 0
  echo.
  echo :do_setup
  echo echo.
  echo echo [Joy4_sub] Bundled venv is unusable on this machine.
  echo echo [Joy4_sub] Running Joy4_sub-Setup.bat to install Python 3.10 and rebuild...
  echo echo.
  echo call "%%ROOT%%Joy4_sub-Setup.bat"
  echo exit /b 0
)

REM ---------- Optional: build a ZIP archive ----------
echo.
choice /C YN /M "Create a ZIP archive of the distribution"
if errorlevel 2 goto :no_zip

echo [BUILD] Compressing to ZIP (PowerShell)...
powershell -NoProfile -Command "Compress-Archive -Path '%OUT%' -DestinationPath '%ZIP_PATH%' -Force"
if errorlevel 1 (
  echo [BUILD] ZIP creation failed but the directory build succeeded.
) else (
  echo [BUILD] ZIP created: %ZIP_PATH%
)

:no_zip
echo.
echo === Build complete ===
echo Portable directory: %OUT%
if exist "%ZIP_PATH%" echo ZIP archive       : %ZIP_PATH%
echo.
echo Copy the directory (or unzip the archive) to the target PC and
echo run Joy4_sub-Start.bat. If the bundled venv does not work on the
echo target machine, Joy4_sub-Setup.bat will auto-install Python 3.10
echo and rebuild the venv.
echo.
pause
exit /b 0

:cancelled
echo [BUILD] Cancelled.
exit /b 1

:fail
echo.
echo [BUILD] Build failed. See messages above.
echo.
pause
exit /b 1
