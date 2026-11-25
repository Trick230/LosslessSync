@echo off
rem LosslessSync launcher (Windows): fetches uv into .lls\windows-x64\bin, then runs
rem scripts\lls.py on a uv-managed Python. Nothing is installed outside this folder.
rem `lls help` for the commands.
setlocal
rem a PowerShell 7 parent leaks its module path into Windows PowerShell 5.1
set "PSModulePath="
set "L=%~dp0.lls\windows-x64"
if not exist "%L%\bin\uv.exe" (
  echo lls: downloading uv 0.13.10 1>&2
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';" ^
    "New-Item -ItemType Directory -Force '%L%\bin','%L%\cache' | Out-Null;" ^
    "$f='%L%\cache\uv-x86_64-pc-windows-msvc.zip';" ^
    "Invoke-WebRequest -UseBasicParsing -OutFile $f 'https://github.com/astral-sh/uv/releases/download/0.13.10/uv-x86_64-pc-windows-msvc.zip';" ^
    "if ((Get-FileHash -Algorithm SHA256 $f).Hash -ne '297b3881fe1adfb922becc2c1613202abb41dc83ea5ff43d472aa087ca1f8683') { Remove-Item $f; throw 'uv sha256 mismatch' };" ^
    "Expand-Archive -Force $f '%L%\bin'" || exit /b 1
)
set "UV_CACHE_DIR=%L%\cache\uv"
set "UV_PYTHON_INSTALL_DIR=%L%\python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_PYTHON_INSTALL_BIN=0"
"%L%\bin\uv.exe" run -q --no-project --python 3.14 "%~dp0scripts\lls.py" %*
exit /b %ERRORLEVEL%
