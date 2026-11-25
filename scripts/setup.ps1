# LosslessSync setup for Windows (the main PC). Idempotent; same steps as setup.sh.
#   scripts\setup.ps1 [-Check] [-Env pc|cloud] [-NoStems] [-NoRust]
# Everything installed lives in %USERPROFILE%\.lls (venv, venv-sep, cargo-target), outside the repo.
param([switch]$Check, [Alias("Env")][string]$Mode = $env:LLS_ENV, [switch]$NoStems, [switch]$NoRust)
# native tools write warnings to stderr: Windows PowerShell 5.1 would turn them into errors
$ErrorActionPreference = "Continue"
function Must([scriptblock]$b) { & $b; if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $b" } }

$Repo = Split-Path -Parent $PSScriptRoot
$Lls = Join-Path $env:USERPROFILE ".lls"
$Venv = Join-Path $Lls "venv"
$Sep = Join-Path $Lls "venv-sep"
$VPy = Join-Path $Venv "Scripts\python.exe"
$SPy = Join-Path $Sep "Scripts\python.exe"
$env:CARGO_TARGET_DIR = Join-Path $Lls "cargo-target"
$env:Path = (Join-Path $env:USERPROFILE ".cargo\bin") + ";" + $env:Path
if (-not $Mode) { $Mode = "pc" }

function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Ok($block) { try { & $block *> $null; $LASTEXITCODE -eq 0 } catch { $false } }

function Find-Python([string]$ver) {
    # py launcher, then the python.org per-user installs, then python on PATH (not the Store alias)
    if (Have py) {
        $p = & py "-$ver" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    }
    $base = Join-Path $env:LOCALAPPDATA "Programs\Python"
    $pat = if ($ver -eq "3") { "Python3*" } else { "Python" + $ver.Replace(".", "") }
    $c = Get-ChildItem -Path $base -Filter $pat -Directory -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.Name -replace "\D", "") } -Descending |
        ForEach-Object { Join-Path $_.FullName "python.exe" } | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($c) { return $c }
    $w = Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike "*WindowsApps*" } | Select-Object -First 1
    if ($w -and $ver -eq "3") { return $w.Source }
    return $null
}

function Find-Mingw {
    # a WinLibs GCC with LLVM (gcc + libclang) for the signalsmith feature
    foreach ($d in @($env:LLS_MINGW, (Join-Path $env:USERPROFILE "tools\mingw64\bin"))) {
        if ($d -and (Test-Path (Join-Path $d "gcc.exe")) -and (Test-Path (Join-Path $d "libclang.dll"))) { return $d }
    }
    $g = Get-Command gcc -ErrorAction SilentlyContinue
    if ($g) { $d = Split-Path $g.Source; if (Test-Path (Join-Path $d "libclang.dll")) { return $d } }
    return $null
}

function Persist($name, $value) {
    [Environment]::SetEnvironmentVariable($name, $value, "User")
    Set-Item -Path "Env:$name" -Value $value
}

function Show-Check {
    $ext = if ($env:LOSSLESSSYNC_EXT_PYTHON) { $env:LOSSLESSSYNC_EXT_PYTHON } else { [Environment]::GetEnvironmentVariable("LOSSLESSSYNC_EXT_PYTHON", "User") }
    "environment : $Mode (Windows $env:PROCESSOR_ARCHITECTURE)"
    "repo        : $Repo"
    $v = if (Test-Path $VPy) { & $VPy -c "import losslesssync,sys;print(sys.version.split()[0], 'losslesssync ok')" 2>$null } else { "missing" }
    "venv        : $v"
    $rb = if (Have ffmpeg) { [bool]((ffmpeg -hide_banner -filters 2>$null) -match "rubberband") } else { $false }
    "ffmpeg      : $(Have ffmpeg), rubberband: $rb"
    $cv = if (Have cargo) { cargo --version } else { "no" }
    "cargo       : $cv"
    $d = if (Test-Path $VPy) { Push-Location $Repo; & $VPy -c "from losslesssync.stretch import native;print(native.dsp() or 'not built')"; Pop-Location } else { "?" }
    "lls-dsp     : $d"
    $g = if (Have nvidia-smi) { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader } else { "none" }
    "gpu         : $g"
    $t = if (Test-Path $SPy) { & $SPy -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>$null } else { "no venv-sep" }
    "stems python: $ext ($t)"
    $kt = Test-Path (Join-Path $Venv "Scripts\kaggle.exe"); $mt = Test-Path (Join-Path $Venv "Scripts\modal.exe")
    "kaggle/modal: $kt/$mt, rclone: $(Have rclone)"
    "media       : LLS_MEDIA_DIR=$env:LLS_MEDIA_DIR LLS_OUT_DIR=$env:LLS_OUT_DIR"
}
if ($Check) { Show-Check; exit 0 }

New-Item -ItemType Directory -Force $Lls | Out-Null

"== ffmpeg"
if (-not (Have ffmpeg)) { "ffmpeg missing: winget install Gyan.FFmpeg (then open a new shell)" }
elseif (-not ((ffmpeg -hide_banner -filters 2>$null) -match "rubberband")) { "WARNING: this ffmpeg has no rubberband filter: winget install Gyan.FFmpeg" }

"== venv $Venv"
if (-not (Test-Path $VPy)) {
    $py = Find-Python "3"
    if (-not $py) { throw "no Python 3 found: install it from python.org (winget install Python.Python.3.13)" }
    Must { & $py -m venv $Venv }
}
Must { & $VPy -m pip install -q --upgrade pip }
Must { & $VPy -m pip install -q -e $Repo }
# cloud GPU CLIs (gpu/run_gpu.py); in the cloud the venv is also the torch-free stems helper
if ($Mode -eq "cloud") { Must { & $VPy -m pip install -q kaggle modal } }
elseif (-not (Ok { & $VPy -m pip install -q kaggle modal })) { "WARNING: kaggle/modal CLIs not installed (no cloud fallback)" }
if ($Mode -eq "cloud") {
    Must { & $VPy -m pip install -q python-stretch soundfile soxr }
    Persist "LLS_ENV" "cloud"
    Persist "LOSSLESSSYNC_EXT_PYTHON" $VPy
}
$up = [Environment]::GetEnvironmentVariable("Path", "User")
$vs = Join-Path $Venv "Scripts"
if (-not (($up -split ";") -contains $vs)) { [Environment]::SetEnvironmentVariable("Path", "$vs;$up", "User") }

if (-not $NoRust) {
    "== lls-dsp (Rust)"
    if (-not (Have cargo)) {
        "cargo missing: winget install Rustlang.Rustup, then: rustup default stable-x86_64-pc-windows-gnu"
    } else {
        Persist "CARGO_TARGET_DIR" $env:CARGO_TARGET_DIR
        $mg = Find-Mingw
        Push-Location (Join-Path $Repo "dsp")
        try {
            if ($mg) {
                $env:Path = "$mg;$env:Path"; $env:LIBCLANG_PATH = $mg
                Must { cargo build -q --release --features signalsmith }
            } else {
                "no WinLibs GCC with libclang (set LLS_MINGW to its bin): building without signalsmith"
                Must { cargo build -q --release }
            }
        } finally { Pop-Location }
    }
}

if ($Mode -eq "pc" -and -not $NoStems) {
    "== stems venv $Sep (torch CUDA)"
    if (-not (Test-Path $SPy)) {
        $py = Find-Python "3.11"
        if (-not $py) { $py = Find-Python "3.12" }
        if (-not $py) { throw "no Python 3.11/3.12 found for the stems venv (winget install Python.Python.3.11)" }
        Must { & $py -m venv $Sep }
    }
    if (-not (Ok { & $SPy -c "import torch, audio_separator, python_stretch, soundfile, triton" })) {
        Must { & $SPy -m pip install -q --upgrade pip }
        if (-not (Ok { & $SPy -c "import torch" })) {
            Must { & $SPy -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126 }
        }
        Must { & $SPy -m pip install -q "audio-separator[gpu]" python-stretch soundfile "triton-windows<3.10" }
    }
    Persist "LOSSLESSSYNC_EXT_PYTHON" $SPy
}

"== done (open a new shell for the environment variables)"
Show-Check
