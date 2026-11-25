"""LosslessSync: automatic dub-to-release audio sync.

Everything it runs (ffmpeg, lls-dsp, the stems Python, models) lives in .lls/ at the repo
root, set up by `lls setup` (see the lls launcher); bin/ there goes first on PATH.
"""
import os
import platform
import sys

__version__ = "0.1.0"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def platform_key() -> str:
    """windows-x64, linux-x64 or linux-arm64 (one .lls/<key> per platform: a dual-boot PC
    can share the repo)."""
    m = platform.machine().lower()
    arch = "arm64" if m in ("aarch64", "arm64") else "x64"
    return f"{'windows' if os.name == 'nt' else sys.platform.replace('darwin', 'macos')}-{arch}"


LLS = os.path.join(ROOT, ".lls")
LOCAL = os.path.join(LLS, platform_key())
BIN = os.path.join(LOCAL, "bin")
MODELS = os.path.join(LLS, "models")


def venv_python(name: str) -> str:
    """The Python of .lls/<platform>/<name> (venv or venv-sep)."""
    sub, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return os.path.join(LOCAL, name, sub, exe)


if os.path.isdir(BIN) and BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = BIN + os.pathsep + os.environ.get("PATH", "")
