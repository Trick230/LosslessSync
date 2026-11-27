"""LosslessSync: automatic dub-to-release audio sync.

Everything it runs (ffmpeg, lls-dsp, the stems Python, models) lives in .lls/ at the repo
root, set up by `lls setup` (see the lls launcher); bin/ there goes first on PATH.
"""
import os
import platform
import sys

# The analysis runs its FFT/numpy work on a thread pool of all cores (features.pmap); BLAS
# calls inside those tasks would each start OpenBLAS's own spinning threads on the same
# cores (cores x cores threads). One BLAS thread per task: a real episode 56 -> 43 s, same
# output. Must be set before numpy is imported; an explicit setting is kept.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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


def dotenv() -> dict:
    """KEY=VALUE lines of the repo's .env (devcloud: dc secrets pull; the lines after
    `# dc: local` are this machine's own), empty values left out."""
    res = {}
    try:
        with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip().strip('"').strip("'"):
                        res[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return res


def dc_launcher() -> str | None:
    """devcloud's dc (https://github.com/Trick230/devcloud): DEVCLOUD (environment or .env),
    .devcloud/ (devcloud's Actions runner) or ~/devcloud; it runs the cloud GPU jobs and the
    Drive."""
    name = "dc.cmd" if os.name == "nt" else "dc"
    for d in (os.environ.get("DEVCLOUD") or dotenv().get("DEVCLOUD", ""),
              os.path.join(ROOT, ".devcloud"), os.path.expanduser(os.path.join("~", "devcloud"))):
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.abspath(os.path.join(d, name))
    return None


if os.path.isdir(BIN) and BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = BIN + os.pathsep + os.environ.get("PATH", "")
