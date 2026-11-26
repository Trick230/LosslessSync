"""The LosslessSync launcher (started by ./lls or lls.cmd, standard library only).

Everything the project runs is downloaded into .lls/<platform>/ at the repo root (nothing is
installed on the system, nothing is added to PATH or the environment for good):
python/ (uv's), venv/ (uv.lock), bin/ (uv, ffmpeg, ffprobe, lls-dsp),
venv-sep/ (the stems Python, `setup --stems`), rust/ (`setup --dev`); .lls/models holds the
separation models. A moved or copied folder works again after `lls setup`.

    lls setup [--stems] [--dev] [--cloud | --pc]   install or repair (repeatable)
    lls check                                      what is there
    lls test [all | CASE,CASE...]                  synthetic cases (default pal,tempo,drop_pal)
    lls py SCRIPT [ARGS]                           a script with the project's Python
    lls cargo ARGS                                 cargo in dsp/ (after setup --dev)
    lls gui                                        the graphical interface
    lls sync|batch|probe ...                       the losslesssync CLI

--stems: torch + audio-separator for --keep-engine stems (3-5 GB; NVIDIA GPU). --cloud: no
torch, separation on a cloud GPU through devcloud (`dc gpu gpu/separate.json`). --dev: Rust
in .lls to rebuild lls-dsp (otherwise it is the binary CI built for these exact dsp/ sources).
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from losslesssync import BIN, LOCAL, MODELS, platform_key, venv_python  # noqa: E402

PLAT = platform_key()
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
PY = "3.14"                     # the project's Python (uv-managed)
SEP_PY = "3.11"                 # the stems Python (torch, audio-separator)
REPO = "Trick230/LosslessSync"  # GitHub releases with the prebuilt lls-dsp
MODEL = "vocals_mel_band_roformer.ckpt"
GH = "https://github.com"

FF = f"{GH}/BtbN/FFmpeg-Builds/releases/download"
FFMPEG = {   # static GPL builds (with librubberband); the dated build is pinned, the rolling
             # 8.1 one is the fallback when it is gone (checked for rubberband instead)
    "windows-x64": (f"{FF}/autobuild-2026-11-25-12-41/ffmpeg-n8.1.3-41-ga8029b6528-win64-gpl-8.1.zip",
                    "a1ef44f882e103d297156be744bf138eb5de1054e864218d43a65d1801570917",
                    f"{FF}/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip"),
    "linux-x64": (f"{FF}/autobuild-2026-11-25-12-41/ffmpeg-n8.1.3-41-ga8029b6528-linux64-gpl-8.1.tar.xz",
                  "52e5d6c80ef213861dcdbdf5f742c551469ee7de91b3a846256660c9d0943997",
                  f"{FF}/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"),
    "linux-arm64": (f"{FF}/autobuild-2026-11-25-12-41/ffmpeg-n8.1.3-41-ga8029b6528-linuxarm64-gpl-8.1.tar.xz",
                    "8d99132e74494ac29423ee850afdd9b3de2de92428926a6e282bc1b3a4abb8bc",
                    f"{FF}/latest/ffmpeg-n8.1-latest-linuxarm64-gpl-8.1.tar.xz"),
    "windows-arm64": (f"{FF}/autobuild-2026-11-25-12-41/ffmpeg-n8.1.3-41-ga8029b6528-winarm64-gpl-8.1.zip",
                      "878b5ab7c7bdf33c98ceed9bb526eef60588c5de8eff224edc6a40306cf96c4f",
                      f"{FF}/latest/ffmpeg-n8.1-latest-winarm64-gpl-8.1.zip"),
}
# macOS is not supported: no static ffmpeg with librubberband for Apple Silicon exists
# GCC + libclang for the signalsmith feature of lls-dsp on Windows (setup --dev; LLS_MINGW
# overrides it)
WINLIBS = ("https://github.com/brechtsanders/winlibs_mingw/releases/download/16.2.0posix-23.1.3-14.0.0-ucrt-r3/"
           "winlibs-x86_64-posix-seh-gcc-16.2.0-llvm-23.1.3-mingw-w64ucrt-14.0.0-r3.zip",
           "1f04e482797dc5f71fe2875cf2b743bfb9f5fd1de0955fdd3118438c0432ba5f")
RUST_HOST = {"windows-x64": "x86_64-pc-windows-gnu", "windows-arm64": "aarch64-pc-windows-msvc",
             "linux-x64": "x86_64-unknown-linux-gnu", "linux-arm64": "aarch64-unknown-linux-gnu"}
STEMS_TORCH = ("torch==2.14.1", "https://download.pytorch.org/whl/cu126")
STEMS = ["audio-separator[gpu]==0.48.0", "python-stretch==0.3.1", "soundfile==0.14.0",
         "soxr==1.1.0"] + (["triton-windows==3.9.0.post30"] if WIN else [])
STEMS_CLOUD = ["numpy", "python-stretch==0.3.1", "soundfile==0.14.0", "soxr==1.1.0"]


class Fail(Exception):
    pass


def say(msg: str) -> None:
    print(f"lls: {msg}", flush=True)


def uv_env() -> dict:
    env = dict(os.environ)
    env.update(UV_CACHE_DIR=os.path.join(LOCAL, "cache", "uv"),
               UV_PYTHON_INSTALL_DIR=os.path.join(LOCAL, "python"),
               UV_PYTHON_PREFERENCE="only-managed", UV_PYTHON_INSTALL_BIN="0",
               UV_PROJECT_ENVIRONMENT=os.path.join(LOCAL, "venv"),
               UV_TOOL_DIR=os.path.join(LOCAL, "tools"), UV_TOOL_BIN_DIR=BIN)
    env.pop("VIRTUAL_ENV", None)
    return env


def uv(*args: str, check: bool = True) -> int:
    rc = subprocess.call([os.path.join(BIN, "uv" + EXE), *args], env=uv_env())
    if check and rc:
        raise Fail(f"uv {' '.join(args[:3])} ... failed ({rc})")
    return rc


def run_env() -> dict:
    """The environment for what lls starts: .lls first, the repo's code first."""
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if os.path.isfile(venv_python("venv-sep")):
        env["LOSSLESSSYNC_EXT_PYTHON"] = venv_python("venv-sep")
    if mode() == "cloud":
        env["LLS_ENV"] = "cloud"
    env.pop("VIRTUAL_ENV", None)
    return env


def mode() -> str:
    try:
        with open(os.path.join(LOCAL, "mode")) as f:
            return f.read().strip()
    except OSError:   # not set up yet: a Claude Code web sandbox is the cloud
        return "cloud" if os.environ.get("CLAUDE_CODE_REMOTE") else "pc"


# ---------------------------------------------------------------- downloads

def fetch(url: str, sha256: str | None = None) -> bytes:
    """Downloads (cached in .lls/<platform>/cache/downloads) and checks the sha256."""
    cache = os.path.join(LOCAL, "cache", "downloads", os.path.basename(url))
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            data = f.read()
        if not sha256 or hashlib.sha256(data).hexdigest() == sha256:
            return data
    say(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "losslesssync-setup"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total, buf, step = int(r.headers.get("Content-Length") or 0), io.BytesIO(), 0
        while chunk := r.read(1 << 20):
            buf.write(chunk)
            if total and buf.tell() * 10 // total > step:
                step = buf.tell() * 10 // total
                print(f"  {step * 10}%", end="\r", flush=True)
    data = buf.getvalue()
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise Fail(f"{url}: sha256 mismatch")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        f.write(data)
    return data


def extract(data: bytes, url: str, names: list[str]) -> None:
    """Writes the archive members with these base names into bin/."""
    os.makedirs(BIN, exist_ok=True)
    found = set()
    if url.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data))
        members = [(m.filename, lambda m=m: z.read(m)) for m in z.infolist() if not m.is_dir()]
    else:
        t = tarfile.open(fileobj=io.BytesIO(data))
        members = [(m.name, lambda m=m: t.extractfile(m).read()) for m in t.getmembers() if m.isfile()]
    for name, read in members:
        base = name.rsplit("/", 1)[-1]
        if base in names and base not in found:
            dst = os.path.join(BIN, base)
            tmp = dst + ".new"
            with open(tmp, "wb") as f:
                f.write(read())
            os.chmod(tmp, 0o755)
            os.replace(tmp, dst)
            found.add(base)
    if missing := set(names) - found:
        raise Fail(f"{os.path.basename(url)}: no {', '.join(sorted(missing))} inside")


def has_rubberband() -> bool:
    ff = os.path.join(BIN, "ffmpeg" + EXE)
    if not os.path.isfile(ff):
        return False
    r = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True)
    return "rubberband" in r.stdout


def setup_ffmpeg() -> None:
    if has_rubberband():
        return
    url, sha, fallback = FFMPEG[PLAT]
    names = ["ffmpeg" + EXE, "ffprobe" + EXE]
    try:
        extract(fetch(url, sha), url, names)
    except urllib.error.HTTPError as e:      # the dated build was deleted upstream
        say(f"{url}: {e.code}; taking the current 8.1 build")
        extract(fetch(fallback), fallback, names)
    if not has_rubberband():
        raise Fail("ffmpeg has no rubberband filter")


# ---------------------------------------------------------------- lls-dsp

def dsp_hash() -> str:
    """Hash of the dsp/ sources (line endings normalised): names the CI release."""
    h = hashlib.sha256()
    d = os.path.join(ROOT, "dsp")
    files = ["Cargo.toml", "Cargo.lock", "build.rs", ".cargo/config.toml"]
    files += sorted("src/" + f for f in os.listdir(os.path.join(d, "src")) if f.endswith(".rs"))
    for rel in files:
        p = os.path.join(d, rel)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(rel.encode() + b"\0" + f.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def dsp_asset() -> str:
    return f"lls-dsp-{PLAT}{EXE}"


def setup_dsp() -> None:
    exe, stamp, want = os.path.join(BIN, "lls-dsp" + EXE), os.path.join(BIN, "lls-dsp.hash"), dsp_hash()
    try:
        with open(stamp) as f:
            if f.read().strip() == want and os.path.isfile(exe):
                return
    except OSError:
        pass
    url = f"{GH}/{REPO}/releases/download/dsp-{want}/{dsp_asset()}"
    try:
        data = fetch(url)
    except urllib.error.HTTPError:
        data = None
    for p in (exe, stamp):                   # an older binary may lack newer subcommands
        if os.path.isfile(p):
            os.remove(p)
    if data:
        os.makedirs(BIN, exist_ok=True)
        with open(exe + ".new", "wb") as f:
            f.write(data)
        os.chmod(exe + ".new", 0o755)
        os.replace(exe + ".new", exe)
        with open(stamp, "w") as f:
            f.write(want)
        return
    say(f"no prebuilt lls-dsp for these dsp/ sources (dsp-{want}; CI publishes it after a push)")
    if cargo():
        build_dsp()
    else:
        say("WARNING: no lls-dsp: the Python fallback is slower (lls setup --dev builds it)")


def rust_env() -> dict:
    env = dict(os.environ)
    rust = os.path.join(LOCAL, "rust")
    env.update(RUSTUP_HOME=os.path.join(rust, "rustup"), CARGO_HOME=os.path.join(rust, "cargo"),
               CARGO_TARGET_DIR=os.path.join(LOCAL, "cargo-target"))
    path = [os.path.join(rust, "cargo", "bin")]
    mingw = os.environ.get("LLS_MINGW") or os.path.join(LOCAL, "mingw64", "bin")
    if WIN and os.path.isdir(mingw):         # a GCC + libclang for the signalsmith feature
        path.append(mingw)
        env.setdefault("LIBCLANG_PATH", mingw)
    env["PATH"] = os.pathsep.join(path + [env.get("PATH", "")])
    return env


def cargo() -> str | None:
    local = os.path.join(LOCAL, "rust", "cargo", "bin", "cargo" + EXE)
    return local if os.path.isfile(local) else shutil.which("cargo")


def setup_mingw() -> None:
    if PLAT != "windows-x64" or os.environ.get("LLS_MINGW") or \
            os.path.isfile(os.path.join(LOCAL, "mingw64", "bin", "gcc.exe")):
        return
    url, sha = WINLIBS
    with zipfile.ZipFile(io.BytesIO(fetch(url, sha))) as z:   # one top folder: mingw64/
        z.extractall(LOCAL)


def setup_rust() -> None:
    setup_mingw()
    if os.path.isfile(os.path.join(LOCAL, "rust", "cargo", "bin", "cargo" + EXE)):
        return
    triple = RUST_HOST[PLAT]
    url = f"https://static.rust-lang.org/rustup/dist/{triple}/rustup-init{EXE}"
    sha = fetch(url + ".sha256").decode().split()[0]
    init = os.path.join(LOCAL, "cache", "rustup-init" + EXE)
    os.makedirs(os.path.dirname(init), exist_ok=True)
    with open(init, "wb") as f:
        f.write(fetch(url, sha))
    os.chmod(init, 0o755)
    # the GNU host on Windows x64 links with MinGW: no Visual Studio needed (ARM64 needs it)
    args = ["-y", "--no-modify-path", "--profile", "minimal", "--default-host", triple]
    if subprocess.call([init, *args], env=rust_env()):
        raise Fail("rustup-init failed")


def build_dsp() -> None:
    exe, env = cargo(), rust_env()
    if not exe:
        raise Fail("no cargo (lls setup --dev)")
    say("building lls-dsp")
    d = os.path.join(ROOT, "dsp")
    if subprocess.call([exe, "build", "-q", "--release", "--features", "signalsmith"], cwd=d, env=env):
        say("signalsmith feature failed (C++ compiler / libclang?): building without it")
        if subprocess.call([exe, "build", "-q", "--release"], cwd=d, env=env):
            raise Fail("cargo build failed")


# ---------------------------------------------------------------- setup

def lock_stamp() -> str:
    with open(os.path.join(ROOT, "uv.lock"), "rb") as f:
        return hashlib.sha256(f.read() + PY.encode()).hexdigest()[:12]


def setup_venv() -> None:
    uv("sync", "--locked", "--python", PY, "--project", ROOT, "-q")
    with open(os.path.join(LOCAL, "venv", "lls.lock"), "w") as f:
        f.write(lock_stamp())


def venv_fresh() -> bool:
    try:
        with open(os.path.join(LOCAL, "venv", "lls.lock")) as f:
            return f.read().strip() == lock_stamp() and os.path.isfile(venv_python("venv"))
    except OSError:
        return False


def relocate() -> None:
    """A copied or moved folder: the venvs hold absolute paths, so
    they are rebuilt (from the cache in .lls, no download)."""
    stamp = os.path.join(LOCAL, "root")
    try:
        with open(stamp) as f:
            old = f.read().strip()
    except OSError:
        old = None
    if old == LOCAL:
        return
    sep = os.path.isdir(os.path.join(LOCAL, "venv-sep"))
    if old is not None:
        say(f"folder moved ({old} -> {LOCAL}): rebuilding the venvs")
        for d in ("venv", "venv-sep", "tools"):
            shutil.rmtree(os.path.join(LOCAL, d), ignore_errors=True)
    os.makedirs(LOCAL, exist_ok=True)
    with open(stamp, "w") as f:
        f.write(LOCAL)
    if old is not None:
        setup_venv()
        if sep:
            setup_stems(mode() == "cloud")


def setup_stems(cloud: bool) -> None:
    if PLAT == "windows-arm64" and not cloud:
        raise Fail("no CUDA torch for Windows ARM: separation goes to the cloud (lls setup --cloud)")
    sep, spy = os.path.join(LOCAL, "venv-sep"), venv_python("venv-sep")
    if not os.path.isfile(spy):
        uv("venv", "-q", "--python", SEP_PY, sep)
    if cloud:
        uv("pip", "install", "-q", "--python", spy, *STEMS_CLOUD)
        return
    torch, index = STEMS_TORCH
    uv("pip", "install", "-q", "--python", spy, torch, "--index-url", index)
    uv("pip", "install", "-q", "--python", spy, *STEMS)
    say("separation model")
    if subprocess.call([spy, os.path.join(ROOT, "gpu", "separate.py"), "--download", MODEL, MODELS],
                       env=run_env(), stdout=subprocess.DEVNULL):
        raise Fail("model download failed")


def setup(args: list[str]) -> int:
    if "--cloud" in args or "--pc" in args:
        os.makedirs(LOCAL, exist_ok=True)
        with open(os.path.join(LOCAL, "mode"), "w") as f:
            f.write("cloud" if "--cloud" in args else "pc")
    cloud = mode() == "cloud"
    say(f"setting up {LOCAL} ({'cloud' if cloud else 'pc'})")
    relocate()
    steps = [("python + venv", setup_venv), ("ffmpeg", setup_ffmpeg)]
    if "--dev" in args:
        steps.append(("rust", setup_rust))
    steps.append(("lls-dsp", build_dsp if "--dev" in args else setup_dsp))
    if "--stems" in args or cloud:
        steps.append(("stems python", lambda: setup_stems(cloud)))
    for name, fn in steps:
        say(name)
        fn()
    return check()


def check() -> int:
    def ok(cond) -> str:
        return "yes" if cond else "no"
    vpy, spy = venv_python("venv"), venv_python("venv-sep")
    env = run_env()

    def out(cmd) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
            return (r.stdout.strip() or r.stderr.strip()).splitlines()[-1] if r.returncode == 0 else "error"
        except (OSError, subprocess.SubprocessError, IndexError):
            return "missing"
    print(f"platform    : {PLAT} ({mode()}), {LOCAL}")
    print(f"venv        : {out([vpy, '-c', 'import sys, losslesssync; print(sys.version.split()[0], losslesssync.ROOT)']) if os.path.isfile(vpy) else 'missing'}"
          f"{'' if venv_fresh() else ' (stale: lls setup)'}")
    print(f"ffmpeg      : {ok(os.path.isfile(os.path.join(BIN, 'ffmpeg' + EXE)))}, rubberband: {ok(has_rubberband())}")
    native = out([vpy, "-c", "from losslesssync.stretch import native; print(native.dsp())"]) if os.path.isfile(vpy) else "?"
    print(f"lls-dsp     : {native} (sources dsp-{dsp_hash()})")
    print(f"gpu         : {out(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']) if shutil.which('nvidia-smi') else 'none'}")
    sep = "missing"
    if os.path.isfile(spy):
        sep = out([spy, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"])
        if sep == "error":
            sep = "no torch (cloud helper)" if mode() == "cloud" else "broken (lls setup --stems)"
    print(f"stems python: {sep}")
    print(f"cargo       : {cargo() or 'no'}")
    print(f"models      : {', '.join(sorted(f for f in os.listdir(MODELS) if not f.startswith('.'))) if os.path.isdir(MODELS) else 'none'}")
    return 0 if os.path.isfile(vpy) and has_rubberband() else 1


# ---------------------------------------------------------------- run

def ensure() -> None:
    relocate()
    if not venv_fresh():
        say("python + venv")
        setup_venv()
    if not has_rubberband():
        say("ffmpeg")
        setup_ffmpeg()


def python(args: list[str]) -> int:
    ensure()
    return subprocess.call([venv_python("venv"), *args], env=run_env())


def main(argv: list[str]) -> int:
    cmd, rest = (argv[0], argv[1:]) if argv else ("help", [])
    try:
        if cmd in ("help", "-h", "--help"):
            print(__doc__)
            return 0
        if cmd == "setup":
            return setup(rest)
        if cmd == "check":
            relocate()
            return check()
        if cmd == "dsp-hash":
            print(dsp_hash())
            return 0
        if cmd == "dsp-asset":
            print(dsp_asset())
            return 0
        if cmd == "cargo":
            exe = cargo()
            if not exe:
                raise Fail("no cargo (lls setup --dev)")
            return subprocess.call([exe, *rest], cwd=os.path.join(ROOT, "dsp"), env=rust_env())
        if cmd == "py":
            return python(rest)
        if cmd == "test":                    # tests/: LLS_CASES picks the synthetic cases
            os.environ["LLS_CASES"] = rest[0] if rest else os.environ.get("LLS_CASES", "")
            return python(["-m", "unittest", "discover", "-v", "-s", os.path.join(ROOT, "tests")])
        if cmd == "gui":
            return python(["-c", "from losslesssync.app.gui import main; main()"])
        return python(["-m", "losslesssync", cmd, *rest])
    except Fail as e:
        say(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
