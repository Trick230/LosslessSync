"""Runs one GPU job on the first cloud provider that takes it (standard library only).

    python gpu/run_gpu.py separate SRC.flac OUT.flac [--model M]

Order: Modal (free monthly credits, spend limit 0 $), Kaggle (free GPU quota, slower to
start), Colab (no API: prints the manual steps). LONG=1 (long jobs) tries Kaggle first.
A provider without its CLI or token, or that fails (quota, credits), passes the job on.
Tokens come from the environment or the repo's .env (devcloud: dc secrets pull).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = "vocals_mel_band_roformer.ckpt"


class Skip(Exception):
    pass


def log(msg: str) -> None:
    print(f"run_gpu: {msg}", file=sys.stderr, flush=True)


def load_env() -> None:
    """KEY=VALUE lines of ROOT/.env, without overriding the environment."""
    p = os.path.join(ROOT, ".env")
    if not os.path.isfile(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:                    # an empty value would hide ~/.modal.toml or kaggle.json
                    os.environ.setdefault(k.strip(), v)


def tool(name: str) -> str | None:
    """A CLI in .lls/<platform>/bin (lls setup), next to this Python, or on PATH."""
    sys.path.insert(0, ROOT)
    from losslesssync import BIN      # the package's __init__ needs only the standard library
    exe = name + (".exe" if os.name == "nt" else "")
    for d in (BIN, os.path.dirname(sys.executable)):
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    return shutil.which(name)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, stdin=subprocess.DEVNULL, **kw)


def modal(src: str, out: str, model: str) -> None:
    exe = tool("modal")
    if not exe:
        raise Skip("no modal CLI (pip install modal)")
    if not (os.environ.get("MODAL_TOKEN_ID") or os.path.isfile(os.path.expanduser("~/.modal.toml"))):
        raise Skip("no Modal token (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or modal token new)")
    rc = subprocess.call([exe, "run", os.path.join(HERE, "modal_run.py"),
                          "--src", src, "--out", out, "--model", model],
                         stdin=subprocess.DEVNULL, stdout=sys.stderr)
    if rc != 0:
        raise Skip(f"modal run failed ({rc})")


def _kaggle_user() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    p = os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle")), "kaggle.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)["username"]
    raise Skip("no Kaggle token (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)")


def kaggle(src: str, out: str, model: str) -> None:
    """The inputs as a new version of a private dataset, gpu/separate.py as a private GPU
    kernel reading it; waits for the kernel and downloads its output."""
    exe = tool("kaggle")
    if not exe:
        raise Skip("no kaggle CLI (pip install kaggle)")
    user = _kaggle_user()
    ds, kn = f"{user}/lls-gpu-input", f"{user}/lls-gpu-sep"
    with tempfile.TemporaryDirectory() as d:
        data, kern, res = (os.path.join(d, x) for x in ("data", "kernel", "out"))
        for x in (data, kern, res):
            os.makedirs(x)
        shutil.copy(src, os.path.join(data, "src.flac"))
        shutil.copy(os.path.join(ROOT, "losslesssync", "stretch", "ext.py"), data)
        nonce = os.urandom(8).hex()      # the output must come from this upload
        with open(os.path.join(data, "model.txt"), "w") as f:
            f.write(f"{model} {nonce}")
        with open(os.path.join(data, "dataset-metadata.json"), "w") as f:
            json.dump({"title": "lls-gpu-input", "id": ds, "licenses": [{"name": "CC0-1.0"}]}, f)
        if run([exe, "datasets", "status", ds]).returncode == 0:
            r = run([exe, "datasets", "version", "-p", data, "-m", "job", "-q"])
        else:
            r = run([exe, "datasets", "create", "-p", data, "-q"])   # private by default
        if r.returncode != 0:
            raise Skip(f"kaggle dataset upload failed: {(r.stderr or r.stdout).strip()[-300:]}")
        log("kaggle: input uploaded, waiting for the dataset")
        _kaggle_wait(exe, ["datasets", "status", ds], ("ready",), ("error",), 900)
        shutil.copy(os.path.join(HERE, "separate.py"), kern)
        with open(os.path.join(kern, "kernel-metadata.json"), "w") as f:
            json.dump({"id": kn, "title": "lls-gpu-sep", "code_file": "separate.py",
                       "language": "python", "kernel_type": "script", "is_private": True,
                       "enable_gpu": True, "enable_internet": True,
                       "machine_shape": "NvidiaTeslaT4", "dataset_sources": [ds]}, f)
        r = run([exe, "kernels", "push", "-p", kern])
        if r.returncode != 0 or "error" in r.stdout.lower():
            raise Skip(f"kaggle kernel push failed: {(r.stderr or r.stdout).strip()[-300:]}")
        log("kaggle: kernel running")
        _kaggle_wait(exe, ["kernels", "status", kn], ("complete",), ("error", "cancel"), 3 * 3600)
        r = run([exe, "kernels", "output", kn, "-p", res])
        got = os.path.join(res, "vocals.flac")
        if not os.path.isfile(got):
            raise Skip(f"kaggle: no vocals.flac in the output {(r.stderr or r.stdout).strip()[-300:]}")
        try:
            with open(os.path.join(res, "nonce.txt")) as f:
                same = f.read().strip() == nonce
        except OSError:
            same = False
        if not same:
            raise Skip("kaggle: the output is another job's (two jobs at once?)")
        shutil.move(got, out)


def _kaggle_wait(exe, args, ok, bad, timeout) -> None:
    end = time.time() + timeout
    while time.time() < end:
        time.sleep(20)                   # the status right after a push is the previous run's
        s = (run([exe, *args]).stdout or "").lower()
        if any(w in s for w in ok):
            return
        if any(w in s for w in bad):
            raise Skip(f"kaggle: {s.strip()[-300:]}")
    raise Skip("kaggle: timed out")


def colab(src: str, out: str, model: str) -> None:
    raise Skip(
        "Colab has no job API; by hand: open a GPU notebook, upload gpu/separate.py, "
        "losslesssync/stretch/ext.py and the source, then run "
        f"'!pip install -q \"audio-separator[gpu]\" soundfile soxr' and "
        f"'!python separate.py src.flac vocals.flac {model}' and download vocals.flac to {out}")


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "separate":
        print(__doc__, file=sys.stderr)
        return 2
    src, out = os.path.abspath(argv[1]), os.path.abspath(argv[2])
    model = argv[argv.index("--model") + 1] if "--model" in argv else MODEL
    load_env()
    order = [kaggle, modal, colab] if os.environ.get("LONG") == "1" else [modal, kaggle, colab]
    for p in order:
        try:
            log(f"trying {p.__name__}")
            p(src, out, model)
            log(f"{p.__name__}: done")
            return 0
        except Skip as e:
            log(f"{p.__name__}: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
