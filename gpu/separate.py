"""GPU job: the vocals stem of one source, with the model and settings of --keep-engine stems
(losslesssync/stretch/ext.py's separate, vocals only). Runs on Modal (modal_run.py) and as
the Kaggle kernel (run_gpu.py), never locally.

    python separate.py SRC.flac OUT.flac [MODEL] [MODEL_DIR]
    python separate.py                      on Kaggle (KAGGLE_KERNEL_RUN_TYPE): the job's dataset
    python separate.py --download MODEL DIR fetch the model only (Modal image build)

The stem comes back as 24-bit FLAC at the source's rate, channels and length.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys

MODEL = "vocals_mel_band_roformer.ckpt"


def _ext(near: str):
    """ext.py: from the package when it is importable, else the copy next to the inputs."""
    try:
        from losslesssync.stretch import ext
        return ext
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for d in (here, near):
        p = os.path.join(d, "ext.py")
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("lls_ext", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("ext.py not found")


def run(src: str, out: str, model: str = MODEL, model_dir: str = "") -> None:
    import soundfile as sf
    model_dir = model_dir or os.path.join(os.path.expanduser("~"), ".cache", "losslesssync", "models")
    ext = _ext(os.path.dirname(os.path.abspath(src)))
    wav = out + ".wav"
    g = ext.gain24(src)                  # the source's overs were scaled down by g for FLAC:
    ext.separate(src, wav, "-", model, model_dir)   # the model normalises its input anyway
    v, sr = sf.read(wav, dtype="float32", always_2d=True)
    ext.write24(out, v * g if g != 1.0 else v, sr)
    os.remove(wav)


def download(model: str, model_dir: str) -> None:
    import logging
    from audio_separator.separator import Separator
    Separator(model_file_dir=model_dir, log_level=logging.WARNING).load_model(model_filename=model)


def kaggle() -> None:
    # the input dataset's mount point has changed across Kaggle versions: search for it
    src = (glob.glob("/kaggle/input/**/src.flac", recursive=True) or [None])[0]
    if not src:
        raise SystemExit("src.flac not in /kaggle/input")
    d = os.path.dirname(src)
    with open(os.path.join(d, "model.txt")) as f:
        model, nonce = (f.read().split() + [MODEL, ""])[:2]
    with open("/kaggle/working/nonce.txt", "w") as f:   # ties the output to this upload
        f.write(nonce)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "audio-separator[gpu]", "soundfile", "soxr"])
    run(src, "/kaggle/working/vocals.flac", model, "/tmp/models")


if __name__ == "__main__":
    a = sys.argv[1:]
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        kaggle()
    elif a[0] == "--download":
        download(a[1], a[2])
    else:
        run(*a)
