"""GPU job: the vocals stem of one source, with the model and settings of --keep-engine stems
(losslesssync/stretch/ext.py's separate, vocals only). Runs on a cloud GPU through devcloud
(`dc gpu gpu/separate.json --in SRC.flac --out DIR [-- --model M]`, spec next to this file).

    python separate.py IN_DIR OUT_DIR [--model M]   the one .flac in IN_DIR -> OUT_DIR/vocals.flac
    python separate.py --prepare                     fetch the default model into DC_GPU_CACHE

The stem comes back as 24-bit FLAC at the source's rate, channels and length.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

MODEL = "vocals_mel_band_roformer.ckpt"


def _ext():
    """ext.py: from the package when it is importable, else the copy shipped next to this."""
    try:
        from losslesssync.stretch import ext
        return ext
    except ImportError:
        pass
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext.py")
    if not os.path.isfile(p):
        raise SystemExit("ext.py not found")
    spec = importlib.util.spec_from_file_location("lls_ext", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_dir() -> str:
    return os.environ.get("DC_GPU_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "losslesssync", "models")


def run(src: str, out: str, model: str = MODEL) -> None:
    import soundfile as sf
    ext = _ext()
    wav = out + ".wav"
    g = ext.gain24(src)                  # the source's overs were scaled down by g for FLAC:
    ext.separate(src, wav, "-", model, model_dir())   # the model normalises its input anyway
    v, sr = sf.read(wav, dtype="float32", always_2d=True)
    ext.write24(out, v * g if g != 1.0 else v, sr)
    os.remove(wav)


def prepare(model: str = MODEL) -> None:
    import logging

    from audio_separator.separator import Separator
    Separator(model_file_dir=model_dir(), log_level=logging.WARNING).load_model(model_filename=model)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--prepare"]:
        prepare()
    else:
        srcs = glob.glob(os.path.join(a[0], "*.flac"))
        if len(srcs) != 1:
            raise SystemExit(f"expected one .flac in {a[0]}, found {len(srcs)}")
        run(srcs[0], os.path.join(a[1], "vocals.flac"),
            a[a.index("--model") + 1] if "--model" in a else MODEL)
