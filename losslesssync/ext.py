"""Helpers that run in a separate Python with the heavy optional packages.

    python -m losslesssync.ext stretch IN OUT K     Signalsmith Stretch (python-stretch)
    python -m losslesssync.ext separate IN VOCALS REST MODEL MODEL_DIR
                                                    vocals stem (audio-separator, GPU)
                                                    and the residual IN - VOCALS

Only numpy and soundfile from the main package's dependencies are needed here, so the
helper Python can be a venv with torch (see README). Outputs are float WAV.
"""
from __future__ import annotations

import os
import sys
import tempfile


def stretch(src: str, dst: str, k: float):
    import numpy as np
    import python_stretch as ps
    import soundfile as sf
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    st = ps.Signalsmith.Stretch()
    st.preset(x.shape[1], sr)
    st.setTimeFactor(1.0 / k)
    y = st.process(np.ascontiguousarray(x.T))
    del x
    sf.write(dst, y.T, sr, subtype="FLOAT", format="RF64")


def separate(src: str, voc_out: str, rest_out: str, model: str, model_dir: str):
    """Vocals stem at the source rate and channel count, and the residual src - vocals."""
    import logging

    import numpy as np
    import soundfile as sf
    import soxr
    from audio_separator.separator import Separator
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(voc_out))) as tmp:
        # fp16 autocast and 4 chunks per batch: 1.9x faster on an RTX 4060, 66 dB from
        # the fp32 stem (inaudible)
        sep = Separator(output_dir=tmp, output_format="WAV", model_file_dir=model_dir,
                        log_level=logging.WARNING, output_single_stem="Vocals",
                        use_autocast=True,
                        mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                     "batch_size": 4, "overlap": None, "pitch_shift": 0})
        sep.load_model(model_filename=model)
        files = sep.separate(src)
        v, sr_v = sf.read(os.path.join(tmp, os.path.basename(files[0])), dtype="float32",
                          always_2d=True)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    if sr_v != sr:
        v = soxr.resample(v, sr_v, sr, quality="VHQ").astype(np.float32)
    if v.shape[1] != x.shape[1]:
        v = v.mean(axis=1, keepdims=True) if x.shape[1] == 1 else np.repeat(v[:, :1], x.shape[1], 1)
    out = np.zeros_like(x)
    n = min(len(v), len(x))
    out[:n] = v[:n]
    del v
    sf.write(voc_out, out, sr, subtype="FLOAT", format="RF64")
    x -= out
    del out
    sf.write(rest_out, x, sr, subtype="FLOAT", format="RF64")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "stretch":
        stretch(a[0], a[1], float(a[2]))
    elif cmd == "separate":
        separate(*a)
    else:
        sys.exit(f"unknown command {cmd}")
