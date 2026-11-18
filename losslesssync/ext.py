"""Helpers that run in a separate Python with the heavy optional packages.

    python -m losslesssync.ext stretch IN OUT K     Signalsmith Stretch (python-stretch)
    python -m losslesssync.ext separate IN OUT MODEL MODEL_DIR
                                                    vocals stem (audio-separator, GPU)

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


def separate(src: str, dst: str, model: str, model_dir: str):
    import logging

    from audio_separator.separator import Separator
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(dst))) as tmp:
        sep = Separator(output_dir=tmp, output_format="WAV", model_file_dir=model_dir,
                        log_level=logging.WARNING, output_single_stem="Vocals")
        sep.load_model(model_filename=model)
        files = sep.separate(src)
        os.replace(os.path.join(tmp, os.path.basename(files[0])), dst)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "stretch":
        stretch(a[0], a[1], float(a[2]))
    elif cmd == "separate":
        separate(*a)
    else:
        sys.exit(f"unknown command {cmd}")
