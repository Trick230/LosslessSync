"""Modal: gpu/separate.py on a cloud GPU. The source goes as bytes in the call and the stem
comes back the same way (no Volumes: they are billed even with a 0 $ spend limit); the
model's weights are baked into the image, so a run downloads nothing.

    modal run gpu/modal_run.py --src SRC.flac --out OUT.flac [--model M]

LLS_MODAL_GPU picks the GPU (default T4, the cheapest; L4 or A10G are faster).
"""
import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
MODEL = "vocals_mel_band_roformer.ckpt"
MODELS = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")      # Triton/Inductor compile C helpers (torch.compile)
    .pip_install("torch", "audio-separator[gpu]", "soundfile", "soxr", "numpy", "scipy")
    .add_local_file(HERE.parent / "losslesssync" / "stretch" / "ext.py", "/root/ext.py", copy=True)
    .add_local_file(HERE / "separate.py", "/root/separate.py", copy=True)
    .run_commands(f"python /root/separate.py --download {MODEL} {MODELS}")
)
app = modal.App("losslesssync-gpu", image=image)


@app.function(gpu=os.environ.get("LLS_MODAL_GPU", "T4"), timeout=3600)
def separate(data: bytes, model: str) -> bytes:
    import sys
    sys.path.insert(0, "/root")
    import separate as job
    src, out = "/tmp/src.flac", "/tmp/vocals.flac"
    pathlib.Path(src).write_bytes(data)
    job.run(src, out, model, MODELS)
    return pathlib.Path(out).read_bytes()


@app.local_entrypoint()
def main(src: str, out: str, model: str = MODEL):
    data = pathlib.Path(src).read_bytes()
    print(f"modal: {len(data) / 1e6:.0f} MB up")
    pathlib.Path(out).write_bytes(separate.remote(data, model))
