"""Helpers that run in a separate Python with the heavy optional packages.

    python -m losslesssync.ext stretch IN OUT K     Signalsmith Stretch (python-stretch)
    python -m losslesssync.ext separate IN VOCALS REST_K MODEL MODEL_DIR
                                                    vocals stem (audio-separator's RoFormer,
                                                    GPU); reads K from stdin and stretches
                                                    the residual IN - VOCALS with
                                                    Signalsmith into REST_K, streamed
                                                    through lls-dsp when it is built

Only numpy and soundfile from the main package's dependencies are needed here, so the
helper Python can be a venv with torch (see README). Outputs are float WAV.
"""
from __future__ import annotations

import os
import sys


def _signalsmith(x, sr: int, k: float):
    """(frames, channels) float32 -> stretched (channels, frames)."""
    import numpy as np
    import python_stretch as ps
    st = ps.Signalsmith.Stretch()
    st.preset(x.shape[1], sr)
    st.setTimeFactor(1.0 / k)
    return st.process(np.ascontiguousarray(x.T))


def stretch(src: str, dst: str, k: float):
    import soundfile as sf
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    y = _signalsmith(x, sr, k)
    del x
    sf.write(dst, y.T, sr, subtype="FLOAT", format="RF64")


def _stretcher(dst: str, channels: int, sr: int):
    """Returns start(k) -> (feed(block), close(), abort()) for the residual's Signalsmith stretch:
    streamed through lls-dsp when it has the subcommand, else python-stretch at the end."""
    import subprocess

    import numpy as np
    try:
        from .native import dsp
        exe = dsp()
    except ImportError:
        exe = None
    if exe:
        r = subprocess.run([exe, "stretch"], stdin=subprocess.DEVNULL, capture_output=True)
        if b"usage: lls-dsp stretch" not in r.stderr:
            exe = None                   # an older build, WSOLA only

    def start(k: float):
        if exe:
            p = subprocess.Popen([exe, "stretch", dst, str(channels), str(sr), repr(k)],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)

            def close():
                p.stdin.close()
                if p.wait():
                    raise RuntimeError(f"lls-dsp stretch failed ({p.returncode})")

            def abort():
                if p.poll() is None:
                    p.kill()
                p.wait()
            return (lambda b: p.stdin.write(np.ascontiguousarray(b, np.float32).tobytes()),
                    close, abort)
        blocks = []

        def close_ps():
            import soundfile as sf
            x = np.concatenate(blocks) if blocks else np.zeros((0, channels), np.float32)
            blocks.clear()
            y = _signalsmith(x, sr, k)
            del x
            sf.write(dst, y.T, sr, subtype="FLOAT", format="RF64")
        return blocks.append, close_ps, blocks.clear
    return start


def separate(src: str, voc_out: str, rest_k_out: str, model: str, model_dir: str):
    """Vocals stem at the source rate and channel count; prints "vocals" on stdout once it is
    written. K comes on stdin, sent once the sync knows it, usually during the separation:
    the residual src - vocals is then stretched by K into REST_K as the stem comes out of
    the GPU, so little of the stretch is left when the vocals are done."""
    import contextlib
    import gc
    import logging
    import queue
    import shutil
    import tempfile
    import threading

    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from audio_separator.separator import Separator
    from audio_separator.separator import separator as sep_mod
    from scipy import signal
    # the GPU does the work: more CPU threads only spin (T05E01: 1.9 cores -> 0.75, same
    # wall time), and in a batch they slow the previous episode's stretch down
    torch.set_num_threads(1)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    n, ch = x.shape

    ks: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: ks.put(sys.stdin.readline().strip() or None),
                     daemon=True).start()
    rest: queue.Queue = queue.Queue()
    err = []
    start = _stretcher(rest_k_out, ch, sr)

    def rest_writer():
        ended = False
        abort = None
        try:
            k = ks.get()
            if k is None:
                return                   # the caller went away
            feed, close, abort = start(float(k))
            while True:
                blk = rest.get()
                if blk is None or blk is abort_mark:
                    ended = True
                    if blk is abort_mark:
                        raise RuntimeError("separation failed")
                    break
                feed(blk)
            close()
        except BaseException as e:       # reported by the main thread
            err.append(e)
            if abort:
                with contextlib.suppress(Exception):
                    abort()
            while not ended:
                blk = rest.get()
                ended = blk is None or blk is abort_mark
    abort_mark = object()
    writer = threading.Thread(target=rest_writer, daemon=True)
    writer.start()

    # fp16 autocast: 1.9x faster on an RTX 4060, 66 dB from the fp32 stem (inaudible)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(voc_out)))  # fallback's stem
    sep = Separator(output_dir=tmp, output_format="WAV",
                    model_file_dir=model_dir, log_level=logging.WARNING,
                    output_single_stem="Vocals", use_autocast=True,
                    mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                 "batch_size": 4, "overlap": None, "pitch_shift": 0})
    sep.load_model(model_filename=model)
    inst = sep.model_instance
    cfg = inst.model_data_cfgdict
    sr_m = inst.sample_rate
    # the same input as audio-separator's prepare_mix (librosa.load: soxr HQ, mono doubled)
    mix = soxr.resample(x, sr, sr_m, quality="HQ").T
    if mix.shape[0] == 1:
        mix = np.concatenate([mix, mix])
    # and its normalisation to a 0.9 peak, undone on the stem (audio-separator normalises
    # the stem on its own instead, so it no longer matched x when the mix was scaled)
    peak = float(np.abs(mix).max()) if mix.size else 0.0
    thr = inst.normalization_threshold
    scale = thr / peak if peak > thr else 1.0
    mix = torch.from_numpy(np.ascontiguousarray(mix * scale, np.float32))
    length = mix.shape[1]
    own = bool(getattr(inst, "is_roformer", False) and cfg.training.target_instrument
               and inst.pitch_shift == 0
               and not inst._use_model_segment_override(length / sr_m))

    def blocks():
        """The stem at sr_m, (frames, channels), in order."""
        if not own:
            # other models, or inputs under 10 s: audio-separator's own path, the stem
            # comes whole at the end (normalised by audio-separator, as before streaming)
            files = sep.separate(src)
            v, sr_v = sf.read(os.path.join(tmp, os.path.basename(files[0])),
                              dtype="float32", always_2d=True)
            yield v if sr_v == sr_m else soxr.resample(v, sr_v, sr_m, quality="VHQ")
            return
        # MDXCSeparator.demix's chunking (override_model_segment_size is off)
        hop = getattr(cfg.model, "stft_hop_length", None) or cfg.audio.hop_length
        chunk = int(hop) * (int(cfg.inference.dim_t) - 1)
        starts = inst._roformer_chunk_starts(length, chunk, chunk // inst.overlap)
        device = next(inst.model_run.parameters()).device
        window = torch.tensor(signal.windows.hamming(chunk), dtype=torch.float32, device=device)
        acc = torch.zeros((mix.shape[0], chunk), dtype=torch.float32, device=device)
        cnt = torch.zeros(chunk, dtype=torch.float32, device=device)
        dev_type = getattr(inst, "torch_device", device).type
        amp = (torch.autocast(dev_type)
               if sep.effective_precision == sep_mod.AUTOCAST and dev_type != "privateuseone"
               else contextlib.nullcontext())
        with torch.no_grad(), amp:
            # MDXCSeparator.demix's overlap-add, handing on each stretch of the stem as
            # soon as no later chunk reaches it
            for i, s in enumerate(starts):
                part = mix[:, s:s + chunk]
                y = inst._run_roformer_model(part.to(device))
                y = y.reshape(-1, y.shape[-2], y.shape[-1])[0].float()
                m = min(part.shape[-1], y.shape[-1], chunk)
                acc[:, :m] += y[:, :m] * window[:m]
                cnt[:m] += window[:m]
                e = (starts[i + 1] if i + 1 < len(starts) else length) - s
                v = (acc[:, :e] / cnt[:e].clamp(min=1e-10)).T.cpu().numpy() / scale
                acc[:, :chunk - e] = acc[:, e:].clone()
                acc[:, chunk - e:] = 0
                cnt[:chunk - e] = cnt[e:].clone()
                cnt[chunk - e:] = 0
                yield v

    rs = soxr.ResampleStream(sr_m, sr, ch, dtype="float32", quality="VHQ")
    voc = sf.SoundFile(voc_out, "w", sr, ch, subtype="FLOAT", format="RF64")
    done = 0                             # output frames written

    def emit(v, last=False):
        nonlocal done
        if v.shape[1] != ch:
            v = v.mean(axis=1, keepdims=True) if ch == 1 else np.repeat(v[:, :1], ch, 1)
        v = rs.resample_chunk(np.ascontiguousarray(v, np.float32), last=last)[:n - done]
        if last and done + len(v) < n:
            v = np.concatenate([v, np.zeros((n - done - len(v), ch), np.float32)])
        voc.write(v)
        rest.put(x[done:done + len(v)] - v)
        done += len(v)

    try:
        for v in blocks():
            emit(v)
        emit(np.zeros((0, ch), np.float32), last=True)
        voc.close()
    except BaseException:
        voc.close()
        ks.put(None)                     # the writer may still be waiting for K
        rest.put(abort_mark)             # stops lls-dsp instead of letting it finish
        writer.join()
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rest.put(None)
    del x, mix, inst, sep
    # free the VRAM now: the next episode's separation starts on "vocals" below,
    # while this process may still be finishing the residual's stretch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _PROTO.write("vocals\n")
    _PROTO.flush()
    writer.join()
    if err:
        raise err[0]


_PROTO = sys.stdout

if __name__ == "__main__":
    # stdout carries the protocol; anything the libraries print goes to stderr
    _PROTO = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    cmd, *a = sys.argv[1:]
    if cmd == "stretch":
        stretch(a[0], a[1], float(a[2]))
    elif cmd == "separate":
        separate(*a)
    else:
        sys.exit(f"unknown command {cmd}")
