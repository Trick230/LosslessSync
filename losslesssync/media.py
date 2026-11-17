"""ffmpeg/ffprobe wrappers: probing, decoding and encoding."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


class MediaError(RuntimeError):
    pass


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"'{name}' not found in PATH")
    return path


def run(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                       stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace")[-3000:]
        raise MediaError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{err}")
    return r


def _parse_duration(s) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    try:  # mkv tag form HH:MM:SS.nnnnnnnnn
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return None


@dataclass
class AudioStream:
    index: int          # absolute stream index in the container
    aidx: int           # audio-relative index (for -map 0:a:N)
    codec: str
    channels: int
    sample_rate: int
    layout: str
    language: str
    title: str
    duration: float | None
    start_time: float   # relative to the container start (seconds)

    def describe(self) -> str:
        d = f"{self.duration / 60:.1f} min" if self.duration else "?"
        t = f" '{self.title}'" if self.title else ""
        return (f"a:{self.aidx} (#{self.index}) {self.codec} {self.channels}ch "
                f"{self.sample_rate} Hz lang={self.language or '-'}{t} dur={d}")


def probe(path: str) -> dict:
    r = run([tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    return json.loads(r.stdout)


def audio_streams(path: str) -> list[AudioStream]:
    info = probe(path)
    fmt = info.get("format", {})
    fmt_start = float(fmt.get("start_time", 0) or 0)
    fmt_dur = _parse_duration(fmt.get("duration"))
    out = []
    for s in info.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        dur = _parse_duration(s.get("duration")) or _parse_duration(tags.get("duration")) or fmt_dur
        st = float(s.get("start_time", fmt_start) or fmt_start)
        out.append(AudioStream(
            index=int(s["index"]), aidx=len(out), codec=s.get("codec_name", "?"),
            channels=int(s.get("channels", 2)), sample_rate=int(s.get("sample_rate", 48000)),
            layout=s.get("channel_layout", ""), language=tags.get("language", ""),
            title=tags.get("title", ""), duration=dur, start_time=st - fmt_start))
    if not out:
        raise MediaError(f"no audio streams in {path}")
    return out


def pick_stream(streams: list[AudioStream], explicit: int | None = None,
                prefer: tuple[str, ...] = (), avoid: tuple[str, ...] = ()) -> AudioStream:
    if explicit is not None:
        for s in streams:
            if s.aidx == explicit:
                return s
        raise MediaError(f"audio track a:{explicit} does not exist ({len(streams)} audio tracks)")
    for lang in prefer:
        for s in streams:
            if s.language.lower().startswith(lang):
                return s
    for s in streams:
        if s.language.lower() not in avoid:
            return s
    return streams[0]


def _decoder_opts(stream: AudioStream) -> list[str]:
    # Never let the decoder apply dynamic range compression to the material we deliver.
    if stream.codec in ("ac3", "eac3"):
        return ["-drc_scale", "0"]
    return []


def decode_mono(path: str, stream: AudioStream, sr: int = 16000) -> np.ndarray:
    cmd = [tool("ffmpeg"), "-nostdin", "-v", "error", *_decoder_opts(stream), "-i", path,
           "-map", f"0:a:{stream.aidx}", "-vn", "-sn", "-dn",
           "-af", f"aresample={sr}:resampler=soxr", "-ac", "1", "-f", "f32le", "-"]
    r = run(cmd)
    return np.frombuffer(r.stdout, dtype=np.float32)


RB_ENGINE_FINER = 0x20000000


def speed_filter(k: float, sr: int, mode: str) -> str:
    """Filter that stretches the source timeline by k (k > 1 slows down / lengthens).

    mode 'resample': plain resampling (pitch changes with speed, i.e. undoes a PAL speed-up).
    mode 'tempo': time stretch keeping pitch (source whose pitch was already corrected).
    mode 'keep': time stretch keeping pitch with Rubber Band R3 (finer engine), formants
    preserved and channels linked; used to keep the PAL pitch on the film timing.
    """
    if abs(k - 1.0) < 1e-12:
        return "anull"
    if mode == "keep":
        # ffmpeg ORs its option fields into the RubberBandOptions word; it has no engine
        # field, so the R3 bit (RubberBandOptionEngineFiner) rides on 'transients'
        return (f"rubberband=tempo={1.0 / k:.12f}:transients={RB_ENGINE_FINER}"
                f":formant=preserved:channels=together")
    if mode == "tempo":
        return f"rubberband=tempo={1.0 / k:.12f}:pitchq=quality:transients=mixed"
    frac = Fraction(k).limit_denominator(40000)
    # Relabel to sr*den, resample to sr*num, relabel back: N samples become N*num/den exactly.
    a, b = sr * frac.denominator, sr * frac.numerator
    while a > 2**31 - 1 or b > 2**31 - 1:
        frac = Fraction(k).limit_denominator(frac.denominator // 2)
        a, b = sr * frac.denominator, sr * frac.numerator
    return (f"asetrate={a},aresample={b}:resampler=soxr:precision=28:cheby=1,"
            f"asetrate={sr}")


def make_intermediate(path: str, stream: AudioStream, k: float, mode: str,
                      out_wav: str, ana_sr: int = 16000) -> np.ndarray:
    """Decode the source with speed correction to a float WAV; return mono analysis copy."""
    sr = stream.sample_rate
    fc = (f"[0:a:{stream.aidx}]{speed_filter(k, sr, mode)},asplit=2[full][ana];"
          f"[ana]aresample={ana_sr}:resampler=soxr,aformat=channel_layouts=mono[m]")
    cmd = [tool("ffmpeg"), "-nostdin", "-v", "error", "-y", *_decoder_opts(stream), "-i", path,
           "-filter_complex", fc,
           "-map", "[full]", "-c:a", "pcm_f32le", "-rf64", "auto", "-f", "wav", out_wav,
           "-map", "[m]", "-f", "f32le", "-"]
    r = run(cmd)
    return np.frombuffer(r.stdout, dtype=np.float32)


def decode_to_wav(path: str, stream: AudioStream, out_wav: str, channels: int, sr: int):
    cmd = [tool("ffmpeg"), "-nostdin", "-v", "error", "-y", *_decoder_opts(stream), "-i", path,
           "-map", f"0:a:{stream.aidx}", "-ac", str(channels),
           "-af", f"aresample={sr}:resampler=soxr", "-c:a", "pcm_f32le", "-rf64", "auto",
           "-f", "wav", out_wav]
    run(cmd, capture=False)


CODECS = {
    "flac": ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24",
             "-compression_level", "8"],
    "wav": ["-c:a", "pcm_s24le", "-rf64", "auto"],
    "ac3": ["-c:a", "ac3"],
    "eac3": ["-c:a", "eac3"],
    "aac": ["-c:a", "aac"],
    "opus": ["-c:a", "libopus"],
}
EXT_CODEC = {".flac": "flac", ".wav": "wav", ".w64": "wav", ".ac3": "ac3", ".eac3": "eac3",
             ".ec3": "eac3", ".m4a": "aac", ".aac": "aac", ".opus": "opus", ".mka": "flac"}


def encode(in_wav: str, out_path: str, codec: str, bitrate: str | None, language: str = "por"):
    cmd = [tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", in_wav, *CODECS[codec]]
    if bitrate and codec not in ("flac", "wav"):
        cmd += ["-b:a", bitrate]
    if not out_path.lower().endswith((".wav", ".w64", ".ac3", ".eac3", ".ec3", ".aac")):
        cmd += ["-metadata:s:a:0", f"language={language}"]
    cmd.append(out_path)
    run(cmd, capture=False)


def mux(ref_path: str, audio_path: str, out_path: str, delay_s: float, n_ref_audio: int,
        language: str = "por", title: str = "Português", default: bool = False):
    cmd = [tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", ref_path]
    if abs(delay_s) > 1e-9:
        cmd += ["-itsoffset", f"{delay_s:.6f}"]
    cmd += ["-i", audio_path, "-map", "0", "-map", "1:a:0", "-c", "copy",
            f"-metadata:s:a:{n_ref_audio}", f"language={language}",
            f"-metadata:s:a:{n_ref_audio}", f"title={title}"]
    if default:
        cmd += ["-disposition:a", "0", f"-disposition:a:{n_ref_audio}", "default"]
    cmd.append(out_path)
    run(cmd, capture=False)
