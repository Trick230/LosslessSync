"""Minimal float32 WAV / RF64 reader (memory-mapped) and writer."""
from __future__ import annotations

import struct

import numpy as np

_KSDATAFORMAT_FLOAT = bytes.fromhex("0300000000001000800000aa00389b71")


class WavInfo:
    def __init__(self, path, offset, frames, channels, sr, mask):
        self.path, self.offset, self.frames = path, offset, frames
        self.channels, self.sr, self.mask = channels, sr, mask

    def memmap(self) -> np.ndarray:
        return np.memmap(self.path, dtype="<f4", mode="r", offset=self.offset,
                         shape=(self.frames, self.channels))


def read_info(path: str) -> WavInfo:
    with open(path, "rb") as f:
        head = f.read(12)
        riff, _, wave = struct.unpack("<4sI4s", head)
        if riff not in (b"RIFF", b"RF64") or wave != b"WAVE":
            raise ValueError(f"{path}: not a WAV/RF64 file")
        ds64_data = None
        fmt = None
        while True:
            h = f.read(8)
            if len(h) < 8:
                raise ValueError(f"{path}: no data chunk")
            cid, size = struct.unpack("<4sI", h)
            if cid == b"ds64":
                body = f.read(size)
                ds64_data = struct.unpack("<Q", body[8:16])[0]
            elif cid == b"fmt ":
                body = f.read(size)
                tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                mask = 0
                if tag == 0xFFFE and size >= 40:
                    mask = struct.unpack("<I", body[20:24])[0]
                    tag = struct.unpack("<H", body[24:26])[0]
                if tag != 3 or bits != 32:
                    raise ValueError(f"{path}: expected 32-bit float PCM")
                fmt = (ch, sr, mask)
            elif cid == b"data":
                if fmt is None:
                    raise ValueError(f"{path}: data before fmt")
                if size == 0xFFFFFFFF and ds64_data is not None:
                    size = ds64_data
                offset = f.tell()
                ch, sr, mask = fmt
                f.seek(0, 2)
                size = min(size, f.tell() - offset)
                return WavInfo(path, offset, size // (4 * ch), ch, sr, mask)
            else:
                f.seek(size + (size & 1), 1)
            if cid in (b"ds64", b"fmt ") and size & 1:
                f.seek(1, 1)


class WavWriter:
    """Writes float32 WAV (RF64 when larger than 4 GB) with a known final length."""

    def __init__(self, path: str, frames: int, channels: int, sr: int, mask: int = 0):
        self.f = open(path, "wb")   # noqa: SIM115 - the writer's own file, closed by close()
        data = frames * channels * 4
        rf64 = data + 100 > 0xFFFFFFFF
        fmt = struct.pack("<HHIIHHHHI16s", 0xFFFE, channels, sr, sr * channels * 4,
                          channels * 4, 32, 22, 32, mask, _KSDATAFORMAT_FLOAT)
        body = b"WAVE"
        if rf64:
            ds64 = struct.pack("<QQQI", 4 + 36 + 8 + len(fmt) + 8 + data, data, frames, 0)
            body += b"ds64" + struct.pack("<I", len(ds64)) + ds64
        body += b"fmt " + struct.pack("<I", len(fmt)) + fmt
        body += b"data" + struct.pack("<I", 0xFFFFFFFF if rf64 else data)
        riff_size = 0xFFFFFFFF if rf64 else len(body) + data
        self.f.write((b"RF64" if rf64 else b"RIFF") + struct.pack("<I", riff_size) + body)
        self.remaining = frames
        self.channels = channels

    def write(self, block: np.ndarray):
        block = np.ascontiguousarray(block, dtype="<f4")
        assert block.ndim == 2 and block.shape[1] == self.channels
        if len(block) > self.remaining:
            raise ValueError("writing past declared length")
        self.f.write(block.tobytes())
        self.remaining -= len(block)

    def write_silence(self, n: int, chunk: int = 1 << 18):
        while n > 0:
            m = min(n, chunk)
            self.write(np.zeros((m, self.channels), np.float32))
            n -= m

    def close(self):
        if self.remaining:
            self.write_silence(self.remaining)
        self.f.close()
