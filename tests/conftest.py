from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_tools import ffmpeg

RATE = 48000


def sine(freq: float, seconds: float, dbfs: float, rate: int = RATE, channels: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    amp = 10 ** (dbfs / 20.0)
    mono = amp * np.sin(2 * np.pi * freq * t)
    return np.column_stack([mono] * channels) if channels > 1 else mono


def write_wav(path: Path, data: np.ndarray, rate: int = RATE) -> Path:
    samples = np.asarray(data)
    if samples.ndim == 1:
        samples = samples[:, None]
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(samples.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def ffmpeg_available() -> bool:
    try:
        ffmpeg.find_ffmpeg()
        return True
    except ffmpeg.FFmpegNotFound:
        return shutil.which("ffmpeg") is not None


needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    """A 3 s, 997 Hz, -20 dBFS mono sine: reads -23.0 LUFS per BS.1770."""
    return write_wav(tmp_path / "sine.wav", sine(997.0, 3.0, -20.0))
