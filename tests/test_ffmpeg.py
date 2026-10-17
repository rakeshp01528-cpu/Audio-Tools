import numpy as np
import pytest

from audio_tools import ffmpeg
from tests.conftest import RATE, needs_ffmpeg, sine, write_wav


@pytest.mark.unit
def test_env_dir_takes_priority(tmp_path, monkeypatch):
    fake = tmp_path / f"ffmpeg{ffmpeg.EXE_SUFFIX}"
    fake.write_text("")
    monkeypatch.setenv(ffmpeg.ENV_FFMPEG_DIR, str(tmp_path))
    assert ffmpeg.find_ffmpeg() == str(fake)


@pytest.mark.unit
def test_missing_tool_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(ffmpeg.ENV_FFMPEG_DIR, str(tmp_path))
    monkeypatch.setattr(ffmpeg, "LOCAL_BIN_DIR", tmp_path / "nothing")
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _: None)
    with pytest.raises(ffmpeg.FFmpegNotFound, match="setup"):
        ffmpeg.find_tool("ffmpeg")
    assert ffmpeg.tool_version("ffmpeg") is None


@pytest.mark.unit
def test_encode_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        ffmpeg.encode(np.zeros(10), RATE, tmp_path / "x.xyz", "xyz")


@needs_ffmpeg
@pytest.mark.integration
def test_probe_and_decode_roundtrip(tmp_path):
    original = sine(440.0, 1.0, -6.0, channels=2)
    path = write_wav(tmp_path / "a.wav", original)
    info = ffmpeg.probe(path)
    assert (info.channels, info.sample_rate) == (2, RATE)
    assert info.duration_seconds == pytest.approx(1.0, abs=0.01)
    data, rate = ffmpeg.decode(path)
    assert rate == RATE
    assert data.shape == original.shape
    assert np.max(np.abs(data - original)) < 1e-3  # 16-bit quantization


@needs_ffmpeg
@pytest.mark.integration
def test_probe_fallback_without_ffprobe(tmp_path, monkeypatch):
    path = write_wav(tmp_path / "a.wav", sine(440.0, 1.5, -6.0, channels=2))
    monkeypatch.setattr(ffmpeg, "find_ffprobe", lambda: (_ for _ in ()).throw(ffmpeg.FFmpegNotFound("x")))
    info = ffmpeg.probe(path)
    assert (info.channels, info.sample_rate) == (2, RATE)
    assert info.duration_seconds == pytest.approx(1.5, abs=0.05)


@needs_ffmpeg
@pytest.mark.integration
def test_encode_creates_parent_dirs(tmp_path):
    out = tmp_path / "deep" / "er" / "out.wav"
    ffmpeg.encode(sine(440.0, 0.5, -6.0), RATE, out, "wav")
    assert out.is_file()


@needs_ffmpeg
@pytest.mark.integration
def test_decode_garbage_file_raises(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.decode(bad)


@pytest.mark.unit
def test_tool_version_handles_broken_binary(tmp_path, monkeypatch):
    broken = tmp_path / f"ffmpeg{ffmpeg.EXE_SUFFIX}"
    broken.write_text("not a program")
    broken.chmod(0o755)
    monkeypatch.setenv(ffmpeg.ENV_FFMPEG_DIR, str(tmp_path))
    assert ffmpeg.tool_version("ffmpeg") is None
