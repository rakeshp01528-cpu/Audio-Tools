from pathlib import Path

import numpy as np
import pytest

from audio_tools import bs1770, pipeline
from audio_tools.pipeline import NormalizeOptions, default_output_path, normalize_samples, true_peak_dbfs
from tests.conftest import RATE, needs_ffmpeg, sine, write_wav


@pytest.mark.unit
def test_true_peak_full_scale_sine_is_about_zero():
    assert true_peak_dbfs(sine(997.0, 1.0, 0.0)) == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_true_peak_uses_loudest_channel():
    left = sine(997.0, 1.0, -20.0)
    right = sine(997.0, 1.0, 0.0)
    assert true_peak_dbfs(np.column_stack([left, right])) == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_true_peak_sees_inter_sample_peaks():
    from tests.test_limiter import faded

    t = np.arange(RATE)
    x = faded(np.sin(2 * np.pi * 0.25 * t + np.pi / 4))  # samples at -3 dBFS, true peak 0 dBFS
    assert true_peak_dbfs(x) == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_true_peak_of_silence():
    assert true_peak_dbfs(np.zeros(100)) == float("-inf")
    assert true_peak_dbfs(np.zeros(0)) == float("-inf")


@pytest.mark.unit
def test_default_output_path():
    out = default_output_path("/music/Song Name.wav", "mp3")
    assert out == Path("/music/Song Name_normalized.mp3")
    assert default_output_path("a/b.wav", ".flac", "out") == Path("out/b_normalized.flac")


@pytest.mark.unit
def test_normalize_samples_hits_target():
    data = sine(997.0, 3.0, -20.0)  # -23 LUFS
    out, stats = normalize_samples(data, RATE, NormalizeOptions(target_lufs=-30.0, true_peak_ceiling_db=-1.0))
    assert stats["gain_db"] == pytest.approx(-7.0, abs=0.15)
    assert stats["limiter_reduction_db"] == 0.0
    assert bs1770.integrated_loudness(out, RATE) == pytest.approx(stats["achieved_lufs"], abs=0.15)


@pytest.mark.unit
def test_loudness_mode_limiter_hits_ceiling():
    data = sine(997.0, 3.0, -20.0)  # -23 LUFS, peak -20 dBFS
    out, stats = normalize_samples(data, RATE, NormalizeOptions(target_lufs=-3.0, true_peak_ceiling_db=-1.0))
    # +20 dB would put the peak at 0 dBFS; the limiter holds it at -1 (steady sine -> ~1 dB quieter).
    assert stats["limiter_reduction_db"] == pytest.approx(1.0, abs=0.1)
    assert stats["true_peak_after_db"] == pytest.approx(-1.0, abs=0.05)
    assert stats["achieved_lufs"] == pytest.approx(-4.0, abs=0.2)
    assert true_peak_dbfs(out) == pytest.approx(-1.0, abs=0.05)


@pytest.mark.unit
def test_loudness_mode_zero_dbfs_ceiling():
    data = sine(997.0, 3.0, -20.0)
    out, stats = normalize_samples(data, RATE, NormalizeOptions(target_lufs=-3.0, true_peak_ceiling_db=0.0))
    # The meter is within ~0.05 dB of the textbook value, so the limiter barely touches it.
    assert stats["limiter_reduction_db"] <= 0.1
    assert stats["achieved_lufs"] == pytest.approx(-3.0, abs=0.1)
    assert true_peak_dbfs(out) <= 0.0 + 1e-3
    assert true_peak_dbfs(out) == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_loudness_mode_without_limiter_turns_whole_file_down():
    data = sine(997.0, 3.0, -20.0)
    opts = NormalizeOptions(target_lufs=-3.0, true_peak_ceiling_db=-1.0, use_limiter=False)
    out, stats = normalize_samples(data, RATE, opts)
    assert stats["limiter_reduction_db"] == 0.0
    assert stats["gain_db"] == pytest.approx(19.0, abs=0.1)
    assert stats["achieved_lufs"] == pytest.approx(-4.0, abs=0.1)
    assert true_peak_dbfs(out) == pytest.approx(-1.0, abs=0.05)


@pytest.mark.unit
@pytest.mark.parametrize("target", [-1.0, 0.0, -6.0])
def test_peak_mode(target):
    data = sine(997.0, 1.0, -20.0, channels=2)
    out, stats = normalize_samples(data, RATE, NormalizeOptions.peak(target))
    assert stats["mode"] == "peak"
    assert stats["target_lufs"] is None
    assert stats["gain_db"] == pytest.approx(20.0 + target, abs=0.05)
    assert true_peak_dbfs(out) == pytest.approx(target, abs=0.05)
    assert bs1770.integrated_loudness(out, RATE) == pytest.approx(stats["achieved_lufs"], abs=0.05)


@pytest.mark.unit
def test_peak_mode_rejects_silence():
    with pytest.raises(ValueError, match="silence"):
        normalize_samples(np.zeros(RATE), RATE, NormalizeOptions.peak(-1.0))


@pytest.mark.unit
def test_options_validation():
    with pytest.raises(ValueError, match="0 dBFS or lower"):
        NormalizeOptions(true_peak_ceiling_db=0.5)
    with pytest.raises(ValueError, match="0 dBFS or lower"):
        NormalizeOptions.peak(1.0)
    with pytest.raises(ValueError, match="mode"):
        NormalizeOptions(mode="banana")


@pytest.mark.unit
def test_normalize_samples_rejects_silence():
    with pytest.raises(ValueError, match="silent"):
        normalize_samples(np.zeros(RATE), RATE, NormalizeOptions())


@pytest.mark.unit
def test_normalize_file_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.normalize_file(tmp_path / "nope.wav")


@needs_ffmpeg
@pytest.mark.integration
def test_normalize_file_same_path_rejected(sine_wav):
    with pytest.raises(ValueError, match="differ"):
        pipeline.normalize_file(sine_wav, sine_wav, NormalizeOptions(output_format="wav"))


@needs_ffmpeg
@pytest.mark.integration
@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3"])
def test_normalize_file_end_to_end(sine_wav, tmp_path, fmt):
    options = NormalizeOptions(target_lufs=-16.0, output_format=fmt, bitrate="192k")
    result = pipeline.normalize_file(sine_wav, tmp_path / f"out.{fmt}", options)
    assert Path(result.output_path).is_file()
    assert result.measured_lufs == pytest.approx(-23.0, abs=0.15)
    assert result.achieved_lufs == pytest.approx(-16.0, abs=0.15)
    measured = pipeline.measure_file(result.output_path)
    assert measured.integrated_lufs == pytest.approx(-16.0, abs=0.3)


@needs_ffmpeg
@pytest.mark.integration
def test_normalize_file_default_output_name(sine_wav):
    result = pipeline.normalize_file(sine_wav, options=NormalizeOptions(output_format="flac"))
    assert result.output_path == str(sine_wav.with_name("sine_normalized.flac"))
    assert Path(result.output_path).is_file()


@needs_ffmpeg
@pytest.mark.integration
def test_measure_file_stereo(tmp_path):
    path = write_wav(tmp_path / "st.wav", sine(997.0, 2.0, -20.0, channels=2))
    m = pipeline.measure_file(path)
    assert m.channels == 2
    assert m.sample_rate == RATE
    assert m.duration_seconds == pytest.approx(2.0, abs=0.01)
    assert m.integrated_lufs == pytest.approx(-20.0, abs=0.15)
    assert m.true_peak_dbfs == pytest.approx(-20.0, abs=0.1)


@needs_ffmpeg
@pytest.mark.integration
def test_six_channel_input_is_downmixed_to_stereo(tmp_path):
    path = write_wav(tmp_path / "six.wav", sine(997.0, 1.0, -20.0, channels=6))
    m = pipeline.measure_file(path)
    assert m.channels == 2
    out = pipeline.normalize_file(path, tmp_path / "six_out.flac", NormalizeOptions(output_format="flac"))
    assert Path(out.output_path).is_file()


@needs_ffmpeg
@pytest.mark.integration
def test_filenames_starting_with_dash_and_colon(tmp_path):
    for name in ("-dash.wav", "weird:name.wav"):
        path = write_wav(tmp_path / name, sine(997.0, 1.0, -20.0))
        result = pipeline.normalize_file(path, options=NormalizeOptions(output_format="flac"))
        assert Path(result.output_path).is_file()


@needs_ffmpeg
@pytest.mark.integration
def test_encode_leaves_no_partial_files(sine_wav, tmp_path):
    out_dir = tmp_path / "out"
    pipeline.normalize_file(sine_wav, out_dir / "x.wav", NormalizeOptions(output_format="wav"))
    assert [p.name for p in out_dir.iterdir()] == ["x.wav"]
