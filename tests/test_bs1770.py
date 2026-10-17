import numpy as np
import pytest

from audio_tools import bs1770
from tests.conftest import RATE, sine


@pytest.mark.unit
def test_mono_sine_minus_20_dbfs_reads_minus_23_lufs():
    lufs = bs1770.integrated_loudness(sine(997.0, 3.0, -20.0), RATE)
    assert lufs == pytest.approx(-23.0, abs=0.15)


@pytest.mark.unit
def test_stereo_sine_sums_plus_3_db():
    lufs = bs1770.integrated_loudness(sine(997.0, 3.0, -20.0, channels=2), RATE)
    assert lufs == pytest.approx(-20.0, abs=0.15)


@pytest.mark.unit
def test_silence_is_negative_infinity():
    assert bs1770.integrated_loudness(np.zeros(RATE * 2), RATE) == float("-inf")


@pytest.mark.unit
def test_too_short_is_negative_infinity():
    assert bs1770.integrated_loudness(np.ones(100) * 0.1, RATE) == float("-inf")


@pytest.mark.unit
def test_relative_gate_ignores_quiet_tail():
    loud = sine(997.0, 3.0, -20.0)
    quiet = sine(997.0, 3.0, -60.0)  # above absolute gate, below relative gate
    lufs = bs1770.integrated_loudness(np.concatenate([loud, quiet]), RATE)
    assert lufs == pytest.approx(-23.0, abs=0.35)


@pytest.mark.unit
def test_fft_fallback_matches_scipy_path(monkeypatch):
    data = np.concatenate([sine(997.0, 1.0, -20.0), sine(60.0, 2.0, -12.0), np.zeros(RATE)])
    data = np.column_stack([data, data * 0.5])
    fast = bs1770.integrated_loudness(data, RATE)
    monkeypatch.setattr(bs1770, "_lfilter", None)
    assert not bs1770.has_fast_filter()
    slow = bs1770.integrated_loudness(data, RATE)
    assert slow == pytest.approx(fast, abs=1e-6)


@pytest.mark.unit
def test_fft_filter_matches_direct_convolution():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    h = rng.standard_normal(37)
    assert np.allclose(bs1770.fft_filter(x, h), np.convolve(x, h)[: len(x)], atol=1e-9)
    assert bs1770.fft_filter(np.zeros(0), h).shape == (0,)


@pytest.mark.unit
def test_float32_input_is_accepted():
    data = sine(997.0, 3.0, -20.0).astype(np.float32)
    assert bs1770.integrated_loudness(data, RATE) == pytest.approx(-23.0, abs=0.15)
    assert bs1770.apply_gain(data, 6.0).dtype == np.float32


@pytest.mark.unit
def test_apply_gain_returns_new_array():
    data = np.ones(10)
    out = bs1770.apply_gain(data, 6.0)
    assert out[0] == pytest.approx(10 ** (6 / 20))
    assert data[0] == 1.0
    assert out is not data


@pytest.mark.unit
def test_gain_for_target():
    assert bs1770.gain_db_for_target(-23.0, -16.0) == pytest.approx(7.0)


@pytest.mark.unit
def test_too_many_channels_rejected():
    with pytest.raises(ValueError):
        bs1770.integrated_loudness(np.zeros((RATE, 6)), RATE)
