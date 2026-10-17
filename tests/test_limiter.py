import numpy as np
import pytest

from audio_tools import limiter
from tests.conftest import RATE, sine


@pytest.mark.unit
def test_sliding_min_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.random(257)
    for window in (1, 2, 7, 64, 300):
        expected = np.array([x[i : i + window].min() for i in range(len(x))])
        assert np.allclose(limiter.sliding_min(x, window), expected)


@pytest.mark.unit
def test_trailing_mean_matches_brute_force():
    x = np.arange(20, dtype=float)
    out = limiter.trailing_mean(x, 4, fill=0.0)
    expected = [np.concatenate([np.zeros(3), x])[i : i + 4].mean() for i in range(20)]
    assert np.allclose(out, expected)


@pytest.mark.unit
def test_release_envelope_recovers_linearly():
    gain_db = np.zeros(100)
    gain_db[10] = -6.0
    env = limiter.release_envelope_db(gain_db, 0.5)
    assert env[10] == -6.0
    assert env[14] == pytest.approx(-4.0)
    assert env[22:].max() == 0.0
    assert env[:10].min() == 0.0


def faded(x: np.ndarray, samples: int = 2400) -> np.ndarray:
    """Raised-cosine fade in/out so an abrupt onset doesn't add real overshoot."""
    env = np.ones(len(x))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(samples) / samples)
    env[:samples] = ramp
    env[-samples:] = ramp[::-1]
    return x * env


@pytest.mark.unit
def test_true_peak_sees_inter_sample_peaks():
    # fs/4 sine sampled at 45 degrees: samples sit at -3.01 dBFS, real peak at 0 dBFS.
    t = np.arange(RATE)
    x = faded(np.sin(2 * np.pi * 0.25 * t + np.pi / 4))
    assert 20 * np.log10(np.max(np.abs(x))) == pytest.approx(-3.01, abs=0.01)
    assert limiter.true_peak_dbfs(x) == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_chunked_envelope_matches_single_pass(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5 * 4096 + 123, 2)).astype(np.float32)
    whole = limiter.peak_envelope(x)
    monkeypatch.setattr(limiter, "ENVELOPE_CHUNK", 4096)
    assert np.allclose(limiter.peak_envelope(x), whole, atol=1e-6)


@pytest.mark.unit
def test_true_peak_low_frequency_equals_sample_peak():
    x = sine(100.0, 1.0, -6.0)
    assert limiter.true_peak_dbfs(x) == pytest.approx(-6.0, abs=0.05)
    assert limiter.true_peak_dbfs(np.zeros(0)) == float("-inf")
    assert limiter.true_peak_dbfs(np.zeros(10)) == float("-inf")


@pytest.mark.unit
def test_gain_curve_is_identity_when_under_ceiling():
    x = sine(440.0, 0.5, -6.0)
    assert np.all(limiter.gain_curve(x, RATE, -1.0) == 1.0)
    out, reduction = limiter.limit(x, RATE, -1.0)
    assert reduction == 0.0
    assert np.array_equal(out[:, 0], x)
    assert out is not x


@pytest.mark.unit
def test_limit_steady_sine_to_ceiling():
    x = sine(440.0, 2.0, 0.0, channels=2)
    out, reduction = limiter.limit(x, RATE, -3.0)
    assert reduction == pytest.approx(3.0 + limiter.SAFETY_MARGIN_DB, abs=0.05)
    assert limiter.true_peak_dbfs(out) <= -3.0 + 1e-3
    # Steady state: constant gain, so the second half is just an attenuated copy.
    half = RATE
    assert np.allclose(out[half:], x[half:] * 10 ** (-(3 + limiter.SAFETY_MARGIN_DB) / 20), atol=2e-3)


@pytest.mark.unit
def test_limit_spike_recovers_and_leaves_rest_alone():
    x = sine(440.0, 2.0, -20.0)
    spike_at = RATE  # one 0 dBFS click in the middle
    x[spike_at : spike_at + 3] = 1.0
    out, reduction = limiter.limit(x, RATE, -6.0)
    # A 3-sample click has inter-sample overshoot, so a little more than 6 dB is right.
    assert 6.0 <= reduction <= 7.5
    assert limiter.true_peak_dbfs(out) <= -6.0 + 1e-3
    # Untouched well before the click, and fully recovered 200 ms after it.
    assert np.allclose(out[: spike_at - RATE // 10, 0], x[: spike_at - RATE // 10])
    assert np.allclose(out[spike_at + RATE // 5 :, 0], x[spike_at + RATE // 5 :], atol=1e-6)
    # Gain starts dropping before the click (lookahead), never after.
    gain = limiter.gain_curve(x, RATE, -6.0)
    assert gain[spike_at - 100] < 1.0
    assert gain[spike_at] == pytest.approx(gain.min(), abs=1e-6)


@pytest.mark.unit
def test_limit_allows_zero_dbfs_ceiling_and_rejects_above():
    x = sine(440.0, 1.0, 3.0)  # over full scale
    out, reduction = limiter.limit(x, RATE, 0.0)
    assert reduction == pytest.approx(3.0 + limiter.SAFETY_MARGIN_DB, abs=0.05)
    assert limiter.true_peak_dbfs(out) <= 1e-3
    with pytest.raises(ValueError, match="0 dBFS or lower"):
        limiter.limit(x, RATE, 0.5)


@pytest.mark.unit
def test_limiter_handles_peak_at_very_start():
    x = sine(440.0, 0.5, 0.0)
    x[:5] = 1.0
    out, _ = limiter.limit(x, RATE, -6.0)
    assert limiter.true_peak_dbfs(out) <= -6.0 + 1e-3


@pytest.mark.unit
def test_peak_envelope_short_input_keeps_length():
    assert limiter.peak_envelope(np.ones(3)).shape == (3,)
    assert limiter.peak_envelope(np.zeros((0, 2))).shape == (0,)


@pytest.mark.unit
def test_polyphase_filters_land_on_quarter_samples():
    n = 2000
    ramp = np.arange(n, dtype=float)
    for k, (h, delay) in enumerate(limiter._PHASES, 1):
        y = np.convolve(ramp, h, mode="full")[delay : delay + n]
        assert np.median(y[200:-200] - ramp[200:-200]) == pytest.approx(k / 4, abs=0.002)


@pytest.mark.unit
def test_true_peak_accuracy_across_band():
    t = np.arange(RATE)
    for freq in np.linspace(100, 20000, 25):
        x = faded(np.sin(2 * np.pi * freq / RATE * t + 0.4))
        assert -0.1 <= limiter.true_peak_dbfs(x) <= 0.01, freq


@pytest.mark.unit
def test_limiter_never_exceeds_ceiling_on_nasty_signals():
    rng = np.random.default_rng(42)
    t = np.arange(RATE)
    signals = [
        rng.standard_normal((RATE, 2)) * 0.5,
        np.sin(2 * np.pi * 0.25 * t + np.pi / 4) * 1.2,  # inter-sample peaks over full scale
        np.where(rng.random(RATE) < 0.001, 1.0, 0.0) * rng.choice([-1, 1], RATE),  # random clicks
        np.sin(2 * np.pi * 15000 / RATE * t) * np.sin(2 * np.pi * 7 / RATE * t) * 1.5,  # HF bursts
    ]
    for ceiling in (0.0, -1.0, -6.0):
        for x in signals:
            out, _ = limiter.limit(x, RATE, ceiling)
            assert limiter.true_peak_dbfs(out) <= ceiling + 1e-3


@pytest.mark.unit
def test_limit_keeps_float32():
    x = sine(440.0, 0.5, 0.0).astype(np.float32)
    out, _ = limiter.limit(x, RATE, -3.0)
    assert out.dtype == np.float32
