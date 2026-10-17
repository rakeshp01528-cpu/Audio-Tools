"""True-peak metering and a lookahead brickwall limiter, in numpy.

True peak follows ITU-R BS.1770-4 Annex 2: the signal is 4x oversampled with
a polyphase windowed-sinc FIR so peaks that fall *between* samples (which a
DAC or a lossy encoder will reconstruct) are seen. Plain `max(abs(x))` and
linear interpolation both miss them.

The limiter is a linked (all channels share one gain) brickwall limiter:
  1. per-sample gain needed to keep the oversampled peak under the ceiling
  2. lookahead: gain reduction starts `lookahead_ms` before the peak
  3. release: gain recovers linearly in dB at `release_db_per_second`
  4. attack smoothing: a short moving average so the gain ramps instead of
     stepping
Every stage is vectorised (no per-sample Python loop) so a 5 minute stereo
file limits in well under a second.
"""

from __future__ import annotations

import numpy as np

OVERSAMPLE = 4
FIR_TAPS = 65  # odd, so the prototype is centred on a sample and phase k lands exactly at k/4
KAISER_BETA = 7.0
DEFAULT_LOOKAHEAD_MS = 5.0
DEFAULT_RELEASE_DB_PER_SECOND = 60.0  # 6 dB of recovery per 100 ms
MAX_CEILING_DBFS = 0.0
# Gain varies across the interpolation window, so the limited signal's true
# peak can land a hair above the target; aim this far under the ceiling.
SAFETY_MARGIN_DB = 0.01
ENVELOPE_CHUNK = 1 << 20  # samples per convolution chunk; keeps transient memory small


def _polyphase_filters(oversample: int = OVERSAMPLE, taps: int = FIR_TAPS) -> list[tuple[np.ndarray, int]]:
    """One (FIR, delay) per fractional phase k/os. Each FIR has unity DC gain.

    With an odd `taps` the prototype centre sits on an integer, so sub-filter k
    evaluates the signal exactly k/os samples after the sample it is aligned
    to once its own group delay is removed.
    """
    centre = (taps - 1) // 2
    t = np.arange(taps)
    prototype = np.sinc((t - centre) / oversample) * np.kaiser(taps, KAISER_BETA)
    phases = []
    for k in range(1, oversample):
        h = prototype[k::oversample]
        # Slice start that aligns sub-filter output i with time i + k/os (verified in tests).
        delay = (centre - k) // oversample + 1
        phases.append((h / h.sum(), delay))
    return phases


_PHASES = _polyphase_filters()


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    samples = np.asarray(data)
    if samples.dtype.kind != "f":
        samples = samples.astype(np.float32)
    return samples[:, None] if samples.ndim == 1 else samples


def peak_envelope(data: np.ndarray) -> np.ndarray:
    """Per-sample true-peak magnitude, max across channels, shape (n,).

    Element i is the largest reconstructed magnitude between sample i and
    sample i+1 (the sample itself plus the points 1/4, 1/2 and 3/4 after it).
    """
    samples = _ensure_2d(data)
    n = samples.shape[0]
    if n == 0:
        return np.zeros(0)
    envelope = np.max(np.abs(samples), axis=1).astype(np.float32)
    for ch in range(samples.shape[1]):
        for h, delay in _PHASES:
            _max_interpolated_into(envelope, samples[:, ch], h, delay)
    return envelope


def _max_interpolated_into(envelope: np.ndarray, channel: np.ndarray, h: np.ndarray, delay: int) -> None:
    """envelope[i] = max(envelope[i], |interp(channel)[i]|), convolving in chunks."""
    n = len(channel)
    taps = len(h)
    for start in range(0, n, ENVELOPE_CHUNK):
        stop = min(start + ENVELOPE_CHUNK, n)
        lo = max(0, start - taps)
        segment = np.asarray(channel[lo : stop + taps], dtype=np.float64)
        # "full" output index j corresponds to sample lo + j - delay
        full = np.convolve(segment, h, mode="full")
        first = (start - lo) + delay
        piece = np.abs(full[first : first + (stop - start)]).astype(np.float32)
        np.maximum(envelope[start:stop], piece, out=envelope[start:stop])


def true_peak_dbfs(data: np.ndarray) -> float:
    """True peak in dBFS; -inf for silence or empty input."""
    samples = _ensure_2d(data)
    if samples.shape[0] == 0:
        return float("-inf")
    peak = float(np.max(peak_envelope(samples)))
    return 20 * np.log10(peak) if peak > 0 else float("-inf")


def sliding_min(x: np.ndarray, window: int) -> np.ndarray:
    """y[i] = min(x[i : i + window]) in O(n) (van Herk / Gil-Werman).

    Holds three full-length float64 arrays at its peak (padded copy, block
    prefix minima, block suffix minima); the result is written into the
    padded copy.
    """
    x = np.asarray(x)
    dtype = x.dtype if x.dtype.kind == "f" else np.float64
    if window <= 1:
        return np.array(x, dtype=dtype, copy=True)
    n = len(x)
    total = n + (-n) % window + window
    padded = np.full(total, np.inf, dtype=dtype)
    padded[:n] = x
    blocks = padded.reshape(-1, window)
    prefix = np.minimum.accumulate(blocks, axis=1).ravel()
    # Accumulating over the reversed view yields the per-block suffix minima,
    # read back through a reversed view, so no extra copy is made.
    suffix = np.minimum.accumulate(padded[::-1].reshape(-1, window), axis=1).ravel()[::-1]
    np.minimum(suffix[:n], prefix[window - 1 : window - 1 + n], out=padded[:n])
    return padded[:n]


def trailing_mean(x: np.ndarray, window: int, fill: float = 1.0) -> np.ndarray:
    """y[i] = mean(x[i - window + 1 : i + 1]), padding the past with `fill`."""
    if window <= 1:
        return np.array(x, dtype=np.float64, copy=True)
    n = len(x)
    cumsum = np.empty(n + window, dtype=np.float64)
    cumsum[0] = 0.0
    cumsum[1:window] = fill
    cumsum[window:] = x
    np.cumsum(cumsum, out=cumsum)
    out = cumsum[window:] - cumsum[:-window]
    out /= window
    return out


def release_envelope_db(gain_db: np.ndarray, db_per_sample: float) -> np.ndarray:
    """Lower envelope of `gain_db` that may only rise at `db_per_sample`.

    env[n] = min_{k<=n} (gain_db[k] + slope * (n - k)), clamped to <= 0 dB.
    Rewritten as slope*n + cummin(gain_db - slope*k) so it is one vector op.
    """
    ramp = np.arange(len(gain_db), dtype=np.float64)
    ramp *= db_per_sample
    out = np.array(gain_db, dtype=np.float64, copy=True)
    out -= ramp
    np.minimum.accumulate(out, out=out)
    out += ramp
    np.minimum(out, 0.0, out=out)
    return out


def _release_in_place(gain_db: np.ndarray, db_per_sample: float) -> np.ndarray:
    """Same as release_envelope_db but consumes `gain_db` (one temporary array)."""
    ramp = np.arange(len(gain_db), dtype=np.float64)
    ramp *= db_per_sample
    gain_db -= ramp
    np.minimum.accumulate(gain_db, out=gain_db)
    gain_db += ramp
    np.minimum(gain_db, 0.0, out=gain_db)
    return gain_db


def gain_curve(
    data: np.ndarray,
    rate: int,
    ceiling_dbfs: float,
    lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
    release_db_per_second: float = DEFAULT_RELEASE_DB_PER_SECOND,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """Linear gain (<= 1) to apply per sample so true peaks stay under the ceiling.

    Pass a precomputed `envelope` (from `peak_envelope`) to avoid measuring twice.
    Intermediate arrays are reused in place so only ~3 float64 copies are alive.
    """
    if ceiling_dbfs > MAX_CEILING_DBFS:
        raise ValueError(f"ceiling must be {MAX_CEILING_DBFS:g} dBFS or lower, got {ceiling_dbfs:g}")
    ceiling = 10 ** ((ceiling_dbfs - SAFETY_MARGIN_DB) / 20.0)
    env = peak_envelope(data) if envelope is None else envelope
    n = len(env)
    if n == 0 or float(env.max()) <= ceiling:
        return np.ones(n)

    env32 = np.asarray(env, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        needed = np.divide(np.float32(ceiling), env32, out=np.ones(n, dtype=np.float32), where=env32 > 0)
        np.minimum(needed, np.float32(1.0), out=needed)
    lookahead = max(1, int(round(lookahead_ms * rate / 1000.0)))
    gain = sliding_min(needed, lookahead + 1).astype(np.float64)  # float64 from here for the dB math
    del needed
    with np.errstate(divide="ignore"):
        np.log10(gain, out=gain)
    gain *= 20.0
    gain = _release_in_place(gain, release_db_per_second / rate)
    gain /= 20.0
    np.power(10.0, gain, out=gain)
    # Pad the past with the first value, not 1.0: the lookahead already covers
    # samples 0..L, so this keeps the smoothed gain under the needed gain there.
    return trailing_mean(gain, lookahead, fill=float(gain[0]))


def limit(
    data: np.ndarray,
    rate: int,
    ceiling_dbfs: float,
    lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
    release_db_per_second: float = DEFAULT_RELEASE_DB_PER_SECOND,
    envelope: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return (limited samples, max gain reduction in dB). Input is not modified."""
    samples = _ensure_2d(data)
    gain = gain_curve(samples, rate, ceiling_dbfs, lookahead_ms, release_db_per_second, envelope)
    reduction_db = float(-20 * np.log10(max(float(gain.min()), 1e-12)))
    if reduction_db <= 1e-9:
        return samples.copy(), 0.0
    return samples * gain[:, None].astype(samples.dtype, copy=False), reduction_db
