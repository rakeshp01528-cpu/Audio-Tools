"""ITU-R BS.1770-4 integrated loudness meter built on numpy.

The K-weighting filters run through scipy.signal.lfilter when scipy imports
cleanly. Otherwise the two biquads are turned into one finite impulse
response and applied by FFT overlap-add, which is pure numpy and about as
fast. That fallback matters on locked-down Windows machines where
application control (WDAC / Smart App Control) blocks one of scipy's DLLs.

Memory is kept to a few float64 copies of ONE channel at a time, so long
stereo files do not need gigabytes.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on the host machine
    from scipy.signal import lfilter as _lfilter
except Exception:  # noqa: BLE001 - any import failure means "no scipy"
    _lfilter = None

CHANNEL_WEIGHTS = (1.0, 1.0, 1.0, 1.41, 1.41)
BLOCK_SECONDS = 0.400
BLOCK_OVERLAP = 0.75
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_DB = -10.0
LOUDNESS_OFFSET_DB = -0.691
FFT_BLOCK = 1 << 17

Coeffs = tuple[tuple[float, float, float], tuple[float, float, float]]


def has_fast_filter() -> bool:
    """True when scipy is available for the K-weighting stage."""
    return _lfilter is not None


def _biquad_coeffs(gain_db: float, q: float, fc: float, rate: int, kind: str) -> Coeffs:
    a_lin = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (fc / rate)
    alpha = np.sin(w0) / (2.0 * q)
    cw0 = np.cos(w0)
    sqrt_a = np.sqrt(a_lin)
    if kind == "high_shelf":
        b0 = a_lin * ((a_lin + 1) + (a_lin - 1) * cw0 + 2 * sqrt_a * alpha)
        b1 = -2 * a_lin * ((a_lin - 1) + (a_lin + 1) * cw0)
        b2 = a_lin * ((a_lin + 1) + (a_lin - 1) * cw0 - 2 * sqrt_a * alpha)
        a0 = (a_lin + 1) - (a_lin - 1) * cw0 + 2 * sqrt_a * alpha
        a1 = 2 * ((a_lin - 1) - (a_lin + 1) * cw0)
        a2 = (a_lin + 1) - (a_lin - 1) * cw0 - 2 * sqrt_a * alpha
    elif kind == "high_pass":
        b0 = (1 + cw0) / 2
        b1 = -(1 + cw0)
        b2 = (1 + cw0) / 2
        a0 = 1 + alpha
        a1 = -2 * cw0
        a2 = 1 - alpha
    else:
        raise ValueError(f"unknown biquad kind: {kind}")
    return (b0 / a0, b1 / a0, b2 / a0), (1.0, a1 / a0, a2 / a0)


def _k_coeffs(rate: int) -> tuple[Coeffs, Coeffs]:
    shelf = _biquad_coeffs(4.0, 1 / np.sqrt(2), 1500.0, rate, "high_shelf")
    high_pass = _biquad_coeffs(0.0, 0.5, 38.0, rate, "high_pass")
    return shelf, high_pass


def _biquad_python(x: np.ndarray, b: tuple, a: tuple) -> np.ndarray:
    """Direct-form I biquad as a plain loop. Only used on short impulse responses."""
    b0, b1, b2 = b
    _, a1, a2 = a
    out = np.empty(len(x), dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for n, xn in enumerate(x.tolist()):
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return out


def _impulse_response_length(rate: int) -> int:
    # The 38 Hz high-pass pole is the slowest; half a second of tail is below 1e-9.
    return 1 << int(np.ceil(np.log2(max(rate // 2, 1024))))


def _k_impulse_response(rate: int) -> np.ndarray:
    length = _impulse_response_length(rate)
    impulse = np.zeros(length)
    impulse[0] = 1.0
    (sb, sa), (hb, ha) = _k_coeffs(rate)
    return _biquad_python(_biquad_python(impulse, sb, sa), hb, ha)


def fft_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal FIR filtering by FFT overlap-add; returns len(x) samples."""
    n, m = len(x), len(h)
    if n == 0:
        return np.zeros(0)
    block = max(FFT_BLOCK, 1 << int(np.ceil(np.log2(m))))
    fft_size = 1 << int(np.ceil(np.log2(block + m - 1)))
    spectrum = np.fft.rfft(h, fft_size)
    out = np.zeros(n + fft_size)
    for start in range(0, n, block):
        chunk = x[start : start + block]
        out[start : start + fft_size] += np.fft.irfft(np.fft.rfft(chunk, fft_size) * spectrum, fft_size)
    return out[:n]


_IR_CACHE: dict[int, np.ndarray] = {}


def k_weight(channel: np.ndarray, rate: int) -> np.ndarray:
    """Apply the BS.1770 K-weighting (shelf + high-pass) to one channel."""
    x = np.asarray(channel, dtype=np.float64)
    if _lfilter is not None:
        (sb, sa), (hb, ha) = _k_coeffs(rate)
        shelved = np.asarray(_lfilter(sb, sa, x), dtype=np.float64)
        del x
        return np.asarray(_lfilter(hb, ha, shelved), dtype=np.float64)
    if rate not in _IR_CACHE:
        _IR_CACHE[rate] = _k_impulse_response(rate)
    return fft_filter(x, _IR_CACHE[rate])


def _block_energies(filtered: np.ndarray, rate: int) -> np.ndarray:
    """Mean-square energy of each 400 ms block (75% overlap) for one channel."""
    num_samples = len(filtered)
    block_len = int(round(BLOCK_SECONDS * rate))
    step_len = int(round(BLOCK_SECONDS * (1.0 - BLOCK_OVERLAP) * rate))
    if num_samples < block_len or step_len <= 0:
        return np.zeros(0)
    num_blocks = (num_samples - block_len) // step_len + 1
    squared = np.square(filtered)  # fresh array; the caller's `filtered` is a temporary anyway
    cumsum = np.empty(num_samples + 1)
    cumsum[0] = 0.0
    np.cumsum(squared, out=cumsum[1:])
    del squared
    starts = np.arange(num_blocks) * step_len
    return (cumsum[starts + block_len] - cumsum[starts]) / block_len


def _block_loudness(energies: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = np.sum(energies * weights[:, None], axis=0)
    with np.errstate(divide="ignore"):
        return LOUDNESS_OFFSET_DB + 10.0 * np.log10(weighted)


def integrated_loudness(data: np.ndarray, rate: int) -> float:
    """Gated integrated loudness in LUFS. Returns -inf for silence/too short.

    `data` is shape (samples,) or (samples, channels), float, full scale 1.0.
    """
    samples = np.asarray(data)
    if samples.ndim == 1:
        samples = samples[:, None]
    num_channels = samples.shape[1]
    if num_channels > len(CHANNEL_WEIGHTS):
        raise ValueError(f"BS.1770 supports up to {len(CHANNEL_WEIGHTS)} channels")

    per_channel = [_block_energies(k_weight(samples[:, ch], rate), rate) for ch in range(num_channels)]
    energies = np.vstack(per_channel)
    if energies.shape[1] == 0:
        return float("-inf")

    weights = np.asarray(CHANNEL_WEIGHTS[:num_channels])
    block_lufs = _block_loudness(energies, weights)

    above_absolute = block_lufs >= ABSOLUTE_GATE_LUFS
    if not np.any(above_absolute):
        return float("-inf")
    relative_gate = (
        _block_loudness(energies[:, above_absolute].mean(axis=1, keepdims=True), weights)[0] + RELATIVE_GATE_DB
    )
    kept = above_absolute & (block_lufs > relative_gate)
    if not np.any(kept):
        return float("-inf")
    return float(_block_loudness(energies[:, kept].mean(axis=1, keepdims=True), weights)[0])


def gain_db_for_target(current_lufs: float, target_lufs: float) -> float:
    return target_lufs - current_lufs


def apply_gain(data: np.ndarray, gain_db: float) -> np.ndarray:
    """Return a new array scaled by `gain_db`, keeping the input's float dtype."""
    samples = np.asarray(data)
    dtype = samples.dtype if samples.dtype.kind == "f" else np.float32
    return samples * dtype.type(10 ** (gain_db / 20.0))
