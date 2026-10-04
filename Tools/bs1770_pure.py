"""Pure-numpy ITU-R BS.1770 loudness meter -- no scipy dependency."""
import numpy as np


def _biquad_coeffs(G, Q, fc, rate, kind):
    A = 10 ** (G / 40.0)
    w0 = 2.0 * np.pi * (fc / rate)
    alpha = np.sin(w0) / (2.0 * Q)
    cw0 = np.cos(w0)
    if kind == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cw0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw0)
        b2 = A * ((A + 1) + (A - 1) * cw0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cw0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw0)
        a2 = (A + 1) - (A - 1) * cw0 - 2 * np.sqrt(A) * alpha
    elif kind == "high_pass":
        b0 = (1 + cw0) / 2
        b1 = -(1 + cw0)
        b2 = (1 + cw0) / 2
        a0 = 1 + alpha
        a1 = -2 * cw0
        a2 = 1 - alpha
    else:
        raise ValueError(kind)
    return (b0 / a0, b1 / a0, b2 / a0), (1.0, a1 / a0, a2 / a0)


def _apply_biquad(x, b, a):
    """Direct-form I biquad, implemented as a plain loop (no scipy)."""
    b0, b1, b2 = b
    _, a1, a2 = a
    xl = x.tolist()
    y = [0.0] * len(xl)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(len(xl)):
        xn = xl[n]
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return np.array(y, dtype=np.float64)


def _k_weight(data, rate):
    b1, a1 = _biquad_coeffs(4.0, 1 / np.sqrt(2), 1500.0, rate, "high_shelf")
    b2, a2 = _biquad_coeffs(0.0, 0.5, 38.0, rate, "high_pass")
    stage1 = _apply_biquad(data, b1, a1)
    stage2 = _apply_biquad(stage1, b2, a2)
    return stage2


def integrated_loudness(data, rate):
    """BS.1770-4 integrated (gated) loudness in LUFS. data: (samples,) or (samples, ch)."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    num_channels = data.shape[1]
    num_samples = data.shape[0]

    filtered = np.zeros_like(data, dtype=np.float64)
    for ch in range(num_channels):
        filtered[:, ch] = _k_weight(data[:, ch].astype(np.float64), rate)

    G = [1.0, 1.0, 1.0, 1.41, 1.41][:num_channels]
    T_g = 0.400
    overlap = 0.75
    step = 1.0 - overlap
    Gamma_a = -70.0

    T = num_samples / rate
    num_blocks = int(np.round((T - T_g) / (T_g * step))) + 1
    z = np.zeros((num_channels, num_blocks))

    for i in range(num_channels):
        for j in range(num_blocks):
            lo = int(T_g * (j * step) * rate)
            hi = int(T_g * (j * step + 1) * rate)
            z[i, j] = (1.0 / (T_g * rate)) * np.sum(filtered[lo:hi, i] ** 2)

    with np.errstate(divide="ignore"):
        l = [-0.691 + 10.0 * np.log10(np.sum([G[i] * z[i, j] for i in range(num_channels)]))
             for j in range(num_blocks)]

    J_g = [j for j, lj in enumerate(l) if lj >= Gamma_a]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_avg = [np.mean([z[i, j] for j in J_g]) for i in range(num_channels)]
        Gamma_r = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg[i] for i in range(num_channels)])) - 10.0

    J_g = [j for j, lj in enumerate(l) if lj > Gamma_r and lj > Gamma_a]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_avg = np.nan_to_num([np.mean([z[i, j] for j in J_g]) for i in range(num_channels)])

    with np.errstate(divide="ignore"):
        LUFS = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg[i] for i in range(num_channels)]))
    return LUFS


def normalize_to_lufs(data, current_lufs, target_lufs):
    gain_db = target_lufs - current_lufs
    return data * (10 ** (gain_db / 20.0))