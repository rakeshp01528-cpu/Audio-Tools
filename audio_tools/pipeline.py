"""Normalization pipeline: decode, measure, gain, true-peak limit, encode.

Two modes:
  loudness  gain to a target integrated loudness (LUFS), then a lookahead
            brickwall limiter keeps true peaks under `true_peak_ceiling_db`
            (anywhere up to 0 dBFS) so the target is actually reached.
  peak      gain so the loudest true peak lands exactly on `target_peak_dbfs`
            (e.g. -1 or 0 dBFS). No loudness target, no limiting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_tools import bs1770, ffmpeg, limiter
from audio_tools.presets import (
    DEFAULT_BITRATE,
    DEFAULT_PRESET,
    DEFAULT_TRUE_PEAK_CEILING_DB,
    LOUDNESS_PRESETS,
)

OUTPUT_SUFFIX = "_normalized"
MODES = ("loudness", "peak")


@dataclass(frozen=True)
class NormalizeOptions:
    mode: str = "loudness"
    target_lufs: float = LOUDNESS_PRESETS[DEFAULT_PRESET]
    true_peak_ceiling_db: float = DEFAULT_TRUE_PEAK_CEILING_DB
    target_peak_dbfs: float = -1.0
    use_limiter: bool = True  # False: turn the whole file down instead (old behaviour)
    lookahead_ms: float = limiter.DEFAULT_LOOKAHEAD_MS
    release_db_per_second: float = limiter.DEFAULT_RELEASE_DB_PER_SECOND
    output_format: str = "mp3"
    bitrate: str = DEFAULT_BITRATE

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for name in ("true_peak_ceiling_db", "target_peak_dbfs"):
            value = getattr(self, name)
            if value > limiter.MAX_CEILING_DBFS:
                raise ValueError(f"{name} must be {limiter.MAX_CEILING_DBFS:g} dBFS or lower, got {value:g}")

    @classmethod
    def peak(cls, target_peak_dbfs: float, **kwargs) -> "NormalizeOptions":
        return cls(mode="peak", target_peak_dbfs=target_peak_dbfs, **kwargs)


@dataclass(frozen=True)
class Measurement:
    path: str
    integrated_lufs: float
    true_peak_dbfs: float
    sample_rate: int
    channels: int
    duration_seconds: float


@dataclass(frozen=True)
class NormalizeResult:
    input_path: str
    output_path: str
    mode: str
    measured_lufs: float
    achieved_lufs: float
    target_lufs: float | None
    gain_db: float
    true_peak_before_db: float
    true_peak_after_db: float
    limiter_reduction_db: float


def true_peak_dbfs(data: np.ndarray) -> float:
    """BS.1770-4 style 4x-oversampled true peak in dBFS."""
    return limiter.true_peak_dbfs(data)


def measure_file(path: str | os.PathLike) -> Measurement:
    data, rate = ffmpeg.decode(path)
    return Measurement(
        path=os.fspath(path),
        integrated_lufs=bs1770.integrated_loudness(data, rate),
        true_peak_dbfs=true_peak_dbfs(data),
        sample_rate=rate,
        channels=data.shape[1],
        duration_seconds=data.shape[0] / rate,
    )


def same_file(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    """True when both paths name the same file (case-insensitively on case-folding filesystems)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return Path(a).resolve() == Path(b).resolve()


def default_output_path(input_path: str | os.PathLike, fmt: str, output_dir: str | os.PathLike | None = None) -> Path:
    src = Path(input_path)
    directory = Path(output_dir) if output_dir else src.parent
    return directory / f"{src.stem}{OUTPUT_SUFFIX}.{fmt.lower().lstrip('.')}"


def _normalize_peak(
    data: np.ndarray, measured_lufs: float, peak_before: float, options: NormalizeOptions
) -> tuple[np.ndarray, dict]:
    if not np.isfinite(peak_before):
        raise ValueError("cannot peak-normalize silence")
    gain_db = options.target_peak_dbfs - peak_before
    out = bs1770.apply_gain(data, gain_db)
    return out, {
        "gain_db": gain_db,
        "achieved_lufs": measured_lufs + gain_db,
        "true_peak_after_db": options.target_peak_dbfs,
        "limiter_reduction_db": 0.0,
    }


def _normalize_loudness(
    data: np.ndarray, rate: int, measured_lufs: float, options: NormalizeOptions
) -> tuple[np.ndarray, dict]:
    if not np.isfinite(measured_lufs):
        raise ValueError("could not measure loudness: the audio is silent or shorter than 400 ms")
    gain_db = bs1770.gain_db_for_target(measured_lufs, options.target_lufs)
    gained = bs1770.apply_gain(data, gain_db)
    envelope = limiter.peak_envelope(gained)
    peak_lin = float(envelope.max()) if envelope.size else 0.0
    peak_gained = 20 * np.log10(peak_lin) if peak_lin > 0 else float("-inf")
    over_db = max(0.0, peak_gained - options.true_peak_ceiling_db)

    if over_db == 0.0:
        return gained, {
            "gain_db": gain_db,
            "achieved_lufs": options.target_lufs,
            "true_peak_after_db": peak_gained,
            "limiter_reduction_db": 0.0,
        }
    if not options.use_limiter:
        out = bs1770.apply_gain(gained, -over_db)
        return out, {
            "gain_db": gain_db - over_db,
            "achieved_lufs": options.target_lufs - over_db,
            "true_peak_after_db": options.true_peak_ceiling_db,
            "limiter_reduction_db": 0.0,
        }
    out, reduction_db = limiter.limit(
        gained,
        rate,
        options.true_peak_ceiling_db,
        options.lookahead_ms,
        options.release_db_per_second,
        envelope=envelope,
    )
    del gained, envelope
    return out, {
        "gain_db": gain_db,
        "achieved_lufs": bs1770.integrated_loudness(out, rate),
        "true_peak_after_db": true_peak_dbfs(out),
        "limiter_reduction_db": reduction_db,
    }


def normalize_samples(data: np.ndarray, rate: int, options: NormalizeOptions) -> tuple[np.ndarray, dict]:
    """Pure in-memory step shared by file and batch paths. Returns (samples, stats)."""
    measured = bs1770.integrated_loudness(data, rate)
    peak_before = true_peak_dbfs(data)
    if options.mode == "peak":
        out, stats = _normalize_peak(data, measured, peak_before, options)
    else:
        out, stats = _normalize_loudness(data, rate, measured, options)
    return out, {
        "mode": options.mode,
        "measured_lufs": measured,
        "target_lufs": options.target_lufs if options.mode == "loudness" else None,
        "true_peak_before_db": peak_before,
        **stats,
    }


def normalize_file(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    options: NormalizeOptions | None = None,
) -> NormalizeResult:
    """Decode, normalize (loudness or peak), and encode one file."""
    options = options or NormalizeOptions()
    input_path = os.fspath(input_path)
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    out = Path(output_path) if output_path else default_output_path(input_path, options.output_format)
    if same_file(out, input_path):
        raise ValueError("output path must differ from the input path")

    data, rate = ffmpeg.decode(input_path)
    samples, stats = normalize_samples(data, rate, options)
    del data
    ffmpeg.encode(samples, rate, out, options.output_format, options.bitrate)
    return NormalizeResult(input_path=input_path, output_path=str(out), **stats)


__all__ = [
    "MODES",
    "Measurement",
    "NormalizeOptions",
    "NormalizeResult",
    "default_output_path",
    "measure_file",
    "normalize_file",
    "normalize_samples",
    "true_peak_dbfs",
]
