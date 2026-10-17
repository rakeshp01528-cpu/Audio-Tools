"""Audio Tools: loudness-normalize and convert audio with ffmpeg + numpy.

The numpy-backed modules are imported lazily so `audio-tools doctor` can run
and explain the problem on a machine where numpy fails to import.
"""

from __future__ import annotations

__version__ = "1.1.0"

_LAZY = {
    "Measurement": "audio_tools.pipeline",
    "NormalizeOptions": "audio_tools.pipeline",
    "NormalizeResult": "audio_tools.pipeline",
    "measure_file": "audio_tools.pipeline",
    "normalize_file": "audio_tools.pipeline",
    "LOUDNESS_PRESETS": "audio_tools.presets",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'audio_tools' has no attribute {name!r}")
