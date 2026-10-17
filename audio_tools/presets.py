"""Common delivery loudness targets (integrated LUFS)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    lufs: float


_PRESETS = (
    Preset("podcast", "Podcast / Apple Music", -16.0),
    Preset("spotify", "Spotify", -14.0),
    Preset("youtube", "YouTube", -14.0),
    Preset("apple_music", "Apple Music", -16.0),
    Preset("broadcast_ebu", "Broadcast (EBU R128)", -23.0),
    Preset("broadcast_atsc", "Broadcast (ATSC A/85)", -24.0),
)

PRESETS: dict[str, Preset] = {p.key: p for p in _PRESETS}
LOUDNESS_PRESETS: dict[str, float] = {p.key: p.lufs for p in _PRESETS}
DEFAULT_PRESET = "podcast"

OUTPUT_FORMATS: tuple[str, ...] = ("mp3", "wav", "flac", "m4a", "ogg", "opus", "aiff")
LOSSY_FORMATS: frozenset[str] = frozenset({"mp3", "m4a", "ogg", "opus"})
DEFAULT_BITRATE = "320k"
DEFAULT_TRUE_PEAK_CEILING_DB = -1.0
