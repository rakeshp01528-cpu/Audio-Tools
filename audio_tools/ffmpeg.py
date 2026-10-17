"""Locate and drive ffmpeg/ffprobe for decoding and encoding.

ffmpeg is the only external program this package needs. It is looked up in
this order:
  1. the AUDIO_TOOLS_FFMPEG_DIR environment variable
  2. the repo-local ./bin folder (where the setup scripts drop a download)
  3. the winget shim folder (Windows only)
  4. PATH

numpy is imported lazily so `audio-tools doctor` can still run and explain
the problem on a machine where numpy itself fails to import.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from audio_tools.presets import LOSSY_FORMATS

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

ENV_FFMPEG_DIR = "AUDIO_TOOLS_FFMPEG_DIR"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BIN_DIR = REPO_ROOT / "bin"
EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
MAX_CHANNELS = 2  # more than stereo is downmixed on decode; lossy encoders can't take it anyway
SAMPLE_DTYPE = "<f4"

ENCODERS: dict[str, str] = {
    "mp3": "libmp3lame",
    "m4a": "aac",
    "aac": "aac",
    "ogg": "libvorbis",
    "opus": "libopus",
    "flac": "flac",
    "wav": "pcm_s16le",
    "aiff": "pcm_s16be",
}
CONTAINER_FOR_FORMAT: dict[str, str] = {"m4a": "ipod", "opus": "opus", "aac": "adts"}


class FFmpegNotFound(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot be located."""


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe exits with an error."""


@dataclass(frozen=True)
class AudioInfo:
    channels: int
    sample_rate: int
    duration_seconds: float


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_dir = os.environ.get(ENV_FFMPEG_DIR)
    if env_dir:
        dirs.append(Path(env_dir))
    if LOCAL_BIN_DIR.is_dir():
        dirs.append(LOCAL_BIN_DIR)
        # Extracted release zips keep binaries under <build>/bin/.
        dirs.extend(sorted(p for p in LOCAL_BIN_DIR.glob("*/bin") if p.is_dir()))
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        # winget puts shims here; a console opened before the install won't have it on PATH yet
        dirs.append(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Links")
    return dirs


def find_tool(name: str) -> str:
    """Return the full path to `ffmpeg` or `ffprobe`, or raise FFmpegNotFound."""
    for directory in _candidate_dirs():
        candidate = directory / f"{name}{EXE_SUFFIX}"
        if candidate.is_file():
            return str(candidate)
    on_path = shutil.which(name)
    if on_path:
        return on_path
    raise FFmpegNotFound(
        f"{name} was not found. Run the setup script (setup.sh / setup.bat) to "
        f"install it, or set {ENV_FFMPEG_DIR} to the folder that contains it."
    )


def find_ffmpeg() -> str:
    return find_tool("ffmpeg")


def find_ffprobe() -> str:
    return find_tool("ffprobe")


def tool_version(name: str) -> str | None:
    try:
        exe = find_tool(name)
        result = _run([exe, "-version"], check=False)
    except (FFmpegNotFound, FFmpegError):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.decode("utf-8", "replace").splitlines()[:1]
    return first_line[0] if first_line else exe


def _run(cmd: list[str], *, stdin: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # e.g. blocked executable, permission denied
        raise FFmpegError(f"could not start {cmd[0]}: {exc}") from exc
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise FFmpegError(f"{Path(cmd[0]).name} failed ({result.returncode}): {stderr}")
    return result


def _safe_path(path: str | os.PathLike) -> str:
    """Absolute path, so names starting with '-' or containing ':' are never read as options."""
    return os.path.abspath(os.fspath(path))


def probe(path: str | os.PathLike) -> AudioInfo:
    """Read channel count, sample rate and duration of the first audio stream."""
    path = _safe_path(path)
    try:
        ffprobe = find_ffprobe()
    except FFmpegNotFound:
        return _probe_with_ffmpeg(path)
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate:format=duration",
            "-of",
            "json",
            path,
        ]
    )
    info = json.loads(result.stdout or b"{}")
    streams = info.get("streams") or []
    if not streams:
        raise FFmpegError(f"no audio stream found in '{path}'")
    stream = streams[0]
    duration = float((info.get("format") or {}).get("duration") or 0.0)
    return AudioInfo(int(stream["channels"]), int(stream["sample_rate"]), duration)


_STREAM_RE = re.compile(r"Audio:.*?(\d+) Hz,\s*([^,]+)")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_LAYOUT_CHANNELS = {"mono": 1, "stereo": 2}


def _probe_with_ffmpeg(path: str) -> AudioInfo:
    """Fallback for builds that ship ffmpeg without ffprobe."""
    result = _run([find_ffmpeg(), "-hide_banner", "-nostdin", "-i", path], check=False)
    text = result.stderr.decode("utf-8", "replace")
    stream = _STREAM_RE.search(text)
    if not stream:
        raise FFmpegError(f"could not read stream info for '{path}'")
    layout = stream.group(2).strip()
    channel_match = re.match(r"(\d+)\s*channels?", layout)
    channels = int(channel_match.group(1)) if channel_match else _LAYOUT_CHANNELS.get(layout, 2)
    duration = 0.0
    dur = _DURATION_RE.search(text)
    if dur:
        duration = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))
    return AudioInfo(channels, int(stream.group(1)), duration)


def decode(path: str | os.PathLike) -> tuple["np.ndarray", int]:
    """Decode any ffmpeg-readable file to float32 samples shaped (n, channels).

    Anything with more than two channels is downmixed to stereo.
    """
    import numpy as np

    path = _safe_path(path)
    info = probe(path)
    channels = min(info.channels, MAX_CHANNELS)
    # Decode to a temp file and read it straight into one array: piping through
    # stdout would hold two copies of the raw audio in memory at once.
    fd, raw_name = tempfile.mkstemp(prefix="audio-tools-", suffix=".f32")
    os.close(fd)
    try:
        _run(
            [
                find_ffmpeg(),
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                path,
                "-vn",
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                str(channels),
                "-ar",
                str(info.sample_rate),
                raw_name,
            ]
        )
        samples = np.fromfile(raw_name, dtype=SAMPLE_DTYPE)
    finally:
        if os.path.exists(raw_name):
            os.unlink(raw_name)
    if samples.size == 0:
        raise FFmpegError(f"decoded zero samples from '{path}'")
    return samples.reshape(-1, channels), info.sample_rate


def encode(
    data: "np.ndarray",
    rate: int,
    output_path: str | os.PathLike,
    fmt: str,
    bitrate: str | None = None,
) -> None:
    """Encode float samples shaped (n, channels) to `output_path` in `fmt`.

    ffmpeg writes to a temporary file in the same folder which is renamed into
    place at the end, so an interrupted run never leaves a truncated file
    under the final name.
    """
    import numpy as np

    fmt = fmt.lower().lstrip(".")
    if fmt not in ENCODERS:
        raise ValueError(f"unsupported output format '{fmt}'. Choose from: {', '.join(ENCODERS)}")
    samples = np.asarray(data)
    if samples.ndim == 1:
        samples = samples[:, None]
    channels = samples.shape[1]
    output_path = Path(_safe_path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if samples.max() > 1.0 or samples.min() < -1.0:
        samples = np.clip(samples, -1.0, 1.0)
    pcm = np.ascontiguousarray(samples, dtype=SAMPLE_DTYPE).tobytes()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output_path.stem}.", suffix=f".part.{fmt}", dir=output_path.parent)
    os.close(fd)
    cmd = [
        find_ffmpeg(),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(rate),
        "-ac",
        str(channels),
        "-i",
        "-",
        "-c:a",
        ENCODERS[fmt],
    ]
    if fmt in LOSSY_FORMATS and bitrate:
        cmd += ["-b:a", bitrate]
    cmd += ["-f", CONTAINER_FOR_FORMAT.get(fmt, fmt), tmp_name]
    try:
        _run(cmd, stdin=pcm)
        os.replace(tmp_name, output_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
