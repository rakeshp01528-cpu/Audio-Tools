"""Environment health check: Python, numpy, scipy (optional), ffmpeg, ffprobe."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from audio_tools import ffmpeg

MIN_PYTHON = (3, 10)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _check_python() -> Check:
    version = ".".join(str(v) for v in sys.version_info[:3])
    ok = sys.version_info >= MIN_PYTHON
    return Check("Python", ok, f"{version} ({sys.executable})")


def _check_numpy() -> Check:
    try:
        import numpy

        return Check("numpy", True, numpy.__version__)
    except Exception as exc:  # noqa: BLE001
        return Check("numpy", False, f"import failed: {exc}")


def _check_scipy() -> Check:
    try:
        from audio_tools import bs1770
    except Exception as exc:  # noqa: BLE001 - numpy itself may be broken
        return Check("scipy (optional)", False, f"skipped: {exc}", required=False)
    if bs1770.has_fast_filter():
        import scipy

        return Check("scipy (optional)", True, f"{scipy.__version__} - fast filtering on", required=False)
    return Check(
        "scipy (optional)", False, "not importable - using the numpy FFT filter instead (just as good)", required=False
    )


def _check_tool(name: str, required: bool) -> Check:
    version = ffmpeg.tool_version(name)
    if version is None:
        return Check(name, False, "not found", required=required)
    return Check(name, True, version, required=required)


def run_checks() -> list[Check]:
    return [
        _check_python(),
        _check_numpy(),
        _check_scipy(),
        _check_tool("ffmpeg", required=True),
        _check_tool("ffprobe", required=False),
    ]


def all_required_ok(checks: list[Check]) -> bool:
    return all(c.ok for c in checks if c.required)


def format_report(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "OK " if c.ok else ("!! " if c.required else "-- ")
        lines.append(f"{mark} {c.name:<18} {c.detail}")
    lines.append("")
    lines.append(
        "Everything needed is in place."
        if all_required_ok(checks)
        else "Something required is missing. Re-run the setup script, then try again."
    )
    return "\n".join(lines)
