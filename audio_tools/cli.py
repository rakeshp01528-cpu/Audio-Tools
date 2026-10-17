"""Command-line entry point. `audio-tools` with no arguments opens the TUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from audio_tools import __version__
from audio_tools.ffmpeg import ENCODERS, FFmpegError, FFmpegNotFound
from audio_tools.presets import (
    DEFAULT_BITRATE,
    DEFAULT_TRUE_PEAK_CEILING_DB,
    LOSSY_FORMATS,
    LOUDNESS_PRESETS,
    OUTPUT_FORMATS,
)

if TYPE_CHECKING:  # pragma: no cover
    from audio_tools.pipeline import NormalizeOptions, NormalizeResult

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma"})
FOLDER_OUTPUT_NAME = "normalized"
LOSSY_PEAK_WARN_DB = -1.0
EXIT_OK, EXIT_FAILED, EXIT_SETUP, EXIT_CANCELLED = 0, 1, 2, 130
PER_FILE_ERRORS = (ValueError, FFmpegError, OSError, MemoryError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-tools",
        description="Loudness-normalize (BS.1770 / LUFS) and convert audio files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    norm = sub.add_parser("normalize", help="normalize one file or every audio file in a folder")
    norm.add_argument("input", help="audio file or folder")
    norm.add_argument("-o", "--output", help="output file (single input) or folder (folder input)")
    norm.add_argument("-f", "--format", choices=OUTPUT_FORMATS, help="output format (default: taken from -o, else mp3)")
    target = norm.add_mutually_exclusive_group()
    target.add_argument("-p", "--preset", choices=sorted(LOUDNESS_PRESETS), help="loudness preset")
    target.add_argument("-l", "--lufs", type=float, help="explicit target in LUFS, e.g. -16")
    target.add_argument(
        "-k",
        "--peak",
        type=float,
        metavar="DBFS",
        help="peak-normalize instead: put the loudest true peak at this level, e.g. -1 or 0",
    )
    norm.add_argument(
        "-c",
        "--ceiling",
        type=float,
        default=DEFAULT_TRUE_PEAK_CEILING_DB,
        help=f"true-peak ceiling in dBFS for loudness mode, up to 0 (default {DEFAULT_TRUE_PEAK_CEILING_DB})",
    )
    norm.add_argument(
        "--no-limiter",
        action="store_true",
        help="turn the whole file down to respect the ceiling instead of limiting peaks",
    )
    norm.add_argument(
        "-b", "--bitrate", default=DEFAULT_BITRATE, help=f"bitrate for lossy formats (default {DEFAULT_BITRATE})"
    )

    meas = sub.add_parser("measure", help="print integrated loudness and true peak")
    meas.add_argument("input", nargs="+", help="audio file(s)")

    sub.add_parser("doctor", help="check that Python, numpy and ffmpeg are working")
    sub.add_parser("tui", help="open the interactive menu (default)")
    return parser


def resolve_format(fmt: str | None, output: str | None, input_is_dir: bool) -> str:
    """Pick the output format from -f and/or the -o extension; refuse a conflict."""
    from_output = None
    if output and not input_is_dir:
        suffix = Path(output).suffix.lower().lstrip(".")
        if suffix in ENCODERS:
            from_output = suffix
    if fmt and from_output and fmt != from_output:
        raise ValueError(f"-f {fmt} conflicts with the output file's .{from_output} extension")
    return fmt or from_output or "mp3"


def options_from_args(args: argparse.Namespace) -> "NormalizeOptions":
    from audio_tools.pipeline import NormalizeOptions

    fmt = resolve_format(args.format, args.output, Path(args.input).is_dir())
    common = {"output_format": fmt, "bitrate": args.bitrate}
    if args.peak is not None:
        return NormalizeOptions.peak(args.peak, **common)
    if args.lufs is not None:
        target = args.lufs
    elif args.preset:
        target = LOUDNESS_PRESETS[args.preset]
    else:
        target = NormalizeOptions().target_lufs
    return NormalizeOptions(
        target_lufs=target,
        true_peak_ceiling_db=args.ceiling,
        use_limiter=not args.no_limiter,
        **common,
    )


def audio_files_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _path_key(path: Path) -> str:
    return str(path.resolve()).lower()


def plan_folder_outputs(files: list[Path], out_dir: Path, fmt: str) -> list[tuple[Path, Path]]:
    """Map each input to a unique output path that never overwrites an input.

    `a.wav` and `a.flac` would both become `a.mp3`; the second gets `a_flac.mp3`.
    Comparison is case-insensitive so it is safe on Windows and macOS volumes.
    """
    inputs = {_path_key(f) for f in files}
    taken: set[str] = set()
    plan: list[tuple[Path, Path]] = []
    for src in files:
        candidates = (
            out_dir / f"{src.stem}.{fmt}",
            out_dir / f"{src.stem}_{src.suffix.lstrip('.').lower()}.{fmt}",
        )
        chosen = next((c for c in candidates if _path_key(c) not in taken and _path_key(c) not in inputs), None)
        if chosen is None:
            raise ValueError(f"cannot pick a safe output name for {src.name} in {out_dir}")
        taken.add(_path_key(chosen))
        plan.append((src, chosen))
    return plan


def format_result(result: "NormalizeResult") -> str:
    lines = [f"Done: {result.output_path}"]
    if result.mode == "peak":
        lines += [
            f"  true peak:          {result.true_peak_before_db:6.1f} -> {result.true_peak_after_db:.1f} dBFS",
            f"  gain applied:       {result.gain_db:+6.1f} dB",
            f"  loudness:           {result.measured_lufs:6.1f} -> {result.achieved_lufs:.1f} LUFS",
        ]
    else:
        lines += [
            f"  loudness:           {result.measured_lufs:6.1f} -> {result.achieved_lufs:.1f} LUFS"
            f"  (target {result.target_lufs:.1f})",
            f"  gain applied:       {result.gain_db:+6.1f} dB",
            f"  true peak:          {result.true_peak_before_db:6.1f} -> {result.true_peak_after_db:.1f} dBFS",
        ]
        if result.limiter_reduction_db:
            lines.append(f"  limiter:            up to {result.limiter_reduction_db:.1f} dB of peak reduction")
    fmt = Path(result.output_path).suffix.lower().lstrip(".")
    if fmt in LOSSY_FORMATS and result.true_peak_after_db > LOSSY_PEAK_WARN_DB:
        lines.append(f"  note: {fmt} encoding can overshoot peaks this close to full scale; -1 dBFS is the safe choice")
    return "\n".join(lines)


def cmd_normalize(args: argparse.Namespace) -> int:
    from audio_tools.pipeline import normalize_file

    options = options_from_args(args)
    source = Path(args.input)
    if not source.is_dir():
        print(format_result(normalize_file(source, args.output, options)))
        return EXIT_OK

    files = audio_files_in(source)
    if not files:
        print(f"No audio files found in {source}", file=sys.stderr)
        return EXIT_FAILED
    out_dir = Path(args.output) if args.output else source / FOLDER_OUTPUT_NAME
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"output must be a folder when the input is a folder: {out_dir}")
    failures = 0
    plan = plan_folder_outputs(files, out_dir, options.output_format)
    for i, (src, dst) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {src.name}")
        try:
            print(format_result(normalize_file(src, dst, options)))
        except PER_FILE_ERRORS as exc:
            failures += 1
            print(f"  skipped: {exc}", file=sys.stderr)
    return EXIT_FAILED if failures else EXIT_OK


def cmd_measure(args: argparse.Namespace) -> int:
    from audio_tools.pipeline import measure_file

    failures = 0
    for path in args.input:
        try:
            m = measure_file(path)
        except PER_FILE_ERRORS as exc:
            failures += 1
            print(f"{Path(path).name}: failed: {exc}", file=sys.stderr)
            continue
        print(
            f"{Path(path).name}: {m.integrated_lufs:.1f} LUFS, true peak {m.true_peak_dbfs:.1f} dBFS, "
            f"{m.channels} ch @ {m.sample_rate} Hz, {m.duration_seconds:.1f} s"
        )
    return EXIT_FAILED if failures else EXIT_OK


def cmd_doctor(_: argparse.Namespace) -> int:
    from audio_tools import doctor

    checks = doctor.run_checks()
    print(doctor.format_report(checks))
    return EXIT_OK if doctor.all_required_ok(checks) else EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "tui"):
            from audio_tools.tui import run_tui

            return run_tui()
        handlers = {"normalize": cmd_normalize, "measure": cmd_measure, "doctor": cmd_doctor}
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return EXIT_CANCELLED
    except FFmpegNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_SETUP
    except (FileNotFoundError, ValueError, FFmpegError, OSError, MemoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
