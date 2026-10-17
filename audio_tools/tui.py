"""A small interactive menu built on the standard library only.

No curses, no third-party widgets: numbered menus and prompts that work in
Windows Command Prompt, PowerShell, Windows Terminal, and macOS Terminal.
Paths can be typed or dragged into the window (surrounding quotes and the
PowerShell `& ` prefix are stripped).
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from audio_tools import doctor
from audio_tools.cli import PER_FILE_ERRORS, audio_files_in, format_result, plan_folder_outputs
from audio_tools.ffmpeg import FFmpegNotFound
from audio_tools.pipeline import NormalizeOptions, default_output_path, measure_file, normalize_file
from audio_tools.presets import (
    DEFAULT_BITRATE,
    DEFAULT_PRESET,
    DEFAULT_TRUE_PEAK_CEILING_DB,
    LOSSY_FORMATS,
    OUTPUT_FORMATS,
    PRESETS,
)

BITRATES = ("320k", "256k", "192k", "128k")


class Quit(Exception):
    """Raised when the user asks to leave the menu."""


def _enable_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":  # pragma: no cover - Windows only
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:  # noqa: BLE001
            return False
    return True


_COLOR = _enable_ansi()


def style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(text: str) -> str:
    return style(text, "1")


def dim(text: str) -> str:
    return style(text, "2")


def green(text: str) -> str:
    return style(text, "32")


def red(text: str) -> str:
    return style(text, "31")


def clean_path(raw: str) -> str:
    """Normalize a typed or drag-dropped path.

    Handles surrounding quotes, PowerShell's `& '...'` prefix and doubled
    single quotes, and the backslash-escaped spaces macOS Terminal inserts.
    """
    text = raw.strip()
    if text.startswith("& "):  # PowerShell drag-and-drop prefix
        text = text[2:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        quote = text[0]
        text = text[1:-1]
        if quote == "'":
            text = text.replace("''", "'")
    elif os.name != "nt" and "\\" in text:
        try:
            parts = shlex.split(text)
            if len(parts) == 1:
                text = parts[0]
        except ValueError:
            pass
    return os.path.expanduser(text.strip())


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" {dim(f'[{default}]')}" if default is not None else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise Quit from None
    if answer.lower() in ("q", "quit", "exit"):
        raise Quit
    return answer or (default or "")


def choose(title: str, options: list[tuple[str, str]], default_index: int = 0) -> str:
    """Show a numbered menu; `options` are (key, label). Returns the chosen key."""
    print()
    print(bold(title))
    for i, (_, label) in enumerate(options, 1):
        marker = "*" if i - 1 == default_index else " "
        print(f"  {marker} {i}) {label}")
    while True:
        answer = ask("Choose", str(default_index + 1))
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print(red(f"Enter a number from 1 to {len(options)} (or q to quit)."))


def ask_existing_path(prompt: str) -> Path:
    while True:
        path = Path(clean_path(ask(prompt)))
        if path.exists():
            return path
        print(red(f"Not found: {path}"))


def ask_float(prompt: str, default: float) -> float:
    while True:
        answer = ask(prompt, f"{default:g}")
        try:
            return float(answer)
        except ValueError:
            print(red("Enter a number, e.g. -16"))


PEAK_CHOICES = (
    ("peak:-1", "Peak normalize to -1 dBFS"),
    ("peak:0", "Peak normalize to 0 dBFS"),
    ("peak:custom", "Peak normalize to a custom level"),
)


def ask_target() -> dict:
    """Ask for the normalization target; returns NormalizeOptions keyword args."""
    options = [(p.key, f"{p.label}  ({p.lufs:g} LUFS)") for p in PRESETS.values()]
    options.append(("custom", "Custom LUFS value"))
    options.extend(PEAK_CHOICES)
    choice = choose("Normalize to", options, list(PRESETS).index(DEFAULT_PRESET))
    if choice.startswith("peak:"):
        level = choice.split(":", 1)[1]
        peak = ask_float("Target true peak (dBFS, 0 or lower)", -1.0) if level == "custom" else float(level)
        return {"mode": "peak", "target_peak_dbfs": peak}
    target = ask_float("Target LUFS", -16.0) if choice == "custom" else PRESETS[choice].lufs
    ceiling = ask_float("True-peak ceiling (dBFS, 0 or lower)", DEFAULT_TRUE_PEAK_CEILING_DB)
    return {"mode": "loudness", "target_lufs": target, "true_peak_ceiling_db": ceiling}


def ask_options() -> NormalizeOptions:
    while True:
        target = ask_target()
        try:
            NormalizeOptions(**target)
            break
        except ValueError as exc:
            print(red(str(exc)))
    fmt = choose("Output format", [(f, f) for f in OUTPUT_FORMATS], 0)
    bitrate = DEFAULT_BITRATE
    if fmt in LOSSY_FORMATS:
        bitrate = choose("Bitrate", [(b, b) for b in BITRATES], 0)
    return NormalizeOptions(output_format=fmt, bitrate=bitrate, **target)


def _report(action, *args) -> bool:
    try:
        result = action(*args)
    except PER_FILE_ERRORS as exc:
        print(red(f"  failed: {exc}"))
        return False
    print(green(format_result(result)))
    return True


def flow_single_file() -> None:
    src = ask_existing_path("Input file (type or drag it here)")
    if src.is_dir():
        print(red("That is a folder. Use the folder option from the main menu."))
        return
    options = ask_options()
    suggested = default_output_path(src, options.output_format)
    out = Path(clean_path(ask("Output file", str(suggested))))
    if out.suffix.lower().lstrip(".") != options.output_format:
        out = out.with_suffix(f".{options.output_format}")
    print(dim("Working..."))
    _report(normalize_file, src, out, options)


def flow_folder() -> None:
    folder = ask_existing_path("Folder with audio files")
    if not folder.is_dir():
        print(red("That is a file. Use the single-file option from the main menu."))
        return
    files = audio_files_in(folder)
    if not files:
        print(red("No audio files found there."))
        return
    print(dim(f"Found {len(files)} audio file(s)."))
    options = ask_options()
    out_dir = Path(clean_path(ask("Output folder", str(folder / "normalized"))))
    try:
        plan = plan_folder_outputs(files, out_dir, options.output_format)
    except ValueError as exc:
        print(red(str(exc)))
        return
    ok = 0
    for i, (src, dst) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {src.name}")
        ok += _report(normalize_file, src, dst, options)
    print(bold(f"\n{ok}/{len(plan)} files written to {out_dir}"))


def flow_measure() -> None:
    src = ask_existing_path("Audio file to measure")
    try:
        m = measure_file(src)
    except PER_FILE_ERRORS as exc:
        print(red(f"  failed: {exc}"))
        return
    print(
        green(
            f"  loudness:   {m.integrated_lufs:6.1f} LUFS\n"
            f"  true peak:  {m.true_peak_dbfs:6.1f} dBFS\n"
            f"  {m.channels} ch @ {m.sample_rate} Hz, {m.duration_seconds:.1f} s"
        )
    )


def flow_doctor() -> None:
    print()
    print(doctor.format_report(doctor.run_checks()))


MENU = [
    ("single", "Normalize and convert one file", flow_single_file),
    ("folder", "Normalize every audio file in a folder", flow_folder),
    ("measure", "Measure loudness only", flow_measure),
    ("doctor", "Check my setup", flow_doctor),
]


def run_tui() -> int:
    print(bold("\nAudio Tools") + dim("  -  loudness normalize + convert   (q to quit at any prompt)"))
    try:
        from audio_tools.ffmpeg import find_ffmpeg

        find_ffmpeg()
    except FFmpegNotFound as exc:
        print(red(f"\n{exc}"))
        flow_doctor()
        return 2
    try:
        while True:
            key = choose("What do you want to do?", [(k, label) for k, label, _ in MENU] + [("quit", "Quit")], 0)
            if key == "quit":
                break
            try:
                next(fn for k, _, fn in MENU if k == key)()
            except KeyboardInterrupt:
                print(red("\nCancelled."))
    except Quit:
        pass
    print(dim("\nBye."))
    return 0
