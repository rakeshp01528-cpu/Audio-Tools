import os

import pytest

from audio_tools import cli, doctor, tui
from audio_tools.presets import LOUDNESS_PRESETS
from tests.conftest import needs_ffmpeg


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"C:\\Music\\a b.wav"', "C:\\Music\\a b.wav"),
        ("'/tmp/x.wav'", "/tmp/x.wav"),
        ("& 'C:\\x y\\a.wav'", "C:\\x y\\a.wav"),
        ("  plain.wav  ", "plain.wav"),
        ("'C:\\it''s.wav'", "C:\\it's.wav"),
    ],
)
def test_clean_path(raw, expected):
    assert tui.clean_path(raw) == expected


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="macOS Terminal drag-and-drop escaping")
def test_clean_path_backslash_escaped_spaces():
    assert tui.clean_path("/Users/me/My\\ Song.wav") == "/Users/me/My Song.wav"


@pytest.mark.unit
def test_plan_folder_outputs_avoids_collisions(tmp_path):
    for name in ("a.wav", "a.flac", "b.mp3"):
        (tmp_path / name).write_text("")
    files = cli.audio_files_in(tmp_path)
    plan = cli.plan_folder_outputs(files, tmp_path, "mp3")  # output folder == input folder
    outputs = [dst.name for _, dst in plan]
    assert outputs == ["a.mp3", "a_wav.mp3", "b_mp3.mp3"]  # b.mp3 must not overwrite itself
    assert len(set(o.lower() for o in outputs)) == len(outputs)


@pytest.mark.unit
def test_cmd_measure_continues_after_bad_file(tmp_path, capsys):
    (tmp_path / "bad.wav").write_bytes(b"nope")
    code = cli.main(["measure", str(tmp_path / "bad.wav"), str(tmp_path / "missing.wav")])
    assert code == 1
    err = capsys.readouterr().err
    assert "bad.wav: failed" in err and "missing.wav: failed" in err


@pytest.mark.unit
def test_keyboard_interrupt_is_cancelled(monkeypatch, capsys):
    def boom(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_doctor", boom)
    assert cli.main(["doctor"]) == 130
    assert "Cancelled" in capsys.readouterr().err


@pytest.mark.unit
def test_options_from_preset_and_lufs():
    parser = cli.build_parser()
    a = parser.parse_args(["normalize", "in.wav", "-p", "spotify", "-f", "flac"])
    assert cli.options_from_args(parser.parse_args(["normalize", "in.wav", "-o", "out.flac"])).output_format == "flac"
    with pytest.raises(ValueError, match="conflicts"):
        cli.options_from_args(parser.parse_args(["normalize", "in.wav", "-o", "out.mp3", "-f", "wav"]))
    o = cli.options_from_args(a)
    assert o.target_lufs == LOUDNESS_PRESETS["spotify"]
    assert o.output_format == "flac"
    b = parser.parse_args(["normalize", "in.wav", "--lufs", "-18.5", "-c", "-2", "-b", "128k"])
    o2 = cli.options_from_args(b)
    assert (o2.target_lufs, o2.true_peak_ceiling_db, o2.bitrate) == (-18.5, -2.0, "128k")
    assert cli.options_from_args(parser.parse_args(["normalize", "x"])).target_lufs == -16.0
    c = cli.options_from_args(parser.parse_args(["normalize", "x", "--peak", "0"]))
    assert (c.mode, c.target_peak_dbfs) == ("peak", 0.0)
    d = cli.options_from_args(parser.parse_args(["normalize", "x", "-c", "0", "--no-limiter"]))
    assert (d.true_peak_ceiling_db, d.use_limiter) == (0.0, False)


@pytest.mark.unit
def test_cli_rejects_ceiling_above_zero(tmp_path, capsys):
    (tmp_path / "a.wav").write_bytes(b"")
    assert cli.main(["normalize", str(tmp_path / "a.wav"), "-c", "1"]) == 1
    assert "0 dBFS or lower" in capsys.readouterr().err


@pytest.mark.unit
def test_preset_and_lufs_are_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["normalize", "x", "-p", "spotify", "--lufs", "-14"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["normalize", "x", "--peak", "0", "--lufs", "-14"])


@pytest.mark.unit
def test_missing_input_exit_code(tmp_path, capsys):
    code = cli.main(["normalize", str(tmp_path / "missing.wav")])
    assert code == 1
    assert "not found" in capsys.readouterr().err


@pytest.mark.unit
def test_empty_folder_exit_code(tmp_path, capsys):
    assert cli.main(["normalize", str(tmp_path)]) == 1
    assert "No audio files" in capsys.readouterr().err


@pytest.mark.unit
def test_audio_files_in_filters_by_extension(tmp_path):
    for name in ("a.wav", "b.MP3", "c.txt", "d.flac"):
        (tmp_path / name).write_text("")
    assert [p.name for p in cli.audio_files_in(tmp_path)] == ["a.wav", "b.MP3", "d.flac"]


@pytest.mark.unit
def test_doctor_report_mentions_each_check(capsys):
    checks = doctor.run_checks()
    report = doctor.format_report(checks)
    for name in ("Python", "numpy", "scipy", "ffmpeg", "ffprobe"):
        assert name in report
    assert cli.main(["doctor"]) == 0
    assert "Python" in capsys.readouterr().out


@pytest.mark.unit
def test_tui_choose_and_quit(monkeypatch, capsys):
    answers = iter(["9", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert tui.choose("Pick", [("a", "A"), ("b", "B")]) == "b"
    assert "Enter a number" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda _: "q")
    with pytest.raises(tui.Quit):
        tui.ask("anything")


@pytest.mark.unit
def test_tui_ask_default_and_float(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert tui.ask("x", "dflt") == "dflt"
    assert tui.ask_float("x", -16.0) == -16.0
    answers = iter(["abc", "-14"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert tui.ask_float("x", -16.0) == -14.0


@needs_ffmpeg
@pytest.mark.integration
def test_cli_normalize_and_measure_end_to_end(sine_wav, tmp_path, capsys):
    out = tmp_path / "o.wav"
    assert cli.main(["normalize", str(sine_wav), "-o", str(out), "-f", "wav", "-p", "podcast"]) == 0
    assert out.is_file()
    assert cli.main(["measure", str(out)]) == 0
    assert "-16." in capsys.readouterr().out


@needs_ffmpeg
@pytest.mark.integration
def test_cli_folder_mode(sine_wav, tmp_path, capsys):
    (sine_wav.parent / "notes.txt").write_text("skip me")
    out_dir = tmp_path / "out"
    assert cli.main(["normalize", str(sine_wav.parent), "-o", str(out_dir), "-f", "flac"]) == 0
    assert (out_dir / "sine.flac").is_file()


@needs_ffmpeg
@pytest.mark.integration
def test_tui_single_file_flow(sine_wav, tmp_path, monkeypatch, capsys):
    out = tmp_path / "tui_out.flac"
    answers = iter(
        [
            str(sine_wav),  # input
            "",  # target: default preset (podcast)
            "",  # ceiling default
            "3",  # format: flac
            str(out),  # output
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    tui.flow_single_file()
    assert out.is_file()
    assert "Done" in capsys.readouterr().out


@pytest.mark.unit
def test_tui_ask_target_peak_and_validation(monkeypatch, capsys):
    n_presets = len(tui.PRESETS)
    answers = iter([str(n_presets + 3)])  # "Peak normalize to 0 dBFS"
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert tui.ask_target() == {"mode": "peak", "target_peak_dbfs": 0.0}
    # Custom peak above 0 is rejected and asked again.
    answers = iter([str(n_presets + 4), "2", str(n_presets + 4), "-0.5", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    opts = tui.ask_options()
    assert "0 dBFS or lower" in capsys.readouterr().out
    assert (opts.mode, opts.target_peak_dbfs, opts.output_format) == ("peak", -0.5, "wav")


@needs_ffmpeg
@pytest.mark.integration
def test_run_tui_quits(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "5")
    assert tui.run_tui() == 0


@needs_ffmpeg
@pytest.mark.integration
def test_cli_folder_mode_with_same_stems(tmp_path, capsys):
    from tests.conftest import sine, write_wav

    write_wav(tmp_path / "a.wav", sine(997.0, 1.0, -20.0))
    write_wav(tmp_path / "A.WAV", sine(997.0, 1.0, -20.0)) if not (tmp_path / "A.WAV").exists() else None
    import shutil

    shutil.copy(tmp_path / "a.wav", tmp_path / "a.aiff")
    assert cli.main(["normalize", str(tmp_path), "-o", str(tmp_path), "-f", "wav"]) == 0
    names = sorted(p.name.lower() for p in tmp_path.iterdir())
    assert "a_wav.wav" in names and "a_aiff.wav" in names
    assert (tmp_path / "a.wav").stat().st_size == (tmp_path / "a.aiff").stat().st_size  # originals untouched
