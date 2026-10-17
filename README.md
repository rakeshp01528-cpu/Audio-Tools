# Audio Tools

Loudness-normalize and convert audio files from a simple menu or the command line.

- Measures integrated loudness per **ITU-R BS.1770-4** (LUFS, the standard behind Spotify, YouTube, Apple Music, EBU R128 broadcast).
- Normalizes to a loudness target **or** to a true-peak level (for example "make the loudest peak exactly -1 dBFS or 0 dBFS").
- A real **true-peak meter** (4x oversampled, BS.1770-4 Annex 2) and a **lookahead brickwall limiter**, so the loudness target is reached without clipping, with any ceiling up to 0 dBFS.
- Converts between wav, mp3, flac, m4a, ogg, opus and aiff via ffmpeg.
- Only one Python dependency (numpy). scipy is optional; without it a numpy FFT filter does the same job at the same speed. No compiled audio libraries that locked-down Windows machines block.
- Safe by default: writes to a temporary file and renames it when done, never overwrites an input, and gives files with the same name distinct outputs in folder mode.

## Install

Clone or download this folder, then:

| | |
|---|---|
| **Windows** | double-click `setup.bat` |
| **macOS / Linux** | run `./setup.sh` in Terminal |

The setup script:
1. finds a working Python 3.10+ (on Windows it skips the Microsoft Store placeholder and installs Python with winget if needed; on macOS it uses Homebrew if needed),
2. creates a `.venv` and installs this package into it, retrying flaky downloads and falling back cleanly if scipy or SSL certificates misbehave,
3. installs ffmpeg (winget / Homebrew, or a direct download into a local `bin/` folder),
4. runs `audio-tools doctor` to confirm everything works.

Re-running setup is safe. If something still fails, the last lines tell you exactly what is missing.

## Use

**Menu:** double-click `run.bat` (Windows) or run `./run.sh` (macOS / Linux). Type or drag a file into the window and follow the prompts. Type `q` at any prompt to quit.

**Command line:**

```bash
./run.sh normalize song.wav                      # -> song_normalized.mp3 at -16 LUFS, -1 dBFS ceiling
./run.sh normalize song.wav -p spotify -f flac   # preset target, flac output
./run.sh normalize song.wav --lufs -14 -c 0      # explicit LUFS, ceiling at 0 dBFS
./run.sh normalize song.wav --peak -1            # peak-normalize: loudest true peak at -1 dBFS
./run.sh normalize song.wav --peak 0 -f wav      # peak-normalize to full scale
./run.sh normalize song.wav -o out.flac          # format taken from the output name
./run.sh normalize ./album -f mp3 -b 256k        # every audio file in a folder -> ./album/normalized/
./run.sh measure song.wav other.mp3              # loudness + true peak, no changes
./run.sh doctor                                  # check Python, numpy, ffmpeg
./run.sh normalize --help
```

On Windows use `run.bat` in place of `./run.sh`. Inside the venv the command is also available as `audio-tools`.

Presets: `podcast` (-16), `spotify` (-14), `youtube` (-14), `apple_music` (-16), `broadcast_ebu` (-23), `broadcast_atsc` (-24).

## How the normalization works

1. **Decode** with ffmpeg to 32-bit float samples.
2. **Measure** integrated loudness (K-weighting, 400 ms blocks, absolute and relative gating) and true peak.
3. **Loudness mode:** apply the gain that hits the target LUFS. If the result would exceed the true-peak ceiling, a lookahead limiter (5 ms lookahead, 60 dB/s release, linked channels, 0.01 dB safety margin) brings only the peaks down, so the loudness target is kept. The achieved loudness and true peak are re-measured and reported. Pass `--no-limiter` to turn the whole file down instead (the older behaviour).
   The true-peak meter oversamples 4x with a 65-tap polyphase FIR (BS.1770-4 Annex 2 style), accurate to within 0.1 dB up to 20 kHz. Files with more than two channels are downmixed to stereo on decode.
   **Peak mode:** apply the single gain that puts the loudest true peak at the requested level. Nothing else is touched.
4. **Encode** with ffmpeg at the requested format and bitrate.

## Python API

```python
from audio_tools import NormalizeOptions, normalize_file, measure_file

normalize_file("in.wav", "out.mp3", NormalizeOptions(target_lufs=-14, true_peak_ceiling_db=-1))
normalize_file("in.wav", "out.wav", NormalizeOptions.peak(0.0, output_format="wav"))
print(measure_file("in.wav").integrated_lufs)
```

## Development

```bash
./setup.sh
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest --cov=audio_tools
```

Layout: `audio_tools/bs1770.py` (meter), `limiter.py` (true peak + limiter), `ffmpeg.py` (decode/encode), `pipeline.py` (normalization), `cli.py`, `tui.py`, `doctor.py`.

A 5-minute 48 kHz stereo file normalizes in about 8 seconds on an Apple Silicon Mac, with or without scipy, using roughly half a gigabyte of memory.

## Troubleshooting

- **"ffmpeg was not found"**: re-run setup. To use your own copy, put `ffmpeg` and `ffprobe` in a `bin/` folder next to this README, or set `AUDIO_TOOLS_FFMPEG_DIR` to the folder containing them.
- **pip SSL errors behind a corporate proxy**: setup retries with PyPI marked as trusted and tells you when it does.
- **scipy "DLL load failed" on Windows**: harmless. The doctor shows scipy as optional and the numpy FFT filter is used instead, at the same speed.
- **Ctrl-C** during a run prints "Cancelled" and leaves no half-written files.
- **Re-running setup** is quick: it skips the install when the package already imports. `./setup.sh --reinstall` or `setup.bat -Reinstall` forces it.
- **Python from the Microsoft Store opens instead of running**: setup detects the Store stub and installs real Python via winget, or opens python.org. Tick "Add python.exe to PATH" when installing manually.
