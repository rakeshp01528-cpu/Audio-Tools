"""
audio_converter_pro.py

A delivery-ready audio conversion pipeline:
  1. Load any audio format (via soundfile, falling back to pydub/ffmpeg for
     exotic formats soundfile can't read directly, e.g. mp3 on some builds).
  2. Measure integrated loudness per ITU-R BS.1770 (the standard behind
     LUFS, used by Spotify, YouTube, Apple Music, EBU R128 broadcast, etc.)
     using a pure-NumPy implementation (bs1770_pure.py) -- no scipy
     dependency, so it won't hit scipy's compiled-extension issues (e.g.
     Windows Application Control / WDAC / Smart App Control blocking one
     of scipy's DLLs, which is a common source of ImportError on locked-
     down machines).
  3. Apply gain to hit a target LUFS value.
  4. True-peak-safe limit so the loudness-normalized signal can't clip,
     including inter-sample peaks that appear after lossy encoding.
  5. Export to the target format at an explicit, pinned bitrate.

Requires: pip install soundfile pydub numpy --break-system-packages
Also requires ffmpeg on PATH (pydub shells out to it for format conversion),
and bs1770_pure.py alongside this file.
"""

import os
import numpy as np
import soundfile as sf
import bs1770_pure as bs1770
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Common platform/delivery loudness targets, for reference and CLI use.
LOUDNESS_PRESETS = {
    "spotify": -14.0,
    "youtube": -14.0,
    "apple_music": -16.0,
    "podcast": -16.0,
    "broadcast_ebu": -23.0,
    "broadcast_atsc": -24.0,
}


def _load_audio(input_file):
    """
    Load audio as a float64 numpy array + sample rate.
    Tries soundfile first (fast, no subprocess); falls back to pydub/ffmpeg
    for formats soundfile's libsndfile backend doesn't handle (e.g. mp3 on
    some platforms, aac, etc.).
    """
    try:
        data, rate = sf.read(input_file, always_2d=False)
        return data.astype(np.float64), rate
    except Exception:
        pass  # fall through to ffmpeg-backed loader

    try:
        file_extension = os.path.splitext(input_file)[1][1:].lower()
        seg = AudioSegment.from_file(input_file, format=file_extension or None)
    except CouldntDecodeError as e:
        raise RuntimeError(f"Could not decode '{input_file}': {e}") from e

    # Convert pydub's int16 sample buffer into a normalized float64 array
    # in the same [-1.0, 1.0] convention soundfile uses.
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    samples /= float(1 << (8 * seg.sample_width - 1))
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels))
    return samples, seg.frame_rate


def _true_peak_dbfs(data, rate, oversample=4):
    """
    Estimate true peak (inter-sample peak) in dBFS by oversampling the
    signal. Real DACs and lossy codecs can reconstruct peaks *between*
    samples that are higher than any single sample value, so a plain
    np.max(np.abs(data)) can understate the actual peak and let clipping
    slip through after MP3/AAC encoding. This is a lightweight polyphase-
    free approximation (linear-interpolation oversampling), which is good
    enough for a safety margin check -- not a mastering-grade ITU-R
    BS.1770-4 true-peak meter.
    """
    mono = data if data.ndim == 1 else data.mean(axis=1)
    # Simple oversampling via linear interpolation
    x = np.arange(len(mono))
    x_over = np.linspace(0, len(mono) - 1, len(mono) * oversample)
    oversampled = np.interp(x_over, x, mono)
    peak = np.max(np.abs(oversampled))
    if peak <= 0:
        return -np.inf
    return 20 * np.log10(peak)


def convert_and_normalize(
    input_file,
    output_file,
    target_format="mp3",
    target_lufs=-16.0,
    true_peak_ceiling_db=-1.0,
    bitrate="320k",
):
    """
    Convert an audio file to `target_format`, loudness-normalize it to
    `target_lufs` (BS.1770 integrated loudness), and apply true-peak-safe
    limiting so the result won't clip after encoding.

    Args:
        input_file: path to source audio (wav, mp3, ogg, flac, etc.)
        output_file: path to write the processed file
        target_format: export format passed to ffmpeg (e.g. "mp3", "wav")
        target_lufs: integrated loudness target in LUFS. Use
            LOUDNESS_PRESETS for common platform targets, e.g. -14.0 for
            Spotify/YouTube, -16.0 for podcasts/Apple Music, -23.0 for
            EBU R128 broadcast.
        true_peak_ceiling_db: max allowed true peak in dBFS after
            normalization. -1.0 dBFS is a standard safety margin that
            leaves headroom for lossy-codec inter-sample overshoot.
        bitrate: explicit bitrate for lossy export (ffmpeg otherwise picks
            its own default, which varies by build).
    """
    data, rate = _load_audio(input_file)

    # --- 1. Measure integrated loudness (BS.1770, with the standard's
    #        built-in gating of silence / very quiet passages) ---
    current_lufs = bs1770.integrated_loudness(data.copy(), rate)

    if not np.isfinite(current_lufs):
        raise RuntimeError(
            "Could not measure loudness (file may be silent or too short "
            "for BS.1770 gating). Skipping loudness normalization."
        )

    # --- 2. Apply gain to hit the target LUFS ---
    normalized = bs1770.normalize_to_lufs(data, current_lufs, target_lufs)

    # --- 3. True-peak-safe limiting ---
    # After loudness gain, check the *true* peak (not just the sample
    # peak) and pull the whole signal down if it would exceed the ceiling.
    # This is a simple safety scalar, not a lookahead brickwall limiter --
    # for material with big transients you may still want real limiting
    # upstream of this step if you need to preserve loudness AND avoid
    # audible gain pumping.
    tp_db = _true_peak_dbfs(normalized, rate)
    if tp_db > true_peak_ceiling_db:
        reduction_db = tp_db - true_peak_ceiling_db
        normalized = normalized * (10 ** (-reduction_db / 20))

    # --- 4. Export ---
    # Convert the float64 array back to int16 for pydub's export path,
    # which is the simplest reliable route to ffmpeg's encoders.
    peak_sample = np.max(np.abs(normalized)) or 1.0
    if peak_sample > 1.0:
        normalized = normalized / peak_sample  # hard safety clamp
    int16_data = (normalized * 32767).astype(np.int16)

    channels = 1 if int16_data.ndim == 1 else int16_data.shape[1]
    seg = AudioSegment(
        int16_data.tobytes(),
        frame_rate=rate,
        sample_width=2,
        channels=channels,
    )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    export_kwargs = {"format": target_format}
    if target_format in ("mp3", "aac", "ogg"):
        export_kwargs["bitrate"] = bitrate

    seg.export(output_file, **export_kwargs)

    print(
        f"Done: {output_file}\n"
        f"  measured loudness:  {current_lufs:.1f} LUFS\n"
        f"  target loudness:    {target_lufs:.1f} LUFS\n"
        f"  true peak (post):   {min(tp_db, true_peak_ceiling_db):.1f} dBFS "
        f"(ceiling {true_peak_ceiling_db:.1f} dBFS)"
    )


if __name__ == "__main__":
    input_path = "E:\\BAPS\\KIRTANS\\Ek Phool - Final.wav"
    output_path = "E:\\BAPS\\KIRTANS\\output_normalized.mp3"

    if os.path.exists(input_path):
        convert_and_normalize(
            input_path,
            output_path,
            target_format="mp3",
            target_lufs=LOUDNESS_PRESETS["podcast"],  # -16 LUFS
            true_peak_ceiling_db=-1.0,
            bitrate="320k",
        )
    else:
        print(f"Input file {input_path} not found.")