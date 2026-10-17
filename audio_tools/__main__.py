"""Allow `python -m audio_tools`."""

import sys

try:
    from audio_tools.cli import main
except ImportError as exc:  # pragma: no cover - only on broken installs
    print(f"audio-tools could not start: {exc}\nRe-run the setup script (setup.sh / setup.bat).", file=sys.stderr)
    sys.exit(2)

sys.exit(main())
