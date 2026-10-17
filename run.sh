#!/usr/bin/env bash
# Launch Audio Tools. No arguments opens the menu; arguments go to the CLI,
# e.g.  ./run.sh normalize song.wav -p spotify -f mp3
# Relative file paths are resolved from the folder you run this in.
resolve_dir() {
  local src="${BASH_SOURCE[0]}" dir
  while [ -L "$src" ]; do
    dir="$(cd -P "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")" && pwd
}
ROOT="$(resolve_dir)" || exit 1
PY="$ROOT/.venv/bin/python"
if ! "$PY" -c 'import audio_tools, numpy' >/dev/null 2>&1; then
  echo "First run: setting things up..."
  "$ROOT/setup.sh" || exit $?
  echo
fi
exec "$PY" -m audio_tools "$@"
