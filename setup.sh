#!/usr/bin/env bash
# One-shot setup for macOS / Linux:
#   - finds a usable Python 3.10+ (installs via Homebrew on macOS if missing)
#   - creates .venv and installs this package into it, retrying flaky downloads
#   - installs ffmpeg (Homebrew, or a static download into ./bin)
#   - runs `audio-tools doctor` at the end
# Re-running is safe and fast. Usage: ./setup.sh [--reinstall]
set -uo pipefail

resolve_dir() {  # follow symlinks without readlink -f (not on older macOS)
  local src="${BASH_SOURCE[0]}" dir
  while [ -L "$src" ]; do
    dir="$(cd -P "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")" && pwd
}
ROOT="$(resolve_dir)" || exit 1
cd "$ROOT" || exit 1

REINSTALL=0
[ "${1:-}" = "--reinstall" ] && REINSTALL=1

VENV="$ROOT/.venv"
PY_IN_VENV="$VENV/bin/python"
BIN_DIR="$ROOT/bin"
PIP_RETRIES=3
TRUST_PYPI=0
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=60

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m%s\033[0m\n' "$*"; }
warn() { printf '    \033[33m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

os="$(uname -s)"
arch="$(uname -m)"
have() { command -v "$1" >/dev/null 2>&1; }

TMPDIRS=()
cleanup() { local d; for d in "${TMPDIRS[@]:-}"; do [ -n "$d" ] && [ -d "$d" ] && rm -rf "$d"; done; }
trap cleanup EXIT
mktmp() { local d; d="$(mktemp -d)" || die "mktemp failed"; TMPDIRS+=("$d"); echo "$d"; }

# Homebrew that is installed but not on PATH yet (common right after installing it)
if ! have brew; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew /home/linuxbrew/.linuxbrew/bin/brew; do
    [ -x "$b" ] && { eval "$("$b" shellenv)"; break; }
  done
fi

# ---------------------------------------------------------------- python
python_ok() {
  local exe; exe="$(command -v "$1" 2>/dev/null)" || return 1
  # Apple's /usr/bin/python3 is a stub that pops the Xcode tools dialog, and is too old anyway.
  [ "$os" = "Darwin" ] && [ "$exe" = "/usr/bin/python3" ] && return 1
  "$exe" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

find_python() {
  local c
  for c in python3.12 python3.13 python3.11 python3.14 python3 python \
           /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if python_ok "$c"; then command -v "$c"; return 0; fi
  done
  return 1
}

say "Looking for Python 3.10+"
PY="$(find_python || true)"
if [ -z "$PY" ]; then
  if [ "$os" = "Darwin" ] && have brew; then
    warn "No usable Python found. Installing python@3.12 with Homebrew..."
    brew install python@3.12 || die "Homebrew failed to install Python. Install it from https://www.python.org/downloads/ and re-run."
    PY="$(find_python || true)"
  elif [ "$os" = "Darwin" ]; then
    die "No usable Python found and Homebrew is not installed. Install Python 3.12 from https://www.python.org/downloads/macos/ (or install Homebrew from https://brew.sh) and re-run."
  else
    die "No usable Python 3.10+ with venv support found. On Debian/Ubuntu: sudo apt install python3 python3-venv python3-pip"
  fi
fi
[ -n "$PY" ] || die "Still no usable Python after the install attempt."
ok "Using $PY ($("$PY" --version 2>&1))"

# ------------------------------------------------------------------ venv
say "Creating virtual environment in .venv"
venv_ok() {
  [ -x "$PY_IN_VENV" ] \
    && "$PY_IN_VENV" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1 \
    && "$PY_IN_VENV" -m pip --version >/dev/null 2>&1
}
if venv_ok; then
  ok "Existing venv is usable."
else
  [ -d "$VENV" ] && { warn "Existing venv is broken; moving it aside."; mv "$VENV" "$VENV.broken.$(date +%Y%m%d-%H%M%S)"; }
  "$PY" -m venv "$VENV" || die "Could not create the venv."
  venv_ok || die "The new venv does not work (python or pip missing). On Debian/Ubuntu: sudo apt install python3-venv"
  ok "Created."
fi

# ------------------------------------------------------------------- pip
pip_install() {
  # pip_install <args...> : retries; on SSL trouble, switches to trusted hosts for the rest of the run.
  local attempt log trust=()
  log="$(mktmp)/pip.log"
  for attempt in $(seq 1 "$PIP_RETRIES"); do
    [ "$TRUST_PYPI" = 1 ] && trust=(--trusted-host pypi.org --trusted-host files.pythonhosted.org)
    if "$PY_IN_VENV" -m pip install --prefer-binary --retries 2 "${trust[@]}" "$@" 2>&1 | tee "$log"; then return 0; fi
    warn "pip attempt $attempt/$PIP_RETRIES failed."
    if [ "$TRUST_PYPI" = 0 ] && grep -qiE 'SSL|CERTIFICATE_VERIFY_FAILED' "$log"; then
      warn "SSL errors talking to PyPI (common behind corporate proxies)."
      warn "Retrying while trusting pypi.org / files.pythonhosted.org without certificate checks."
      TRUST_PYPI=1
    else
      sleep 2
    fi
  done
  return 1
}

if [ "$REINSTALL" = 0 ] && "$PY_IN_VENV" -c 'import audio_tools, numpy' >/dev/null 2>&1; then
  say "audio-tools is already installed in .venv (use --reinstall to force)"
else
  say "Upgrading pip"
  pip_install --upgrade pip >/dev/null || warn "Could not upgrade pip; continuing with the bundled one."

  say "Installing audio-tools (+ scipy for faster filtering, if available)"
  if ! pip_install -e ".[fast]"; then
    warn "Install with scipy failed; installing without it (slower, but works)."
    pip_install -e . || die "Installing the package failed. Check your internet connection / proxy and re-run."
  fi
  ok "Installed."
fi

# ---------------------------------------------------------------- ffmpeg
tool_runs() { [ -x "$1" ] && "$1" -version >/dev/null 2>&1; }
have_local_ffmpeg() { tool_runs "$BIN_DIR/ffmpeg" && tool_runs "$BIN_DIR/ffprobe"; }
have_path_ffmpeg() { have ffmpeg && have ffprobe && ffmpeg -version >/dev/null 2>&1; }

download_static_ffmpeg_mac() {
  # evermeet.cx ships x86_64 builds; they need Rosetta on Apple Silicon.
  if [ "$arch" = "arm64" ] && ! /usr/bin/arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
    warn "Static ffmpeg downloads are Intel-only and Rosetta is not installed."
    warn "Run:  softwareupdate --install-rosetta --agree-to-license   (or install Homebrew and re-run)."
    return 1
  fi
  mkdir -p "$BIN_DIR"
  local tool url tmp
  tmp="$(mktmp)"
  for tool in ffmpeg ffprobe; do
    url="https://evermeet.cx/ffmpeg/getrelease/$tool/zip"
    warn "Downloading $tool from $url"
    curl -fsSL --retry 3 -o "$tmp/$tool.zip" "$url" || return 1
    unzip -oq "$tmp/$tool.zip" -d "$BIN_DIR" || return 1
    chmod +x "$BIN_DIR/$tool"
    xattr -d com.apple.quarantine "$BIN_DIR/$tool" 2>/dev/null || true
    tool_runs "$BIN_DIR/$tool" || { warn "Downloaded $tool does not run on this Mac."; mv "$BIN_DIR/$tool" "$BIN_DIR/$tool.unusable"; return 1; }
  done
}

say "Checking ffmpeg"
for t in ffmpeg ffprobe; do  # a broken ./bin copy would shadow a good PATH one
  [ -e "$BIN_DIR/$t" ] && ! tool_runs "$BIN_DIR/$t" && { warn "./bin/$t does not run; moving it aside."; mv "$BIN_DIR/$t" "$BIN_DIR/$t.unusable"; }
done
if have_local_ffmpeg; then
  ok "ffmpeg found in ./bin"
elif have_path_ffmpeg; then
  ok "ffmpeg found on PATH."
elif [ "$os" = "Darwin" ]; then
  if have brew; then
    warn "Installing ffmpeg with Homebrew (this can take a few minutes)..."
    brew install ffmpeg || warn "Homebrew install failed; trying a static download instead."
  fi
  if ! have_path_ffmpeg && ! have_local_ffmpeg; then
    download_static_ffmpeg_mac && ok "ffmpeg placed in ./bin" || warn "Could not install ffmpeg. Install it manually (brew install ffmpeg) and re-run."
  fi
else
  if have apt-get; then warn "Run: sudo apt-get install -y ffmpeg";
  elif have dnf; then warn "Run: sudo dnf install -y ffmpeg";
  elif have pacman; then warn "Run: sudo pacman -S ffmpeg";
  else warn "Install ffmpeg with your package manager, or drop ffmpeg + ffprobe into ./bin"; fi
fi

# ---------------------------------------------------------------- doctor
say "Checking the install"
"$PY_IN_VENV" -m audio_tools doctor
status=$?
echo
if [ $status -eq 0 ]; then
  ok "Setup complete. Start the menu with:  ./run.sh"
else
  warn "Setup finished with problems (see above). Fix them and re-run ./setup.sh"
fi
exit $status
