#!/bin/bash
# Strata Voice — one-command installer for macOS on Apple Silicon.
#
#   ./install.sh                 interactive (asks which tier)
#   ./install.sh --light         Lightweight tier, no questions
#   ./install.sh --recommended   Recommended tier, no questions
#
# Idempotent: safe to re-run any time. It never overwrites an existing
# ~/.vui (your profile + memories) — it only seeds settings on first install.
set -euo pipefail

BOLD=$(tput bold 2>/dev/null || true); DIM=$(tput dim 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)
ok()   { printf "  ✓ %s\n" "$1"; }
info() { printf "  · %s\n" "$1"; }
die()  { printf "\n  ✗ %s\n" "$1"; [ -n "${2:-}" ] && printf "    → %s\n" "$2"; echo; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

LIGHT_CHAT="gemma4:e4b"   # Gemma E4B: selective-activation (~4B active), fastest
LIGHT_MEM=""              # same as chat — one small model does both on 16 GB Macs
REC_CHAT="qwen3.6:latest"
REC_MEM=""   # same as chat: models load one at a time, so a second
             # model would evict the chat model on every background memory job
EMBED_MODEL="nomic-embed-text"

echo
printf "%s✦ Strata Voice installer%s\n" "$BOLD" "$RESET"
echo

# ---- 1. preflight -------------------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || die "This app needs macOS." "MLX (the on-device ML stack) is Apple-only."
[ "$(uname -m)" = "arm64" ]  || die "This app needs Apple Silicon (M1 or newer)." "Intel Macs can't run MLX."
ok "Apple Silicon Mac"

# espeak-ng (inside the voice stack) has a ~160-character limit on its data
# path and fails with a baffling native error beyond it. Catch that up front.
if [ "${#HERE}" -gt 88 ]; then
  die "This folder's path is too deep for the speech engine (${#HERE} chars; needs ≤ 88)." \
      "Move the folder somewhere shorter, e.g.:  mv \"$HERE\" ~/Strata-Voice && cd ~/Strata-Voice && ./install.sh"
fi
ok "install path length OK"

command -v git >/dev/null || die "git is missing." "Install the Xcode command line tools:  xcode-select --install"
ok "git"

if ! command -v brew >/dev/null; then
  die "Homebrew is missing (needed to install Python 3.12 / ffmpeg / Ollama)." \
      'Install it first:  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
fi
ok "Homebrew"

if ! command -v python3.12 >/dev/null; then
  info "Python 3.12 not found — installing via Homebrew…"
  brew install python@3.12 || die "Couldn't install Python 3.12." "Try:  brew install python@3.12"
fi
ok "Python 3.12 ($(python3.12 -V 2>&1 | cut -d' ' -f2))"

if ! command -v ffmpeg >/dev/null; then
  info "ffmpeg not found — installing via Homebrew…"
  brew install ffmpeg || die "Couldn't install ffmpeg." "Try:  brew install ffmpeg"
fi
ok "ffmpeg"

if ! command -v ollama >/dev/null; then
  info "Ollama not found — installing via Homebrew…"
  brew install ollama || die "Couldn't install Ollama." "Download it from https://ollama.com/download instead, then re-run this script."
fi
ok "Ollama"

# ---- 2. python environment ----------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  info "Creating Python environment…"
  python3.12 -m venv .venv
fi
info "Installing Python packages (a few minutes on first run)…"
info "${DIM}strata-memory installs from GitHub — this needs access to the repo.${RESET}"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt || die "Python package install failed." \
  "If the failure mentions strata-memory: the repo may be private — you need access (or wait for the public release)."
ok "Python environment ready"

# ---- 3. ollama up -------------------------------------------------------------
if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
  info "Starting Ollama…"
  (ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 30); do
    sleep 1
    curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null && break
  done
  curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null || \
    die "Ollama didn't start." "Open the Ollama app (or run 'ollama serve') and re-run this script."
fi
ok "Ollama is running"

# ---- 4. tier ------------------------------------------------------------------
TIER="${1:-}"
if [ "$TIER" != "--light" ] && [ "$TIER" != "--recommended" ]; then
  echo
  printf "%sWhich install?%s\n" "$BOLD" "$RESET"
  echo   "  [1] Lightweight  (~10 GB)  runs on 16 GB Macs; fastest replies"
  echo   "  [2] Recommended  (~24 GB)  best replies (36B brain); needs a 32 GB+ Mac"
  printf "  choice [2]: "
  read -r CHOICE || CHOICE=""
  [ "$CHOICE" = "1" ] && TIER="--light" || TIER="--recommended"
fi

if [ "$TIER" = "--light" ]; then
  CHAT_MODEL="$LIGHT_CHAT"; MEM_MODEL="$LIGHT_MEM"; TIER_NAME="Lightweight"
else
  CHAT_MODEL="$REC_CHAT";   MEM_MODEL="$REC_MEM";   TIER_NAME="Recommended"
fi
echo
info "Tier: $TIER_NAME  ·  chat: $CHAT_MODEL$( [ -n "$MEM_MODEL" ] && echo "  ·  memory: $MEM_MODEL" )"

# ---- 5. pull models -----------------------------------------------------------
pull() {
  # ollama list shows bare names as "name:latest" — match both forms
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qxe "$1" -e "$1:latest"; then
    ok "$1 (already downloaded)"
  else
    info "Downloading $1 …"
    ollama pull "$1" || die "Couldn't pull $1." "Check your connection and re-run — the script picks up where it left off."
  fi
}
pull "$CHAT_MODEL"
[ -n "$MEM_MODEL" ] && [ "$MEM_MODEL" != "$CHAT_MODEL" ] && pull "$MEM_MODEL"
pull "$EMBED_MODEL"

# ---- 6. seed settings (never clobber an existing install) ----------------------
SETTINGS="$HOME/.vui/voicechat/settings.json"
if [ -f "$SETTINGS" ]; then
  info "Existing settings found at ~/.vui — leaving your profile, memories, and settings untouched."
  printf "  update just the model choices to this tier? [y/N]: "
  read -r UPD || UPD=""
  if [ "$UPD" = "y" ] || [ "$UPD" = "Y" ]; then
    .venv/bin/python - "$SETTINGS" "$CHAT_MODEL" "$MEM_MODEL" <<'PY'
import json, sys
path, chat, mem = sys.argv[1], sys.argv[2], sys.argv[3]
s = json.load(open(path))
s["ollama_model"] = chat
s["memory_model"] = mem
json.dump(s, open(path, "w"), indent=2)
print("  ✓ model choices updated")
PY
  fi
else
  mkdir -p "$(dirname "$SETTINGS")"
  .venv/bin/python - "$SETTINGS" "$CHAT_MODEL" "$MEM_MODEL" <<'PY'
import json, sys
path, chat, mem = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({"ollama_model": chat, "memory_model": mem, "configured": True},
          open(path, "w"), indent=2)
print("  ✓ settings seeded for first run")
PY
fi

# ---- 7. optional speech-model prewarm ------------------------------------------
echo
printf "Download the speech models now (~1 GB)? Otherwise the first launch does it. [Y/n]: "
read -r WARM || WARM="n"
if [ "$WARM" != "n" ] && [ "$WARM" != "N" ]; then
  info "Warming up Parakeet (speech-to-text) + Kokoro (voice)…"
  .venv/bin/python - <<'PY'
import voicechat as vc
from mlx_audio.tts.utils import load_model
vc.load_asr()
load_model(vc.TTS_MODEL)
print("  ✓ speech models cached")
PY
fi

echo
printf "%s✦ Done.%s  Start it any time with:  %s./start.sh%s\n" "$BOLD" "$RESET" "$BOLD" "$RESET"
echo
