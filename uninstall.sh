#!/bin/bash
# Strata Voice — uninstaller. Prompts before every destructive step and
# defaults to NO for anything that touches your data.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

MODELS=(
  "gemma4:e4b"
  "qwen3.6:latest"
  "nomic-embed-text"
)

echo
echo "✦ Strata Voice uninstaller"
echo
echo "This can remove:"
echo "  1. the running server + the Python environment in this folder (.venv)"
echo "  2. your data: profile, memories, conversations   (~/.vui)  — asks twice"
echo "  3. the Ollama models the installer pulled"
echo "  4. the cached speech models (Hugging Face cache)"
echo "  5. this folder itself is NOT touched — delete it yourself when done"
echo

# 1. stop server + remove venv
PIDS=$(pgrep -f "$HERE/.venv/bin/python server.py" 2>/dev/null || true)
[ -n "$PIDS" ] && { echo "· stopping server…"; kill $PIDS 2>/dev/null || true; sleep 1; }
if [ -d .venv ]; then
  printf "Remove the Python environment (.venv)? [Y/n]: "
  read -r A
  [ "$A" != "n" ] && [ "$A" != "N" ] && rm -rf .venv && echo "  ✓ removed .venv"
fi

# 2. user data — double confirm, default NO
if [ -d "$HOME/.vui" ]; then
  printf "Delete ALL your data — profile, memories, every conversation (~/.vui)? [y/N]: "
  read -r A
  if [ "$A" = "y" ] || [ "$A" = "Y" ]; then
    printf "  Really delete your memories? This cannot be undone. Type 'delete' to confirm: "
    read -r B
    if [ "$B" = "delete" ]; then
      rm -rf "$HOME/.vui" && echo "  ✓ removed ~/.vui"
    else
      echo "  · kept ~/.vui"
    fi
  else
    echo "  · kept ~/.vui (your memories are safe)"
  fi
fi

# 3. ollama models
if command -v ollama >/dev/null && curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
  printf "Remove the downloaded Ollama models (%s)? [y/N]: " "${MODELS[*]}"
  read -r A
  if [ "$A" = "y" ] || [ "$A" = "Y" ]; then
    for m in "${MODELS[@]}"; do
      ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$m" && ollama rm "$m" && echo "  ✓ removed $m"
    done
  fi
fi

# 4. speech-model cache
HF="$HOME/.cache/huggingface/hub"
if [ -d "$HF" ]; then
  printf "Remove the cached speech models (Parakeet, Kokoro, ~1 GB)? [y/N]: "
  read -r A
  if [ "$A" = "y" ] || [ "$A" = "Y" ]; then
    rm -rf "$HF"/models--mlx-community--parakeet* "$HF"/models--prince-canuma--Kokoro* \
           "$HF"/models--openai--whisper* "$HF"/models--mlx-community--silero* \
           "$HF"/models--mlx-community--Qwen3-ASR* 2>/dev/null || true
    echo "  ✓ removed cached speech models"
  fi
fi

echo
echo "✦ Done. To finish, delete this folder:  rm -rf \"$HERE\""
echo
