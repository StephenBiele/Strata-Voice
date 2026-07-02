#!/bin/bash
# Strata Voice — start the app and open it in your browser.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PORT="${VOICE_PORT:-8765}"
URL="http://localhost:$PORT"

[ -x .venv/bin/python ] || { echo "✗ Not installed yet — run ./install.sh first."; exit 1; }

# Ollama must be serving for replies (and semantic recall)
if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
  echo "· starting Ollama…"
  (ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 30); do
    sleep 1
    curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null && break
  done
fi

# Free the port only if it's held by a previous instance of THIS app.
STALE=$(lsof -ti ":$PORT" 2>/dev/null || true)
if [ -n "$STALE" ]; then
  for pid in $STALE; do
    if ps -p "$pid" -o command= | grep -q "server\.py"; then
      echo "· stopping previous Strata Voice server (pid $pid)"
      kill "$pid" 2>/dev/null || true
    else
      echo "✗ Port $PORT is in use by another program (pid $pid). Stop it, or run: VOICE_PORT=8770 ./start.sh"
      exit 1
    fi
  done
  sleep 1
fi

echo "· starting Strata Voice… (first run downloads the speech models, ~1 GB)"
.venv/bin/python server.py &
SERVER_PID=$!

# open the browser once the server answers
for _ in $(seq 1 120); do
  sleep 1
  if curl -s --max-time 2 "$URL/config" >/dev/null; then
    open "$URL"
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "✗ Server exited during startup — scroll up for the error."; exit 1; }
done

wait "$SERVER_PID"
