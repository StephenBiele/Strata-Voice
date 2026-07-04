#!/bin/sh
# Data-safety test runner (docs/DATA-SAFETY.md). Spins a THROWAWAY server on a
# spare port with a THROWAWAY database — the real ~/.vui is never touched.
set -e
cd "$(dirname "$0")/.."
PORT=8809
TMP="$(mktemp -d)"
export VOICE_DB="$TMP/strata.db"

VOICE_PORT=$PORT VOICE_VAD_PORT=8810 .venv/bin/python -u server.py > "$TMP/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT

i=0
until curl -s -m2 "http://127.0.0.1:$PORT/config" >/dev/null 2>&1; do
  i=$((i+1)); [ $i -gt 120 ] && { echo "server never came up"; tail -5 "$TMP/server.log"; exit 1; }
  sleep 1
done

TEST_PORT=$PORT .venv/bin/python tests/data_safety_check.py
