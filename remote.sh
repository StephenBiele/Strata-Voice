#!/bin/sh
# Remote access over Tailscale: use Strata Voice from your phone, anywhere.
#   ./remote.sh on      share it on your tailnet (private, encrypted, HTTPS)
#   ./remote.sh off     stop sharing
#   ./remote.sh status  show the current sharing state
#
# Requires the Tailscale app (tailscale.com) signed in on this Mac and on your
# phone, with both on the same tailnet. Never uses Funnel — this stays private
# to your tailnet. See docs/REMOTE-ACCESS.md for the phone setup.
set -e

TS="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
command -v tailscale >/dev/null 2>&1 && TS="tailscale"
[ -x "$TS" ] || command -v "$TS" >/dev/null 2>&1 || {
  echo "Tailscale isn't installed — get it from https://tailscale.com/download"; exit 1; }

url(){
  "$TS" status --json 2>/dev/null | /usr/bin/python3 -c \
    'import json,sys; d=json.load(sys.stdin); print("https://"+d["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true
}

case "${1:-on}" in
  on)
    # main app at the root; the hands-free VAD channel rides /vadsvc
    "$TS" serve --bg --set-path /vadsvc http://127.0.0.1:8766 >/dev/null
    "$TS" serve --bg http://127.0.0.1:8765 >/dev/null
    U="$(url)"
    echo "✓ sharing on your tailnet (private)"
    echo "  on your phone (with Tailscale connected): $U"
    echo "  add it to your home screen for a full-screen app with the icon"
    echo "  note: the Mac must be awake — and keep the server running"
    ;;
  off)
    "$TS" serve reset
    echo "✓ stopped sharing"
    ;;
  status)
    "$TS" serve status
    ;;
  *)
    echo "usage: ./remote.sh [on|off|status]"; exit 1
    ;;
esac
