#!/bin/sh
# Build "Strata Voice.app" — the Dock-able native shell around app.py.
# The bundle bakes in this repo's absolute path, so rebuild after moving the
# folder. Safe to re-run any time; the app can be copied to /Applications or
# kept here and dragged to the Dock.
set -e
cd "$(dirname "$0")"
REPO="$(pwd)"
APP="Strata Voice.app"

[ -x .venv/bin/python ] || { echo "No .venv — run ./install.sh first."; exit 1; }
.venv/bin/python -c "import webview" 2>/dev/null || {
  echo "Installing the native window library (pywebview)…"
  .venv/bin/pip install --quiet -r requirements.txt
}

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp icons/StrataVoice.icns "$APP/Contents/Resources/StrataVoice.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Strata Voice</string>
  <key>CFBundleDisplayName</key><string>Strata Voice</string>
  <key>CFBundleIdentifier</key><string>com.stratavoice.app</string>
  <key>CFBundleExecutable</key><string>StrataVoice</string>
  <key>CFBundleIconFile</key><string>StrataVoice</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Strata Voice listens so you can talk to it. Audio is processed on this Mac and never leaves it.</string>
</dict>
</plist>
PLIST

# macOS reads app identity (name, icon, the NSMicrophoneUsageDescription that
# makes the mic prompt possible) from the bundle of the RUNNING EXECUTABLE.
# exec'ing the venv python from a script leaves identity with Python.app and
# WKWebView denies the mic without ever prompting — so copy the real
# interpreter INSIDE the bundle and point it at the venv via PYTHONPATH.
PYHOME="$(.venv/bin/python -c 'import sys;print(sys.base_prefix)')"
PYVER="$(.venv/bin/python -c 'import sys;print("python%d.%d"%sys.version_info[:2])')"
# framework builds ship a stub that re-execs Resources/Python.app/…/Python —
# copy that FINAL binary or the re-exec hands identity back to Python.app
PYBIN="$PYHOME/Resources/Python.app/Contents/MacOS/Python"
[ -f "$PYBIN" ] || PYBIN="$(.venv/bin/python -c 'import os,sys;print(os.path.realpath(sys.executable))')"
cp "$PYBIN" "$APP/Contents/MacOS/python"
chmod +x "$APP/Contents/MacOS/python"

cat > "$APP/Contents/MacOS/StrataVoice" <<LAUNCH
#!/bin/sh
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
export PYTHONHOME="$PYHOME"
export PYTHONPATH="$REPO/.venv/lib/$PYVER/site-packages"
cd "$REPO"
exec "\$DIR/python" "$REPO/app.py"
LAUNCH
chmod +x "$APP/Contents/MacOS/StrataVoice"

echo "✓ built $APP"
echo "  · open it:            open \"$APP\""
echo "  · keep it handy:      drag it to the Dock, or copy to /Applications"
echo "  · moved the repo?     just run ./make_app.sh again"
