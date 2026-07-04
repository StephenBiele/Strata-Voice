#!/usr/bin/env python3
"""Strata Voice — the native macOS shell.

A WKWebView window around the local server: a real window and Dock presence
instead of a browser tab. Starts Ollama and the voice server if they aren't
already running, and on quit shuts down only what it started — a server you
launched yourself in a terminal is left alone.

Built into "Strata Voice.app" by make_app.sh (which bakes in the repo path).
"""
import atexit
import os
import socket
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("VOICE_PORT", "8765"))
OLLAMA_PORT = 11434
URL = f"http://127.0.0.1:{PORT}"
LOG = Path.home() / ".vui" / "app-server.log"

_spawned: list = []


def _listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait(port: int, secs: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < secs:
        if _listening(port):
            return True
        time.sleep(0.5)
    return False


def _ensure_stack() -> None:
    """Bring up Ollama + the voice server if they aren't already running."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "a")
    if not _listening(OLLAMA_PORT):
        try:
            _spawned.append(subprocess.Popen(["ollama", "serve"],
                                             stdout=log, stderr=log))
            _wait(OLLAMA_PORT, 15)
        except FileNotFoundError:
            pass  # no Ollama — the UI's model settings will say so
    if not _listening(PORT):
        _spawned.append(subprocess.Popen(
            [str(HERE / ".venv/bin/python"), str(HERE / "server.py")],
            cwd=HERE, stdout=log, stderr=log))
        # first launch downloads models — give it a while, then open anyway
        _wait(PORT, 180)


@atexit.register
def _cleanup() -> None:
    for p in _spawned:
        try:
            p.terminate()
        except Exception:
            pass


def _allow_mic() -> None:
    """WKWebView asks its UI delegate before a page may use the microphone;
    pywebview doesn't implement that hook, so hands-free capture would silently
    fail. Grant it for our own UI — the macOS microphone permission prompt
    (NSMicrophoneUsageDescription) still gates actual access."""
    import objc
    from webview.platforms.cocoa import BrowserView

    def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
            self, wv, origin, frame, mtype, handler):
        handler(1)  # WKPermissionDecisionGrant

    objc.classAddMethods(
        BrowserView.BrowserDelegate,
        [webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_])


if __name__ == "__main__":
    _ensure_stack()
    import webview
    _allow_mic()
    webview.create_window("Strata Voice", URL,
                          width=1280, height=860, min_size=(900, 600))
    webview.start()
