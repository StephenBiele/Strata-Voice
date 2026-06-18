"""Web server for the voice assistant — a lightweight, reliable call UI.

Serves a single-page front-end (index.html) and the turn endpoint plus a
small API for past chats, memories, and reference documents. The browser
captures mic audio while the user holds Space (push-to-talk), encodes a
16 kHz WAV in-page, and POSTs it here. We run the same pipeline as the CLI
— Parakeet (ASR) -> Ollama (LLM) -> Kokoro (TTS) — backed by Strata Memory,
and return the spoken reply as a WAV the page plays back.

No streaming / WebRTC / VAD: turns are explicit, which is what makes it
feel instant and never mis-fire. Single-threaded on purpose — MLX's Metal
GPU stream lives in the thread that loaded the models (this one).

Run:  python server.py     then open http://localhost:8765
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import keyring
import numpy as np
import soundfile as sf

import voicechat as vc  # reuse llm_reply / apply_directives / list_memories

HOST = os.environ.get("VOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("VOICE_PORT", "8765"))
ASSISTANT_NAME = os.environ.get("VOICE_NAME", "Sage")
HERE = Path(__file__).parent

STORE = Path(vc.DB_PATH).parent / "voicechat"
SESSIONS_FILE = STORE / "sessions.json"
DOCS_FILE = STORE / "documents.json"
PROFILE_FILE = STORE / "profile.json"
PROFILE_FIELDS = ("name", "preferred_name", "location", "gender")
SETTINGS_FILE = STORE / "settings.json"

# API keys are NEVER written to disk — they live in the OS keychain.
KEYRING_SERVICE = "voicechat"
KEYRING_ACCOUNT = "openai_api_key"

DEFAULT_SETTINGS = {
    "assistant_name": ASSISTANT_NAME,
    "persona": vc.PERSONA_PROMPT,
    "thinking": False,
    "backend": "ollama",                 # 'ollama' | 'openai'
    "ollama_url": vc.OLLAMA_URL,
    "ollama_model": vc.LLM_MODEL,
    "openai_base": "",                   # e.g. https://api.openai.com/v1
    "openai_model": "",
    "configured": False,                 # model setup completed in onboarding
}
# What a POST /settings is allowed to write (api_key handled separately).
SETTINGS_FIELDS = (
    "assistant_name", "persona", "thinking", "backend",
    "ollama_url", "ollama_model", "openai_base", "openai_model", "configured",
)
DOC_CHAR_CAP = 8000          # per-document text injected into the prompt
DOC_TOTAL_CAP = 16000        # across all docs

# Loaded once at startup.
_asr = None
_tts = None
_strata = None
_lock = threading.Lock()

_history: list[dict] = []
_session: dict | None = None   # the in-progress call


# ---- persistence -------------------------------------------------------------
def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---- settings + secure API key ----------------------------------------------
def _settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    s.update(_read_json(SETTINGS_FILE, {}))
    return s


def _save_settings(partial: dict) -> dict:
    s = _settings()
    for k in SETTINGS_FIELDS:
        if k in partial:
            v = partial[k]
            s[k] = bool(v) if k in ("thinking", "configured") else v
    _write_json(SETTINGS_FILE, {k: s[k] for k in SETTINGS_FIELDS})
    return s


def _get_api_key() -> str | None:
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
    except Exception as e:
        print(f"[keyring] read failed: {e}")
        return None


def _set_api_key(key: str) -> None:
    try:
        if key:
            keyring.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, key)
        else:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception as e:
        print(f"[keyring] write failed: {e}")


def _llm_cfg(overrides: dict | None = None) -> dict:
    """Resolve the LLM config the way llm_complete expects it, pulling the
    API key from the keychain. `overrides` lets /settings/test try unsaved
    values (including a freshly-typed api_key) without persisting them."""
    s = _settings()
    o = overrides or {}
    cfg = {
        "backend": o.get("backend", s["backend"]),
        "thinking": o.get("thinking", s["thinking"]),
        "ollama_url": o.get("ollama_url", s["ollama_url"]),
        "ollama_model": o.get("ollama_model", s["ollama_model"]),
        "openai_base": o.get("openai_base", s["openai_base"]),
        "openai_model": o.get("openai_model", s["openai_model"]),
    }
    cfg["api_key"] = o.get("api_key") or _get_api_key() or ""
    return cfg


def _load_models() -> None:
    global _asr, _tts, _strata
    print("Loading models (first run downloads them)…")
    import parakeet_mlx
    from mlx_audio.tts.utils import load_model
    from strata.gateway.api import Strata

    STORE.mkdir(parents=True, exist_ok=True)
    _asr = parakeet_mlx.from_pretrained(vc.ASR_MODEL)
    _tts = load_model(vc.TTS_MODEL)
    Path(vc.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    _strata = Strata.open(db_path=vc.DB_PATH)
    print(f"Ready · LLM={vc.LLM_MODEL} · ASR=Parakeet-V3 · TTS=Kokoro · DB={vc.DB_PATH}")


# ---- speech synthesis --------------------------------------------------------
def _synth(text: str) -> np.ndarray:
    """Synthesize speech, one sentence at a time.

    Kokoro-MLX has a broadcast bug in its harmonic source generator that
    trips on long single segments. Splitting into sentences keeps each
    generate() call short enough to avoid it; a sentence that still fails is
    skipped rather than killing the whole turn.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    gap = np.zeros(int(vc.TTS_SR * 0.12), dtype=np.float32)
    out: list[np.ndarray] = []
    for part in parts or [text]:
        try:
            segs = list(_tts.generate(part, voice=vc.TTS_VOICE, lang_code="a"))
            for s in segs:
                out.append(np.asarray(s.audio).astype(np.float32))
            out.append(gap)
        except Exception as e:
            print(f"[tts] skipped sentence ({e}): {part!r}")
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out)


def _tts_wav_b64(text: str) -> str:
    audio = _synth(text)
    if audio.size == 0:
        return ""
    buf = io.BytesIO()
    sf.write(buf, audio, vc.TTS_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---- documents ---------------------------------------------------------------
def _extract_text(name: str, data: bytes) -> str:
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    if ext == "docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs).strip()
    # txt / md / csv / json / anything text-ish
    return data.decode("utf-8", errors="ignore").strip()


def _profile_context() -> str:
    p = _read_json(PROFILE_FILE, {})
    labels = {
        "name": "Full name",
        "preferred_name": "Prefers to be called",
        "location": "Location",
        "gender": "Gender",
    }
    lines = [f"- {labels[k]}: {p[k]}" for k in PROFILE_FIELDS if p.get(k)]
    return "USER PROFILE (address them by their preferred name):\n" + "\n".join(lines) if lines else ""


def _doc_context() -> list[str]:
    docs = _read_json(DOCS_FILE, [])
    out, total = [], 0
    for d in docs:
        txt = (d.get("text") or "")[:DOC_CHAR_CAP]
        if not txt:
            continue
        if total + len(txt) > DOC_TOTAL_CAP:
            txt = txt[: DOC_TOTAL_CAP - total]
        out.append(f"=== {d.get('name','document')} ===\n{txt}")
        total += len(txt)
        if total >= DOC_TOTAL_CAP:
            break
    return out


# ---- sessions ----------------------------------------------------------------
def _start_session() -> None:
    global _session, _history
    _finalize_session()
    _history = []
    _session = {
        "id": str(int(time.time() * 1000)),
        "started_at": time.time(),
        "ended_at": None,
        "title": None,
        "turns": [],
    }


def _finalize_session() -> None:
    global _session
    if _session and _session["turns"]:
        _session["ended_at"] = _session["ended_at"] or time.time()
        sessions = _read_json(SESSIONS_FILE, [])
        sessions = [s for s in sessions if s["id"] != _session["id"]]
        sessions.append(_session)
        _write_json(SESSIONS_FILE, sessions)
    _session = None


def _persist_session() -> None:
    if not _session:
        return
    sessions = _read_json(SESSIONS_FILE, [])
    sessions = [s for s in sessions if s["id"] != _session["id"]]
    sessions.append(_session)
    _write_json(SESSIONS_FILE, sessions)


# ---- turn --------------------------------------------------------------------
def _handle_turn(wav_bytes: bytes) -> dict:
    with _lock:
        tmp = HERE / ".turn_in.wav"
        tmp.write_bytes(wav_bytes)
        text = _asr.transcribe(str(tmp)).text.strip()
        if not text:
            return {"ok": True, "empty": True}

        if _session is None:
            _start_session()

        _history.append({"role": "user", "content": text})
        mem = vc.list_memories(_strata)
        s = _settings()
        reply_raw = vc.llm_reply(
            _history, [m["text"] for m in mem],
            documents=_doc_context(), profile=_profile_context(),
            persona=s["persona"], cfg=_llm_cfg(),
        )
        reply = vc.apply_directives(_strata, reply_raw, mem)
        _history.append({"role": "assistant", "content": reply})

        # transcript
        _session["turns"].append({"role": "user", "content": text, "t": time.time()})
        _session["turns"].append({"role": "assistant", "content": reply, "t": time.time()})
        if not _session["title"]:
            _session["title"] = text[:60]
        _persist_session()

        audio_b64 = _tts_wav_b64(reply) if reply else ""
        memories = [m["text"] for m in vc.list_memories(_strata)]
        return {
            "ok": True,
            "transcript": text,
            "reply": reply,
            "audio": audio_b64,
            "memories": memories,
        }


# ---- HTTP --------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj))

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n) if n else b""

    # -- GET --
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = (HERE / "index.html").read_text(encoding="utf-8")
            html = html.replace("{{ASSISTANT_NAME}}", _settings()["assistant_name"])
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/config":
            return self._json(200, {"name": _settings()["assistant_name"]})
        if u.path == "/settings":
            s = _settings()
            out = {k: s[k] for k in SETTINGS_FIELDS}
            out["has_api_key"] = bool(_get_api_key())
            out["persona_default"] = vc.PERSONA_PROMPT
            return self._json(200, out)
        if u.path == "/ollama/models":
            url = (parse_qs(u.query).get("url", [None])[0] or _settings()["ollama_url"]).rstrip("/")
            try:
                import httpx
                r = httpx.get(f"{url}/api/tags", timeout=5)
                names = [m["name"] for m in r.json().get("models", []) if m.get("name")]
                return self._json(200, {"ok": True, "models": sorted(names)})
            except Exception as e:
                return self._json(200, {"ok": False, "error": str(e), "models": []})
        if u.path == "/memories":
            with _lock:
                mem = vc.list_memories(_strata)
            return self._json(200, {"memories": mem})
        if u.path == "/profile":
            p = _read_json(PROFILE_FILE, {})
            # Treat as onboarded if they finished onboarding OR any real info is
            # already entered — so the welcome never re-appears once it's filled.
            return self._json(200, {
                "profile": {k: p.get(k, "") for k in PROFILE_FIELDS},
                "onboarded": bool(p.get("onboarded") or p.get("name")),
            })
        if u.path == "/sessions":
            sessions = _read_json(SESSIONS_FILE, [])
            out = [{
                "id": s["id"], "title": s.get("title") or "Untitled",
                "started_at": s["started_at"], "ended_at": s.get("ended_at"),
                "turns": len(s["turns"]),
            } for s in sessions if s["turns"]]
            out.sort(key=lambda s: s["started_at"], reverse=True)
            return self._json(200, {"sessions": out})
        if u.path == "/session":
            sid = parse_qs(u.query).get("id", [""])[0]
            sessions = _read_json(SESSIONS_FILE, [])
            s = next((x for x in sessions if x["id"] == sid), None)
            return self._json(200 if s else 404, s or {"error": "not found"})
        if u.path == "/documents":
            docs = _read_json(DOCS_FILE, [])
            out = [{"id": d["id"], "name": d["name"], "chars": len(d.get("text", "")),
                    "added_at": d.get("added_at")} for d in docs]
            return self._json(200, {"documents": out})
        return self._json(404, {"ok": False, "error": "not found"})

    # -- POST --
    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/reset":
            with _lock:
                _start_session()
            return self._json(200, {"ok": True})
        if u.path == "/end":
            with _lock:
                _finalize_session()
            return self._json(200, {"ok": True})
        if u.path == "/profile":
            body = json.loads(self._body() or b"{}")
            p = _read_json(PROFILE_FILE, {})
            for k in PROFILE_FIELDS:
                if k in body:
                    p[k] = str(body[k]).strip()
            p["onboarded"] = True
            _write_json(PROFILE_FILE, p)
            return self._json(200, {"ok": True})
        if u.path == "/settings":
            body = json.loads(self._body() or b"{}")
            # API key → keychain only; never persisted to settings.json.
            if "api_key" in body:
                _set_api_key((body.get("api_key") or "").strip())
            _save_settings(body)
            return self._json(200, {"ok": True})
        if u.path == "/settings/test":
            body = json.loads(self._body() or b"{}")
            try:
                cfg = _llm_cfg(body)
                msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
                reply = vc.llm_complete(msgs, cfg)
                return self._json(200, {"ok": True, "reply": reply[:80]})
            except Exception as e:
                return self._json(200, {"ok": False, "error": str(e)})
        if u.path == "/turn":
            try:
                result = _handle_turn(self._body())
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._json(200, {"ok": False, "error": str(e)})
            return self._json(200, result)
        if u.path == "/upload":
            name = unquote(self.headers.get("X-Filename", "document"))
            data = self._body()
            try:
                text = _extract_text(name, data)
            except Exception as e:
                return self._json(200, {"ok": False, "error": f"could not read file: {e}"})
            if not text:
                return self._json(200, {"ok": False, "error": "no text found in file"})
            docs = _read_json(DOCS_FILE, [])
            doc = {"id": str(int(time.time() * 1000)), "name": name,
                   "text": text, "added_at": time.time()}
            docs.append(doc)
            _write_json(DOCS_FILE, docs)
            return self._json(200, {"ok": True, "id": doc["id"], "name": name,
                                    "chars": len(text)})
        if u.path == "/document/delete":
            body = json.loads(self._body() or b"{}")
            did = str(body.get("id", ""))
            docs = [d for d in _read_json(DOCS_FILE, []) if d["id"] != did]
            _write_json(DOCS_FILE, docs)
            return self._json(200, {"ok": True})
        if u.path == "/memory/delete":
            body = json.loads(self._body() or b"{}")
            mid = body.get("id")
            with _lock:
                try:
                    _strata.delete_memory(int(mid), mode="hard")
                    ok = True
                except Exception as e:
                    return self._json(200, {"ok": False, "error": str(e)})
            return self._json(200, {"ok": ok})
        if u.path == "/session/delete":
            body = json.loads(self._body() or b"{}")
            sid = str(body.get("id", ""))
            sessions = [s for s in _read_json(SESSIONS_FILE, []) if s["id"] != sid]
            _write_json(SESSIONS_FILE, sessions)
            return self._json(200, {"ok": True})
        return self._json(404, {"ok": False, "error": "not found"})


def main() -> int:
    _load_models()
    srv = HTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"\n  ✦ {ASSISTANT_NAME} is listening at {url}")
    print("    Open it in your browser, click Start, hold Space to talk.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        _finalize_session()
        if _strata is not None:
            _strata.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
