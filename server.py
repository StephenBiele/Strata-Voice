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
    "tts_voice": "af_heart",
    "tts_speed": 1.0,
}
# What a POST /settings is allowed to write (api_key handled separately).
SETTINGS_FIELDS = (
    "assistant_name", "persona", "thinking", "backend",
    "ollama_url", "ollama_model", "openai_base", "openai_model", "configured",
    "tts_voice", "tts_speed",
)

# English Kokoro voices (American 'a' / British 'b'). The first letter is also
# Kokoro's lang_code, so voice[0] routes pronunciation. Non-English packs are
# omitted because they mangle English text.
_VOICE_NAMES = {
    "af_heart": "Heart", "af_bella": "Bella", "af_nicole": "Nicole",
    "af_sarah": "Sarah", "af_sky": "Sky", "af_aoede": "Aoede", "af_kore": "Kore",
    "af_nova": "Nova", "af_alloy": "Alloy", "af_jessica": "Jessica", "af_river": "River",
    "am_adam": "Adam", "am_michael": "Michael", "am_echo": "Echo", "am_eric": "Eric",
    "am_fenrir": "Fenrir", "am_liam": "Liam", "am_onyx": "Onyx", "am_puck": "Puck",
    "bf_emma": "Emma", "bf_isabella": "Isabella", "bf_alice": "Alice", "bf_lily": "Lily",
    "bm_george": "George", "bm_lewis": "Lewis", "bm_daniel": "Daniel", "bm_fable": "Fable",
}


def _voice_list() -> list[dict]:
    out = []
    for vid, name in _VOICE_NAMES.items():
        accent = "British" if vid[0] == "b" else "American"
        gender = "female" if vid[1] == "f" else "male"
        out.append({"id": vid, "label": f"{name} · {accent} {gender}",
                    "accent": accent, "gender": gender})
    return out


def _lang_code(voice: str) -> str:
    return "b" if voice[:1] == "b" else "a"
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
    vc.patch_kokoro_tts()
    Path(vc.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    _strata = Strata.open(db_path=vc.DB_PATH)
    print(f"Ready · LLM={vc.LLM_MODEL} · ASR=Parakeet-V3 · TTS=Kokoro · DB={vc.DB_PATH}")


# ---- speech synthesis --------------------------------------------------------
def _for_speech(text: str) -> str:
    """Smooth punctuation that makes Kokoro's cadence choppy: dashes and
    semicolons become clean clause breaks, ellipses collapse, and runs of
    commas/spaces are tidied. Only affects what's spoken, not what's stored."""
    text = re.sub(r"\s*[—–]\s*", ", ", text)   # em/en dash -> comma pause
    text = text.replace(";", ",").replace("…", ".")
    text = re.sub(r"\.{2,}", ".", text)          # ellipsis -> single stop
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)  # collapse repeated commas
    text = re.sub(r"\s+([,.!?])", r"\1", text)    # no space before punctuation
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _synth_sentence(text: str, voice: str, speed: float) -> np.ndarray:
    """Synthesize one short segment. Returns float32 @ TTS_SR (empty on failure)."""
    text = _for_speech(text)
    if not text:
        return np.zeros(0, dtype=np.float32)
    try:
        segs = list(_tts.generate(text, voice=voice, speed=speed,
                                  lang_code=_lang_code(voice)))
        if segs:
            return np.concatenate([np.asarray(s.audio) for s in segs]).astype(np.float32)
    except Exception as e:
        print(f"[tts] skipped sentence ({e}): {text!r}")
    return np.zeros(0, dtype=np.float32)


def _synth(text: str, voice: str | None = None, speed: float | None = None) -> np.ndarray:
    """Synthesize speech, one sentence at a time.

    Kokoro-MLX has a broadcast bug in its harmonic source generator that
    trips on long single segments. Splitting into sentences keeps each
    generate() call short enough to avoid it; a sentence that still fails is
    skipped rather than killing the whole turn.
    """
    s = _settings()
    voice = voice or s["tts_voice"]
    speed = speed if speed is not None else float(s["tts_speed"])
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    gap = np.zeros(int(vc.TTS_SR * 0.12), dtype=np.float32)
    out: list[np.ndarray] = []
    for part in parts or [text]:
        audio = _synth_sentence(part, voice, speed)
        if audio.size:
            out.append(audio)
            out.append(gap)
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


def _audio_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, vc.TTS_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENT_BOUNDARY = re.compile(r'[.!?]+["\')\]]*(\s|$)')


def _spoken_region(full: str) -> str:
    """The part of the streamed text that should be spoken: everything before
    the first memory directive, with any <think> reasoning removed (including
    a not-yet-closed one)."""
    m = re.search(r"\[MEM", full)
    t = full[:m.start()] if m else full
    t = _THINK_TAG.sub("", t)
    i = t.lower().find("<think>")
    if i != -1:
        t = t[:i]
    return t


def _pop_sentences(spoken: str, emitted: int) -> tuple[list[str], int]:
    """Pull complete sentences from spoken[emitted:]; return (sentences, new_emitted)."""
    out: list[str] = []
    region = spoken[emitted:]
    while True:
        m = _SENT_BOUNDARY.search(region)
        if not m:
            break
        end = m.end()
        seg = region[:end].strip()
        if seg:
            out.append(seg)
        emitted += end
        region = region[end:]
    return out, emitted


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
    return ("USER PROFILE (background facts — draw on these only when the user's "
            "message actually calls for it; do not insert their name or location "
            "into replies where it isn't needed):\n" + "\n".join(lines)) if lines else ""


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
        captured = vc.capture_memory(_strata, text)   # deterministic, no LLM
        if captured:
            print("[memory]", captured)
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


def _handle_turn_stream(wav_bytes: bytes):
    """Streaming turn: yields NDJSON-ready dicts. Synthesizes and ships audio
    sentence-by-sentence as the LLM streams, so playback starts much sooner."""
    with _lock:
        tmp = HERE / ".turn_in.wav"
        tmp.write_bytes(wav_bytes)
        text = _asr.transcribe(str(tmp)).text.strip()
        if not text:
            yield {"type": "empty"}
            return
        yield {"type": "meta", "transcript": text}

        if _session is None:
            _start_session()
        _history.append({"role": "user", "content": text})
        captured = vc.capture_memory(_strata, text)   # deterministic, no LLM
        if captured:
            print("[memory]", captured)

        s = _settings()
        mem = vc.list_memories(_strata)
        voice, speed = s["tts_voice"], float(s["tts_speed"])
        messages = vc.build_messages(
            _history, [m["text"] for m in mem],
            documents=_doc_context(), profile=_profile_context(), persona=s["persona"],
        )

        full, emitted, seq = "", 0, 0
        try:
            for tok in vc.llm_stream(messages, _llm_cfg()):
                full += tok
                sents, emitted = _pop_sentences(_spoken_region(full), emitted)
                for seg in sents:
                    audio = _synth_sentence(seg, voice, speed)
                    if audio.size:
                        seq += 1
                        yield {"type": "audio", "seq": seq, "text": seg, "audio": _audio_b64(audio)}
        except Exception as e:
            print(f"[stream] llm error: {e}")
            yield {"type": "error", "error": str(e)}
            return

        tail = _spoken_region(full)[emitted:].strip()
        if tail:
            audio = _synth_sentence(tail, voice, speed)
            if audio.size:
                seq += 1
                yield {"type": "audio", "seq": seq, "text": tail, "audio": _audio_b64(audio)}

        # parse directives from the full reply, persist memory + transcript
        reply = vc.apply_directives(_strata, full, mem)
        _history.append({"role": "assistant", "content": reply})
        _session["turns"].append({"role": "user", "content": text, "t": time.time()})
        _session["turns"].append({"role": "assistant", "content": reply, "t": time.time()})
        if not _session["title"]:
            _session["title"] = text[:60]
        _persist_session()

        # LLM extraction pass — captures durable facts from natural speech that
        # the deterministic patterns miss. Runs after the reply is fully streamed,
        # so it never delays the spoken response.
        try:
            cur = [m["text"] for m in vc.list_memories(_strata)]
            new_facts = vc.extract_facts_llm(text, cur, _llm_cfg())
            added = vc.add_facts(_strata, new_facts)
            if added:
                print("[memory] extracted:", added)
        except Exception as e:
            print("[memory] extraction error:", e)

        memories = [m["text"] for m in vc.list_memories(_strata)]
        yield {"type": "done", "reply": reply, "memories": memories}


# ---- recall eval -------------------------------------------------------------
def _run_eval() -> dict:
    """Probe the live model's recall using the user's real profile + memories.
    Read-only: builds one-off messages, never touches history or the store."""
    profile = _read_json(PROFILE_FILE, {})
    mems = [m["text"] for m in vc.list_memories(_strata)]
    if not profile.get("name") and not profile.get("location") and not mems:
        return {"empty": True, "score": None, "probes": [], "coverage": None}

    s = _settings()
    cfg = _llm_cfg()

    def ask(question: str) -> str:
        msgs = vc.build_messages(
            [{"role": "user", "content": question}], mems,
            documents=_doc_context(), profile=_profile_context(), persona=s["persona"])
        return vc.llm_complete(msgs, cfg)

    def hit(answer: str, kw: str) -> bool:
        return bool(kw) and kw.lower() in (answer or "").lower()

    probes = []
    # Deterministic, non-leading probes for the core profile facts.
    name_exp = [x for x in (profile.get("preferred_name"), profile.get("name")) if x]
    if name_exp:
        a = ask("What's my name?")
        probes.append({"label": "Name", "question": "What's my name?",
                       "answer": a, "pass": any(hit(a, n) for n in name_exp)})
    if profile.get("location"):
        tok = profile["location"].replace(",", " ").split()[0]
        a = ask("Where do I live?")
        probes.append({"label": "Location", "question": "Where do I live?",
                       "answer": a, "pass": hit(a, tok)})

    # Coverage: one open recall question, scored by a semantic judge so entailment
    # counts (saying "Colorado" recalls "based in the United States").
    coverage = None
    if mems:
        q = "What do you know about me? Tell me everything you remember."
        a = ask(q)
        verdicts = vc.judge_recall(mems, a, cfg)
        results = [{"fact": f, "hit": v} for f, v in zip(mems, verdicts)]
        coverage = {"question": q, "answer": a, "results": results,
                    "hits": sum(r["hit"] for r in results), "total": len(results)}

    passed = sum(p["pass"] for p in probes) + (coverage["hits"] if coverage else 0)
    total = len(probes) + (coverage["total"] if coverage else 0)
    return {"empty": False, "probes": probes, "coverage": coverage,
            "score": round(100 * passed / total) if total else None}


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
        if u.path == "/voices":
            return self._json(200, {"voices": _voice_list()})
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
            # ids are 64-bit ints; stringify so JS doesn't round them past 2^53.
            mem = [{"id": str(m["id"]), "text": m["text"]} for m in mem]
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
            cur_id = _session["id"] if _session else None
            out = [{
                "id": s["id"], "title": s.get("title") or "Untitled",
                "started_at": s["started_at"], "ended_at": s.get("ended_at"),
                "turns": len(s["turns"]),
                "current": s["id"] == cur_id,
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
        if u.path == "/tts/preview":
            body = json.loads(self._body() or b"{}")
            voice = body.get("voice") or _settings()["tts_voice"]
            speed = float(body.get("speed", _settings()["tts_speed"]))
            sample = body.get("text") or "Hi, this is how I sound. I'm ready when you are."
            with _lock:
                audio = _synth(sample, voice=voice, speed=speed)
            if audio.size == 0:
                return self._json(200, {"ok": False, "error": "synthesis failed"})
            buf = io.BytesIO()
            sf.write(buf, audio, vc.TTS_SR, format="WAV", subtype="PCM_16")
            return self._json(200, {"ok": True,
                                    "audio": base64.b64encode(buf.getvalue()).decode("ascii")})
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
        if u.path == "/turn/stream":
            wav = self._body()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for line in _handle_turn_stream(wav):
                    self.wfile.write((json.dumps(line) + "\n").encode("utf-8"))
                    self.wfile.flush()
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self.wfile.write((json.dumps({"type": "error", "error": str(e)}) + "\n").encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass
            return
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
        if u.path == "/session/review":
            body = json.loads(self._body() or b"{}")
            sid = str(body.get("id", ""))
            sess = next((s for s in _read_json(SESSIONS_FILE, []) if s["id"] == sid), None)
            if not sess:
                return self._json(200, {"ok": False, "error": "conversation not found"})
            with _lock:
                try:
                    result = vc.review_session(_strata, sess.get("turns", []), _llm_cfg())
                except Exception as e:
                    return self._json(200, {"ok": False, "error": str(e)})
            return self._json(200, {"ok": True, **result})
        if u.path == "/memory/audit":
            with _lock:
                try:
                    issues = vc.audit_memories(_strata, _llm_cfg())
                except Exception as e:
                    return self._json(200, {"ok": False, "error": str(e)})
            return self._json(200, {"ok": True, "issues": issues})
        if u.path == "/memory/resolve":
            body = json.loads(self._body() or b"{}")
            ids = body.get("remove") or []
            removed = 0
            with _lock:
                for mid in ids:
                    try:
                        _strata.delete_memory(int(mid), mode="hard")
                        removed += 1
                    except Exception as e:
                        print(f"[audit] resolve delete failed for {mid}: {e}")
            return self._json(200, {"ok": True, "removed": removed})
        if u.path == "/eval/run":
            with _lock:
                try:
                    result = _run_eval()
                except Exception as e:
                    import traceback; traceback.print_exc()
                    return self._json(200, {"ok": False, "error": str(e)})
            return self._json(200, {"ok": True, **result})
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
