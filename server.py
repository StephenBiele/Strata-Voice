"""Web server for the voice assistant — a lightweight, reliable call UI.

Serves a single-page front-end (index.html), the streaming turn endpoints,
and a small API for past chats, memories, and reference documents. The
browser captures mic audio hands-free (speech start/stop detected via the
Silero VAD channel on :VAD_PORT), encodes a 16 kHz WAV in-page, and POSTs
it to /turn/stream. Pipeline: Parakeet (ASR) -> LLM (Ollama or any
OpenAI-compatible API) -> Kokoro or Chatterbox (TTS), backed by Strata
Memory; the reply streams back as NDJSON with per-sentence audio chunks.

The main server is single-threaded on purpose — MLX's Metal GPU stream
lives in the thread that loaded the models (this one). Anything slow and
non-conversational (memory writes, recaps, fact harvest, VAD) runs on
background threads or the separate VAD port so it never blocks a turn.

Run:  python server.py     then open http://localhost:8765
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import queue
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
# Hands-free VAD rides its own port + thread: the main server is single-threaded
# and holds _lock for a whole streaming turn, but barge-in detection must keep
# running exactly then. The page (served from :PORT) calls this cross-origin.
VAD_PORT = int(os.environ.get("VOICE_VAD_PORT", "8766"))
ASSISTANT_NAME = os.environ.get("VOICE_NAME", "Sage")
HERE = Path(__file__).parent

STORE = Path(vc.DB_PATH).parent / "voicechat"
SESSIONS_FILE = STORE / "sessions.json"
DOCS_FILE = STORE / "documents.json"
PROFILE_FILE = STORE / "profile.json"
PROFILE_FIELDS = ("name", "preferred_name", "location", "gender")
SETTINGS_FILE = STORE / "settings.json"
# Chatterbox voices are reference clips: bundled presets ship in the repo; the
# user's own cloned voices live outside the repo with their data. A voice is just
# a short audio file the engine mimics.
PRESET_VOICES_DIR = HERE / "voices"
CUSTOM_VOICES_DIR = STORE / "voices"
VOICE_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

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
    # Voice engine: 'kokoro' (fast, neutral) or 'chatterbox' (expressive, performs
    # inline [laugh]/[sigh] tags). tts_emotion lets the model emit those tags when
    # it fits; only meaningful on chatterbox (kokoro strips them either way).
    "tts_engine": "kokoro",              # 'kokoro' | 'chatterbox'
    "tts_emotion": True,                 # allow inline expressive tags (chatterbox only)
    "tts_cb_voice": "builtin",           # chatterbox voice id: builtin | preset:x | custom:x
    # voice cadence (experimental): how the spoken reply is chunked + smoothed
    "tts_chunking": "hybrid",            # 'sentence' | 'clause' | 'hybrid' | 'whole'
    "tts_trim": True,                    # trim edge silence from each chunk
    "tts_gap_ms": 30,                    # consistent gap appended between chunks
    "tts_smoothing": "natural",          # 'natural' | 'verbatim' | 'flowing'
    # LLM generation controls
    "llm_temperature": 0.6,              # 0 = deterministic, higher = more varied
    "llm_top_p": 1.0,                    # nucleus sampling (1 = off)
    "llm_max_tokens": 0,                 # cap on reply length (0 = model default)
    "llm_num_ctx": 0,                    # context window, Ollama only (0 = model default)
    # Speech recognition (what turns your voice into text)
    "asr_model": vc.ASR_MODEL,           # HF model id
    "asr_backend": vc.ASR_BACKEND,       # loader id (only 'mlx-audio' ships)
    # Microphone input device (browser deviceId; "" = system default)
    "mic_device": "",
    # Model for background memory work ("" = same as chat model) — an Ollama
    # model name or a model id on the OpenAI-compatible endpoint, per backend.
    # Memory jobs run behind the conversation, so a bigger, slower model can
    # write more accurate memories without touching voice latency.
    "memory_model": "",
    # Web lookups (off by default: queries leave the machine, so it's opt-in).
    # When on, a quick pre-turn check decides if the question needs fresh info;
    # results live in memory for ~5 minutes and are never written to disk.
    "web_search": False,
    # Hands-free (VAD): talk without holding; it detects speech start/stop
    "vad_enabled": True,                 # hands-free is the default talk mode
    "vad_barge_in": True,                # speaking while it talks interrupts it
    "vad_threshold": 0.5,                # voice sensitivity (higher = stricter)
    "vad_silence_ms": 500,               # pause length that ends your turn
    "vad_prefix_ms": 500,                # lead-in kept from before speech was DETECTED
                                         # (detection can lag soft starts by 300-400ms,
                                         # so a short lead-in clips first words)
    "vad_min_speech_ms": 250,            # ignore blips shorter than this
    "debug_settings": False,             # show the in-call tuning panel
}
# What a POST /settings is allowed to write (api_key handled separately).
SETTINGS_FIELDS = (
    "assistant_name", "persona", "thinking", "backend",
    "ollama_url", "ollama_model", "openai_base", "openai_model", "configured",
    "tts_voice", "tts_speed", "tts_engine", "tts_emotion", "tts_cb_voice",
    "tts_chunking", "tts_trim", "tts_gap_ms", "tts_smoothing",
    "llm_temperature", "llm_top_p", "llm_max_tokens", "llm_num_ctx",
    "asr_model", "asr_backend", "mic_device", "memory_model", "web_search",
    "vad_enabled", "vad_barge_in", "vad_threshold", "vad_silence_ms",
    "vad_prefix_ms", "vad_min_speech_ms", "debug_settings",
)

# Speech-recognition model catalog. Each entry is a (model id + loader backend)
# combo with a plain-language name and one-line blurb, so the picker reads for
# humans. All ids are verified to exist on Hugging Face; a pick that still fails
# to load is caught and reverted (see _reload_asr), so it can't brick a call.
ASR_MODELS = [
    {"id": "parakeet", "model": "mlx-community/parakeet-tdt-0.6b-v3", "backend": "mlx-audio",
     "label": "Parakeet — Balanced (recommended) — 0.6B",
     "blurb": "Fast + accurate for everyday English. Low memory."},
    {"id": "whisper-turbo", "model": "openai/whisper-large-v3-turbo", "backend": "mlx-audio",
     "label": "Whisper Turbo — ~0.8B",
     "blurb": "Strong on accents & other languages. Good accuracy."},
    {"id": "whisper-small", "model": "openai/whisper-small", "backend": "mlx-audio",
     "label": "Whisper Small — 0.24B",
     "blurb": "Lighter multilingual option. Lower memory."},
    {"id": "whisper-tiny", "model": "openai/whisper-tiny", "backend": "mlx-audio",
     "label": "Whisper Tiny — 0.04B",
     "blurb": "Fastest but roughest quality. Smallest footprint."},
    {"id": "qwen3-asr", "model": "mlx-community/Qwen3-ASR-0.6B-4bit", "backend": "mlx-audio",
     "label": "Qwen3-ASR — 0.6B / 1.7B",
     "blurb": "Newest multilingual. Very good speed & accuracy (experimental)."},
]
# Numeric settings and their coercion (so JSON strings from the UI store cleanly).
_FLOAT_FIELDS = {"tts_speed", "llm_temperature", "llm_top_p", "vad_threshold"}
_INT_FIELDS = {"tts_gap_ms", "llm_max_tokens", "llm_num_ctx",
               "vad_silence_ms", "vad_prefix_ms", "vad_min_speech_ms"}

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
DOC_WHOLE_CAP = 2400         # docs this small are injected whole (a short story is fine)
DOC_CHUNK_CHARS = 700        # target passage size when a larger doc is chunked
DOC_CHUNK_OVERLAP = 1        # sentences of overlap carried between passages
DOC_TOP_K = 4                # relevant passages pulled from each large doc

# doc_id -> {"n": <char count when embedded>, "chunks": [(text, vector)]}. In-memory
# only; rebuilt on demand (documents are immutable once uploaded, so keying on id
# + length is safe). Lets us embed each passage once, then score cheaply per turn.
_doc_vec_cache: dict = {}

# Loaded once at startup.
_asr = None
_asr_key = None     # (model, backend) currently loaded into _asr, for change detection
_tts = None
_tts_engine = "kokoro"   # which engine is loaded into _tts, for change detection
_strata = None

# Ephemeral web-search results: in MEMORY only, never disk. Each entry lives
# ~5 minutes so "tell me more" can dig into the same results without a fresh
# search, then it's gone — web content never enters the transcript or memory.
_web_cache: dict = {}          # normalized query -> {"t", "query", "results"}
WEB_TTL_S = 300
WEB_CACHE_MAX = 3              # keep only the last few searches


def _web_remember(query: str, results: list) -> None:
    _web_cache[query.lower().strip()] = {"t": time.time(), "query": query,
                                         "results": results}
    while len(_web_cache) > WEB_CACHE_MAX:            # oldest out first
        oldest = min(_web_cache, key=lambda k: _web_cache[k]["t"])
        del _web_cache[oldest]


def _web_fresh_match(query: str):
    """A still-fresh cache entry asking essentially the same thing, or None.
    The gate rephrases repeats ("store hours today" vs "close today"), so exact
    match isn't enough — word-set overlap catches same-ground queries while
    staying below the level where genuinely new questions would false-match."""
    now = time.time()
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not words:
        return None
    for k, e in _web_cache.items():
        if now - e["t"] > WEB_TTL_S:
            continue
        kw = set(re.findall(r"[a-z0-9]+", k))
        if kw and len(words & kw) / len(words | kw) >= 0.55:
            return e
    return None


def _web_block() -> tuple[str | None, list[dict]]:
    """Still-fresh search results as (prompt block, source list for the UI's
    verification chip), purging expired entries."""
    now = time.time()
    for k in [k for k, e in _web_cache.items() if now - e["t"] > WEB_TTL_S]:
        del _web_cache[k]
    if not _web_cache:
        return None, []
    parts, sources, seen = [], [], set()
    for e in sorted(_web_cache.values(), key=lambda e: e["t"]):
        lines = "\n".join(f"- {r['title']}: {r['description']} ({r['url']})"
                          for r in e["results"])
        parts.append(f"Search \"{e['query']}\":\n{lines}")
        for r in e["results"]:
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                sources.append({"title": r["title"], "url": r["url"]})
    return "\n\n".join(parts), sources
_embedder = None    # real embedding model if available, else None (dump-all recall)
_lock = threading.Lock()

# Background fact-extraction jobs: (user_text, event_id, llm_cfg, candidate,
# context). Filled at the tail of each turn, drained by the memory worker so the
# slow extraction LLM call never runs while the turn holds _lock (which would
# stall the next turn).
_mem_jobs: "queue.Queue" = queue.Queue()

# Live background-work counters, surfaced at GET /status so the UI can show a
# subtle "updating memory…" indicator (and warn that brand-new facts/recaps may
# not be recallable until the work finishes). Plain ints under the GIL — a
# transiently stale count only affects a status pill, nothing else.
_bg = {"memory": 0, "recap": 0}


def _bg_start(kind: str) -> None:
    _bg[kind] = _bg.get(kind, 0) + 1


def _bg_end(kind: str) -> None:
    _bg[kind] = max(0, _bg.get(kind, 0) - 1)

_history: list[dict] = []
_session: dict | None = None   # the in-progress call
_forgotten: list[str] = []     # keywords deleted THIS session — the model is told
                               # to never reference them again (they linger in the
                               # chat history, which is how "Pixel" kept coming up)


# ---- persistence -------------------------------------------------------------
# Parsed-JSON cache keyed by (path, mtime): settings/profile/documents are read
# on every turn but change rarely. SESSIONS_FILE is deliberately NOT cached — it
# is mutated concurrently (turns + recap threads), and handing all readers a
# shared parsed object would create serialize-during-mutation races that the
# read-fresh-from-disk pattern avoids.
_json_cache: dict = {}


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        if path != SESSIONS_FILE:
            mtime = path.stat().st_mtime_ns
            hit = _json_cache.get(path)
            if hit and hit[0] == mtime:
                return hit[1]
            obj = json.loads(path.read_text())
            _json_cache[path] = (mtime, obj)
            return obj
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    _json_cache.pop(path, None)


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
            if k in ("thinking", "configured", "vad_enabled", "vad_barge_in",
                     "debug_settings"):
                v = bool(v)
            elif k in _FLOAT_FIELDS:
                try: v = float(v)
                except (TypeError, ValueError): v = DEFAULT_SETTINGS[k]
            elif k in _INT_FIELDS:
                try: v = int(float(v))
                except (TypeError, ValueError): v = DEFAULT_SETTINGS[k]
            s[k] = v
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
        # generation controls (map llm_* settings -> the keys llm_complete expects)
        "temperature": o.get("temperature", s["llm_temperature"]),
        "top_p": o.get("top_p", s["llm_top_p"]),
        "max_tokens": o.get("max_tokens", s["llm_max_tokens"]),
        "num_ctx": o.get("num_ctx", s["llm_num_ctx"]),
    }
    cfg["api_key"] = o.get("api_key") or _get_api_key() or ""
    return cfg


def _mem_llm_cfg() -> dict:
    """LLM config for background memory work (polish/extract/harvest/recap).
    Honors the optional memory_model setting: a bigger model can parse noisy
    speech far better, and since these jobs run behind the conversation, its
    slowness never touches voice latency. Empty = same model as chat."""
    cfg = _llm_cfg()
    mm = _settings().get("memory_model", "").strip()
    if mm:   # works on either backend — a different Ollama model or a different
             # model id at the same OpenAI-compatible endpoint
        cfg["ollama_model" if cfg.get("backend") == "ollama" else "openai_model"] = mm
    return cfg


def _load_tts(engine: str):
    """Load a TTS engine and return (model, engine). 'chatterbox' loads the
    expressive Turbo model; anything else loads Kokoro (+ its SineGen patch)."""
    from mlx_audio.tts.utils import load_model
    if engine == "chatterbox":
        # Turbo logs a per-call warning that it ignores CFG/exaggeration — quiet it.
        logging.getLogger("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo"
                          ).setLevel(logging.ERROR)
        m = load_model(vc.TTS_CHATTERBOX)
        global _cb_builtin_conds, _cb_active_voice, _cb_conds_cache
        _cb_builtin_conds = getattr(m, "_conds", None)   # snapshot the default voice
        _cb_active_voice, _cb_conds_cache = "builtin", {}
        return m, "chatterbox"
    m = load_model(vc.TTS_MODEL)
    vc.patch_kokoro_tts()
    return m, "kokoro"


def _reload_tts(engine: str):
    """Swap the live voice engine. Returns None on success, else an error string.
    Fail-safe like _reload_asr: the new engine must load before it replaces the
    current one, so a bad pick can't brick a call."""
    global _tts, _tts_engine
    try:
        new, eng = _load_tts(engine)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    _tts, _tts_engine = new, eng
    print(f"[tts] switched to {eng}")
    return None


# ---- Chatterbox voices (reference-clip cloning) ------------------------------
# Chatterbox has one built-in voice; every other voice is cloned from a short
# reference clip. Presets ship in PRESET_VOICES_DIR (with a manifest for friendly
# labels + CC0 attribution); the user's uploaded/cloned voices go in
# CUSTOM_VOICES_DIR. Deriving a voice ("conditionals") costs ~0.5s, so we cache
# by path and only switch when the selection changes — synthesis then runs at
# full speed.
_cb_builtin_conds = None    # the model's own built-in conditionals, to restore
_cb_active_voice = "builtin"   # which voice is currently prepared into _tts
_cb_conds_cache: dict = {}     # abs path -> prepared conditionals object


def _voice_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return slug or "voice"


def _cb_voice_path(voice_id: str):
    """Resolve a voice id ('preset:foo' / 'custom:bar') to its clip path, or None
    for the built-in voice / unknown id."""
    if not voice_id or voice_id == "builtin":
        return None
    kind, _, name = voice_id.partition(":")
    d = PRESET_VOICES_DIR if kind == "preset" else CUSTOM_VOICES_DIR
    for ext in VOICE_AUDIO_EXTS:
        p = d / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _cb_voice_catalog() -> list[dict]:
    """All selectable Chatterbox voices: built-in, bundled presets, user uploads."""
    out = [{"id": "builtin", "label": "Built-in", "kind": "builtin", "note": "Chatterbox's default voice."}]
    man = _read_json(PRESET_VOICES_DIR / "manifest.json", {}) if PRESET_VOICES_DIR.exists() else {}
    seen = set()
    presets = []
    if PRESET_VOICES_DIR.exists():
        for p in sorted(PRESET_VOICES_DIR.iterdir()):
            if p.suffix.lower() not in VOICE_AUDIO_EXTS or p.stem in seen:
                continue
            seen.add(p.stem)
            meta = man.get(p.name, man.get(p.stem, {}))
            presets.append((meta.get("order", 999), {
                "id": "preset:" + p.stem, "kind": "preset",
                "label": meta.get("label", p.stem.replace("_", " ").title()),
                "note": meta.get("note", ""), "source": meta.get("source", "")}))
    out += [v for _, v in sorted(presets, key=lambda x: x[0])]   # manifest order
    if CUSTOM_VOICES_DIR.exists():
        for p in sorted(CUSTOM_VOICES_DIR.iterdir()):
            if p.suffix.lower() not in VOICE_AUDIO_EXTS:
                continue
            out.append({"id": "custom:" + p.stem, "kind": "custom",
                        "label": p.stem.replace("_", " ").title(), "note": "Your voice."})
    return out


def _cb_select_voice(voice_id: str) -> None:
    """Make `voice_id` the active Chatterbox voice (prepare/restore its
    conditionals). Cheap when already active or cached. No-op off chatterbox."""
    global _cb_active_voice
    if _tts_engine != "chatterbox" or _tts is None:
        return
    if voice_id == _cb_active_voice:
        return
    path = _cb_voice_path(voice_id)
    if path is None:                       # built-in / missing → restore default
        if _cb_builtin_conds is not None:
            _tts._conds = _cb_builtin_conds
        _cb_active_voice = "builtin"
        return
    key = str(path)
    cached = _cb_conds_cache.get(key)
    if cached is not None:
        _tts._conds = cached
    else:
        try:
            _tts.prepare_conditionals(key)   # derives + sets _tts._conds
            _cb_conds_cache[key] = _tts._conds
        except Exception as e:
            print(f"[tts] voice clone failed for {voice_id} ({e}); using built-in")
            if _cb_builtin_conds is not None:
                _tts._conds = _cb_builtin_conds
            _cb_active_voice = "builtin"
            return
    _cb_active_voice = voice_id


def _ingest_voice_clip(raw: bytes, name: str) -> dict:
    """Save an uploaded reference clip as a custom voice: normalize to mono 24 kHz
    WAV, capped at 20s (ffmpeg, already a prerequisite). Returns the catalog entry;
    raises on unreadable audio or a clip too short to clone from."""
    import subprocess
    import tempfile
    CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    slug = _voice_slug(name)
    dest = CUSTOM_VOICES_DIR / f"{slug}.wav"
    i = 2
    while dest.exists():                       # never clobber an existing voice
        dest = CUSTOM_VOICES_DIR / f"{slug}_{i}.wav"
        i += 1
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tf.write(raw)
        tmp = Path(tf.name)
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-ac", "1", "-ar", str(vc.TTS_SR),
                        "-t", "20", str(dest)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise ValueError(f"couldn't read that audio file ({e})")
    finally:
        tmp.unlink(missing_ok=True)
    if not dest.exists() or sf.info(str(dest)).frames / vc.TTS_SR < 3.0:
        dest.unlink(missing_ok=True)
        raise ValueError("clip too short — use at least 5 seconds of clear speech")
    return {"id": "custom:" + dest.stem, "kind": "custom",
            "label": dest.stem.replace("_", " ").title(), "note": "Your voice."}


def _cb_delete_custom(voice_id: str) -> bool:
    """Delete a custom (uploaded) voice; fall back to built-in if it was active."""
    global _cb_active_voice
    if not voice_id.startswith("custom:"):
        return False
    path = _cb_voice_path(voice_id)
    if path is None:
        return False
    _cb_conds_cache.pop(str(path), None)
    if _cb_active_voice == voice_id:
        if _cb_builtin_conds is not None and _tts is not None:
            _tts._conds = _cb_builtin_conds
        _cb_active_voice = "builtin"
    path.unlink(missing_ok=True)
    return True


def _load_models() -> None:
    global _asr, _asr_key, _tts, _tts_engine, _strata, _embedder
    print("Loading models (first run downloads them)…")
    from strata.gateway.api import Strata

    STORE.mkdir(parents=True, exist_ok=True)
    # Honor the saved speech-recognition choice; fall back to the default if that
    # model can't load, so a bad saved pick never blocks startup.
    s = _settings()
    backend = s.get("asr_backend") or vc.ASR_BACKEND
    if backend == "parakeet-mlx":     # retired A/B loader — old settings coerce cleanly
        backend = "mlx-audio"
    want = (s.get("asr_model") or vc.ASR_MODEL, backend)
    try:
        _asr = vc.load_asr(model=want[0], backend=want[1]); _asr_key = want
    except Exception as e:
        print(f"[asr] saved model {want} failed to load ({e}); using default")
        _asr = vc.load_asr(); _asr_key = (vc.ASR_MODEL, vc.ASR_BACKEND)
    # Honor the saved voice engine; fall back to Kokoro if it can't load, so a bad
    # saved pick never blocks startup.
    try:
        _tts, _tts_engine = _load_tts(s.get("tts_engine") or "kokoro")
    except Exception as e:
        print(f"[tts] saved engine failed to load ({e}); using Kokoro")
        _tts, _tts_engine = _load_tts("kokoro")
    Path(vc.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Embeddings follow the configured Ollama endpoint — which may be another
    # machine on the network — instead of assuming localhost.
    _embedder = vc.make_embedder(s.get("ollama_url") or vc.OLLAMA_URL)
    _strata = Strata.open(db_path=vc.DB_PATH, embedder=_embedder)
    if _embedder is not None:
        n = vc.warm_index(_strata)
        print(f"  · warmed vector index with {n} memor{'y' if n==1 else 'ies'}")
    _backfill_events()
    print(f"Ready · LLM={vc.LLM_MODEL} · ASR={_asr_label(_asr_key)} · "
          f"TTS={_tts_engine} · DB={vc.DB_PATH}")


def _asr_label(key) -> str:
    """Friendly catalog label for a loaded (model, backend), or the raw id."""
    if not key:
        return "?"
    for m in ASR_MODELS:
        if (m["model"], m["backend"]) == tuple(key):
            return m["label"].split(" — ")[0] + f" ({key[1]})"
    return f"{key[0]} ({key[1]})"


_ASR_PROBE = None


def _asr_smoke(asr) -> None:
    """Run a tiny silent clip through the model. Some models load fine but fail at
    transcribe time (e.g. missing HF processor), so this proves it actually runs
    before we commit the swap. Raises if the model can't transcribe."""
    global _ASR_PROBE
    if _ASR_PROBE is None:
        _ASR_PROBE = str(HERE / ".asr_probe.wav")
        sf.write(_ASR_PROBE, np.zeros(8000, dtype="float32"), 16000)
    asr.transcribe(_ASR_PROBE)


def _reload_asr(model: str, backend: str):
    """Swap the live ASR model. Returns None on success, else an error string.
    Fail-safe: the new model must both load AND pass a smoke transcription before
    it replaces the current one — so a bad pick can never brick a real call; the
    picker just reports what went wrong and keeps the working model."""
    global _asr, _asr_key
    try:
        new = vc.load_asr(model=model, backend=backend)
        _asr_smoke(new)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    _asr = new
    _asr_key = (model, backend)
    print(f"[asr] switched to {backend}:{model}")
    return None


def _backfill_events() -> None:
    """One-time: seed the episodic timeline from past saved sessions so it isn't
    empty when the feature first ships. Skips if any turn events already exist."""
    if vc.list_events(_strata):
        return
    sessions = _read_json(SESSIONS_FILE, [])
    n = 0
    for s in sorted(sessions, key=lambda s: s.get("started_at", 0)):
        for turn in s.get("turns", []):
            if turn.get("role") != "user":
                continue
            ts = turn.get("t")
            ts_ms = int(ts * 1000) if isinstance(ts, (int, float)) else None
            if vc.record_event(_strata, turn.get("content", ""), ts_ms=ts_ms):
                n += 1
    if n:
        print(f"  · backfilled {n} past turn{'s' if n != 1 else ''} into the timeline")


def _backfill_recaps() -> None:
    """Background, one-time: generate a conversation recap for recent past sessions
    that lack one, so cross-conversation recall works even for sessions that never
    reached finalize (abrupt stop / pre-feature). Uses its own Strata connection
    (the DB is WAL, so concurrent connections are safe; recap vectors are unused —
    relevant_recaps re-embeds the recap text via the live embedder). The slow
    summary LLM calls run off the lock so they never delay turns or startup."""
    try:
        from strata.gateway.api import Strata
        todo = [s for s in _read_json(SESSIONS_FILE, []) if s.get("turns") and not s.get("summary")]
        todo = sorted(todo, key=lambda s: s.get("started_at", 0))[-12:]
        if not todo:
            return
        bf = Strata.open(db_path=vc.DB_PATH)   # default embedder; we only need the canonical write
        done = 0
        try:
            _bg_start("recap")
            for s in todo:
                recap = vc.summarize_session(s["turns"], _mem_llm_cfg())   # slow LLM, off-lock
                if recap:
                    ts = s.get("ended_at") or s.get("started_at")
                    ts_ms = int(ts * 1000) if isinstance(ts, (int, float)) else None
                    with _lock:                                        # brief: sqlite write + json
                        vc.record_summary(bf, recap, ts_ms=ts_ms)
                        cur = _read_json(SESSIONS_FILE, [])
                        for c in cur:
                            if c.get("id") == s.get("id"):
                                c["summary"] = recap
                        _write_json(SESSIONS_FILE, cur)
                    done += 1
                # a session that never reached finalize also never got its fact
                # harvest — recover it here so an interrupted background job is
                # always completed on the next launch (dedup makes this safe)
                try:
                    existing = [m["text"] for m in vc.list_memories(bf)]
                    facts = vc.harvest_session_facts(s["turns"], existing, _mem_llm_cfg())
                    if facts:
                        with _lock:
                            added = vc.add_facts(bf, facts)
                        if added:
                            print("[memory] harvested (backfill):", added)
                except Exception as e:
                    print(f"[memory] backfill harvest failed: {e}")
        finally:
            _bg_end("recap")
            bf.close()
        if done:
            print(f"  · backfilled {done} conversation recap{'s' if done != 1 else ''}")
    except Exception as e:
        print(f"[recap] backfill failed: {e}")


def _memory_worker() -> None:
    """Background memory-writing worker — ALL fact writes flow through here so
    nothing is ever stored verbatim from a voice transcript. Per job:
    (1) an explicit "remember …" candidate is polished (judged + rewritten into
    a clean third-person fact) and stored, and (2) the extraction pass distills
    any other durable facts from the turn. Both are multi-second LLM calls, so
    they run here instead of inline (inline they held _lock past the end of the
    turn, stalling the user's next question).

    Pattern mirrors _backfill_recaps: own Strata connection (DB is WAL, so a second
    connection is safe), the slow LLM calls run off-lock, and only the brief
    canonical write is taken under _lock (so the two connections never collide on
    SQLite). New facts land in the canonical store and are visible to the main
    connection's next read, so dump-all recall sees them right away; only large-
    store *vector* recall of a just-written fact may lag until the next index warm."""
    try:
        from strata.gateway.api import Strata
        st = Strata.open(db_path=vc.DB_PATH,
                         embedder=vc.make_embedder(_settings().get("ollama_url") or vc.OLLAMA_URL))
    except Exception as e:
        print(f"[memory] memory worker disabled ({e}); "
              "new facts won't be saved this session (forgets still work)")
        st = None
    while True:
        job = _mem_jobs.get()
        try:
            if job is None:
                return
            if st is None:
                continue
            text, event_id, cfg, candidate, ctx = job
            _bg_start("memory")
            t0 = time.monotonic()
            try:
                # explicit "remember …" first: the smoothing layer judges + rewrites
                # it (or falls back to verbatim if the LLM is unreachable) so an
                # explicit command is never lost and never stored as garble.
                if candidate:
                    polished = vc.polish_fact(candidate, cfg)          # slow LLM, off-lock
                    if polished:
                        with _lock:
                            added = vc.add_facts(st, [polished], event_id)
                        if added:
                            print("[memory] remembered:", added)
                existing = [m["text"] for m in vc.list_memories(st)]   # read, off-lock
                new_facts = vc.extract_facts_llm(text, existing, cfg, context=ctx)  # slow LLM, off-lock
                if new_facts:
                    with _lock:                                        # brief write, serialized
                        added = vc.add_facts(st, new_facts, event_id)
                    if added:
                        print("[memory] extracted:", added)
                print(f"[memory] turn job done in {time.monotonic()-t0:.1f}s "
                      f"({_mem_jobs.qsize()} queued)")
            finally:
                _bg_end("memory")
        except Exception as e:
            print("[memory] extraction error:", e)
        finally:
            _mem_jobs.task_done()


# ---- hands-free VAD micro-server (:VAD_PORT) ----------------------------------
# Lives on its own port + thread so speech detection keeps working while the
# main (single-threaded) server is busy streaming a turn — which is exactly when
# barge-in matters. The Silero model is lazy-loaded inside the serving thread
# (same MLX thread philosophy as the main models) and never touches _lock.
# State is owned exclusively by the VAD thread: the browser sends one chunk at a
# time and waits for the response, so there is no concurrent access by design.
_vad_model = None
_vad_sv = None       # current StreamingVad session (None = not armed)


def _vad_clamp(body: dict):
    """Build a clamped ServerVadConfig from a client config dict."""
    from mlx_audio.realtime_vad import ServerVadConfig
    def f(key, default, lo, hi, cast):
        try:
            return min(hi, max(lo, cast(body.get(key, default))))
        except (TypeError, ValueError):
            return default
    return ServerVadConfig(
        threshold=f("threshold", 0.5, 0.05, 0.95, float),
        prefix_padding_ms=f("prefix_padding_ms", 300, 0, 1000, int),
        silence_duration_ms=f("silence_duration_ms", 500, 150, 3000, int),
    )


class VadHandler(BaseHTTPRequestHandler):
    # HTTP/1.0 = no keep-alive, on purpose: this server is single-threaded, and a
    # browser holding one idle keep-alive connection would block every request
    # arriving on a second pooled connection (observed as multi-second stalls).
    # Localhost connection setup is ~free at 5 req/s, so close after every reply.
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):  # quiet
        pass

    def _cors(self):
        # The page origin is :PORT, this server is :VAD_PORT — cross-origin.
        # Local-only (binds 127.0.0.1), no secrets, so a wildcard is fine.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n) if n else b""

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        global _vad_model, _vad_sv
        u = urlparse(self.path)
        try:
            if u.path == "/vad/start":
                body = json.loads(self._body() or b"{}")
                from mlx_audio.realtime_vad import StreamingVad
                if _vad_model is None:
                    from mlx_audio.vad import load_model
                    _vad_model = load_model("mlx-community/silero-vad")
                    print("[vad] silero-vad loaded")
                cfg = _vad_clamp(body)
                # a fresh session always replaces the old one, so a page reload
                # can never leave a stale detector behind
                _vad_sv = StreamingVad(_vad_model, cfg)
                return self._json(200, {"ok": True, "config": cfg.to_dict(),
                                        "frame_ms": 32})
            if u.path == "/vad/feed":
                data = self._body()
                if _vad_sv is None:
                    return self._json(200, {"ok": False, "error": "no session"})
                if len(data) % 2:
                    data = data[:-1]
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                try:
                    events = _vad_sv.process(samples)
                except Exception as e:
                    # MLX hiccup (e.g. GPU contention) — chunk is skipped, the
                    # detector's clock only advances on processed audio, so the
                    # client should just keep feeding.
                    print(f"[vad] process failed: {e}")
                    return self._json(200, {"ok": False, "retry": True})
                return self._json(200, {
                    "ok": True,
                    "events": [{"kind": e.kind.value, "ms": e.audio_ms} for e in events],
                    "in_speech": _vad_sv.in_speech,
                })
            if u.path == "/vad/config":
                body = json.loads(self._body() or b"{}")
                if _vad_sv is None:
                    return self._json(200, {"ok": False, "error": "no session"})
                cfg = _vad_clamp(body)
                try:
                    # live retune preserving the audio clock: transplant the
                    # detector's counters into a fresh one with the new config
                    from mlx_audio.realtime_vad import TurnDetector
                    old = _vad_sv._detector
                    nd = TurnDetector(cfg)
                    nd._elapsed_ms = old._elapsed_ms
                    nd._in_speech = old._in_speech
                    nd._silence_ms = old._silence_ms
                    _vad_sv._detector = nd
                    _vad_sv._config = cfg
                    return self._json(200, {"ok": True, "clock_preserved": True,
                                            "config": cfg.to_dict()})
                except AttributeError:
                    # library internals changed — rebuild; client resets counters
                    from mlx_audio.realtime_vad import StreamingVad
                    _vad_sv = StreamingVad(_vad_model, cfg)
                    return self._json(200, {"ok": True, "clock_preserved": False,
                                            "config": cfg.to_dict()})
            if u.path == "/vad/stop":
                _vad_sv = None
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "unknown endpoint"})
        except Exception as e:
            try:
                self._json(500, {"ok": False, "error": str(e)})
            except Exception:
                pass


# ---- speech synthesis --------------------------------------------------------
def _for_speech(text: str, mode: str = "natural") -> str:
    """Punctuation pass before synthesis — Kokoro pauses on punctuation, so this
    shapes cadence. Only affects what's spoken, not what's stored.
      verbatim — speak punctuation as written (just tidy spacing)
      natural  — dashes/semicolons become comma pauses; collapse comma runs
      flowing  — drop commas/semicolons/colons/dashes to minimize mid-line pauses
    """
    if mode == "verbatim":
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()
    if mode == "flowing":
        text = re.sub(r"\s*[—–]\s*", " ", text)        # dash -> just a space
        text = re.sub(r"[;:,]", " ", text)             # no clause pauses at all
        text = text.replace("…", ".")
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"\s+([.!?])", r"\1", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()
    # natural (default)
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    text = text.replace(";", ",").replace("…", ".")
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _trim_silence(audio: np.ndarray, keep_ms: int = 20, thresh: float = 0.01) -> np.ndarray:
    """Trim near-silent head/tail (the main cause of weird inter-chunk pauses),
    keeping a small cushion so words don't clip."""
    if audio.size == 0:
        return audio
    loud = np.where(np.abs(audio) > thresh)[0]
    if loud.size == 0:
        return audio
    keep = int(vc.TTS_SR * keep_ms / 1000)
    start = max(0, int(loud[0]) - keep)
    end = min(audio.size, int(loud[-1]) + keep)
    return audio[start:end]


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(vc.TTS_SR * max(0, ms) / 1000), dtype=np.float32)


def _emotion_active() -> bool:
    """Inline expressive tags are only performed on chatterbox with the toggle on."""
    return _tts_engine == "chatterbox" and bool(_settings().get("tts_emotion", True))


def _synth_sentence(text: str, voice: str, speed: float, *,
                    trim: bool = True, smoothing: str = "natural") -> np.ndarray:
    """Synthesize one segment. Returns float32 @ TTS_SR (empty on failure)."""
    text = _for_speech(text, smoothing)
    if _emotion_active():
        # official tags only (off-vocab guesses get read aloud), none at the front
        text = vc.fix_leading_tags(vc.sanitize_emotion_tags(text))
    else:
        text = vc.strip_emotion_tags(text)   # engine can't speak tags → drop them
    if not text:
        return np.zeros(0, dtype=np.float32)
    try:
        if _tts_engine == "chatterbox":
            segs = list(_tts.generate(text, temperature=0.8))
        else:
            segs = list(_tts.generate(text, voice=voice, speed=speed,
                                      lang_code=_lang_code(voice)))
        if segs:
            audio = np.concatenate([np.asarray(s.audio) for s in segs]).astype(np.float32)
            return _trim_silence(audio) if trim else audio
    except Exception as e:
        print(f"[tts] skipped sentence ({e}): {text!r}")
    return np.zeros(0, dtype=np.float32)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?,;:])\s+")


def _chunk_text(text: str, chunking: str) -> list[str]:
    """Split a full reply into synthesis chunks per the chosen strategy."""
    text = text.strip()
    if not text:
        return []
    if chunking == "whole":
        return [text]
    sents = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()] or [text]
    if chunking == "hybrid":
        return [sents[0], " ".join(sents[1:])] if len(sents) > 1 else sents
    if chunking == "clause":
        out = []
        for snt in sents:
            out.extend(p.strip() for p in _CLAUSE_SPLIT.split(snt) if p.strip())
        return out or sents
    return sents  # sentence


def _synth_reply(text: str, *, chunking: str | None = None, trim: bool | None = None,
                 gap_ms: int | None = None, smoothing: str | None = None,
                 voice: str | None = None, speed: float | None = None) -> np.ndarray:
    """Non-streaming full synthesis honoring the cadence settings. Used by the
    voice Preview and the non-streaming /turn path; the streaming path mirrors
    this chunk-by-chunk. Kokoro-MLX's harmonic-source broadcast bug is handled by
    patch_kokoro_tts(); a chunk that still fails is skipped, not fatal."""
    s = _settings()
    chunking = chunking or s["tts_chunking"]
    trim = s["tts_trim"] if trim is None else trim
    gap_ms = int(s["tts_gap_ms"] if gap_ms is None else gap_ms)
    smoothing = smoothing or s["tts_smoothing"]
    voice = voice or s["tts_voice"]
    speed = speed if speed is not None else float(s["tts_speed"])
    gap = _silence(gap_ms)
    out: list[np.ndarray] = []
    for chunk in _chunk_text(text, chunking):
        audio = _synth_sentence(chunk, voice, speed, trim=trim, smoothing=smoothing)
        if audio.size:
            out.append(audio)
            if gap.size:
                out.append(gap)
    if not out:
        return np.zeros(0, dtype=np.float32)
    if gap.size and len(out) >= 2:
        out = out[:-1]   # no trailing gap
    return np.concatenate(out)


def _audio_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, vc.TTS_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENT_BOUNDARY = re.compile(r'[.!?]+["\')\]]*(\s|$)')
_CLAUSE_BOUNDARY = re.compile(r'[.!?,;:]+["\')\]]*(\s|$)')


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


def _pop_sentences(spoken: str, emitted: int, clause: bool = False) -> tuple[list[str], int]:
    """Pull complete sentences (or clauses) from spoken[emitted:]; return
    (segments, new_emitted)."""
    boundary = _CLAUSE_BOUNDARY if clause else _SENT_BOUNDARY
    out: list[str] = []
    region = spoken[emitted:]
    while True:
        m = boundary.search(region)
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


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _chunk_doc(text: str) -> list[str]:
    """Split a document into overlapping ~DOC_CHUNK_CHARS passages on sentence /
    paragraph boundaries, so retrieval returns coherent excerpts, not fragments."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks, cur = [], []
    size = 0
    for s in sents:
        cur.append(s)
        size += len(s) + 1
        if size >= DOC_CHUNK_CHARS:
            chunks.append(" ".join(cur))
            cur = cur[-DOC_CHUNK_OVERLAP:] if DOC_CHUNK_OVERLAP else []
            size = sum(len(x) + 1 for x in cur)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


def _doc_chunks_embedded(doc: dict) -> list:
    """Return [(chunk_text, vector)] for a document, embedding once and caching.
    Returns [] if there's no embedder (caller falls back to whole-text)."""
    if _embedder is None:
        return []
    text = doc.get("text") or ""
    cached = _doc_vec_cache.get(doc["id"])
    if cached and cached["n"] == len(text):
        return cached["chunks"]
    pairs = []
    for ch in _chunk_doc(text):
        try:
            pairs.append((ch, _embedder.embed(ch)))
        except Exception as e:
            print(f"[docs] chunk embed failed: {e}")
            return []
    _doc_vec_cache[doc["id"]] = {"n": len(text), "chunks": pairs}
    return pairs


def _doc_context(query: str | None = None) -> list[str]:
    """Reference-file text for the prompt. Small docs go in whole; larger ones are
    chunked and only the passages most relevant to `query` are injected (RAG), so
    a long document doesn't blow the context or get blindly truncated."""
    from strata.vector.embedder import cosine
    docs = _read_json(DOCS_FILE, [])
    q = (query or "").strip()
    out, total = [], 0
    for d in docs:
        text = d.get("text") or ""
        if not text:
            continue
        name = d.get("name", "document")
        # whole-document path: small doc, no query, or no embedder available
        if len(text) <= DOC_WHOLE_CAP or not q or _embedder is None:
            block = f"=== {name} ===\n{text[:DOC_CHAR_CAP]}"
        else:
            chunks = _doc_chunks_embedded(d)
            if not chunks:                                   # embed failed -> whole
                block = f"=== {name} ===\n{text[:DOC_CHAR_CAP]}"
            else:
                try:
                    qv = _embedder.embed(q)
                    ranked = sorted(chunks, key=lambda c: cosine(qv, c[1]), reverse=True)
                    top = ranked[:DOC_TOP_K]
                    order = {id(c): i for i, c in enumerate(chunks)}   # restore reading order
                    top.sort(key=lambda c: order[id(c)])
                    excerpt = "\n…\n".join(c[0] for c in top)
                except Exception as e:
                    print(f"[docs] retrieval failed: {e}")
                    excerpt = text[:DOC_CHAR_CAP]
                block = f"=== {name} (excerpts most relevant to the question) ===\n{excerpt}"
        block = block[: DOC_TOTAL_CAP - total]
        out.append(block)
        total += len(block)
        if total >= DOC_TOTAL_CAP:
            break
    return out


# ---- sessions ----------------------------------------------------------------
def _recent_context(max_entries: int = 6) -> str:
    """The last few exchanges (excluding the just-appended current one) as plain
    text, for reference resolution in the background fact extraction."""
    prior = _history[:-2] if len(_history) >= 2 else []
    tail = prior[-max_entries:]
    return "\n".join(f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
                     for t in tail)


def _start_session() -> None:
    global _session, _history, _forgotten
    _finalize_session()
    _history = []
    _forgotten = []
    _session = {
        "id": str(int(time.time() * 1000)),
        "started_at": time.time(),
        "ended_at": None,
        "title": None,
        "turns": [],
    }


def _finalize_session() -> None:
    """End the current session: persist it immediately, then recap it in the
    background. The recap is a full LLM call (seconds) — running it inline held
    _lock, so ending a call and restarting froze the new call's first turns
    behind the old call's recap."""
    global _session
    if _session and _session["turns"]:
        _session["ended_at"] = _session["ended_at"] or time.time()
        sessions = _read_json(SESSIONS_FILE, [])
        sessions = [s for s in sessions if s["id"] != _session["id"]]
        sessions.append(_session)
        _write_json(SESSIONS_FILE, sessions)
        snap = _session   # recap thread works from the snapshot
        threading.Thread(target=_recap_session, args=(snap,),
                         name="recap-finalize", daemon=True).start()
    _session = None


def _recap_session(sess: dict) -> None:
    """Background: recap one finished session into Strata's episodic layer, then
    harvest durable facts from the WHOLE transcript. The harvest is what catches
    facts scattered across turns ("I have an interview" … "it's next Tuesday" …
    "it's a builder role") that per-turn extraction can't assemble.
    Same discipline as _backfill_recaps — slow LLM calls off-lock, brief writes
    under _lock. If the process dies first, the startup backfill catches BOTH
    (it re-recaps and re-harvests any session left without a summary)."""
    _bg_start("recap")
    t0 = time.monotonic()
    try:
        recap = vc.summarize_session(sess["turns"], _mem_llm_cfg())   # slow, off-lock
        if recap:
            with _lock:
                vc.record_summary(_strata, recap, ts_ms=int(sess["ended_at"] * 1000))
                cur = _read_json(SESSIONS_FILE, [])
                for c in cur:
                    if c.get("id") == sess["id"]:
                        c["summary"] = recap
                _write_json(SESSIONS_FILE, cur)
            print("[recap]", recap)
    except Exception as e:
        print("[recap] finalize failed:", e)
    try:
        with _lock:
            existing = [m["text"] for m in vc.list_memories(_strata)]
        facts = vc.harvest_session_facts(sess["turns"], existing, _mem_llm_cfg())  # slow, off-lock
        if facts:
            with _lock:
                added = vc.add_facts(_strata, facts)
            if added:
                print("[memory] harvested:", added)
    except Exception as e:
        print("[memory] harvest failed:", e)
    finally:
        _bg_end("recap")
        print(f"[recap] session wrap-up done in {time.monotonic()-t0:.1f}s")


def _persist_session() -> None:
    if not _session:
        return
    sessions = _read_json(SESSIONS_FILE, [])
    sessions = [s for s in sessions if s["id"] != _session["id"]]
    sessions.append(_session)
    _write_json(SESSIONS_FILE, sessions)


# ---- turn --------------------------------------------------------------------
def _stream_turn(text: str, *, speak: bool, emit_tokens: bool, private: bool = False):
    """Shared turn core for both voice and text. Runs the full memory + LLM +
    events + session pipeline and yields NDJSON-ready dicts:
      - token deltas ({"type":"token"}) when emit_tokens (text chat),
      - synthesized audio ({"type":"audio"}) when speak (voice, or text+TTS).
    Modality-agnostic: the only difference upstream is how `text` was obtained.

    When `private` (incognito), the turn leaves no trace: no session/transcript,
    no episodic event, no memory capture/extraction, no recap. It still READS
    existing profile + memories for context and keeps ephemeral in-call history."""
    if not private and _session is None:
        _start_session()
    _history.append({"role": "user", "content": text})   # ephemeral; not persisted
    event_id = None if private else vc.record_event(_strata, text)   # episodic spine (L0 event)
    if not private:
        captured = vc.capture_memory(_strata, text, event_id)   # forgets only, immediate
        if captured:
            print("[memory]", captured)
            for c in captured:
                if c.startswith("- forget: "):
                    _forgotten.append(c[len("- forget: "):])

    s = _settings()
    mem = vc.list_memories(_strata)
    mem_text = vc.select_memories(_strata, text, semantic=_embedder is not None, mems=mem)
    recent = vc.relevant_recaps(_strata, text, _embedder)   # gated by relevance
    # Web lookup (opt-in): a quick gate decides if this turn needs fresh info
    # ("double check that", store hours, scores…). New results join the
    # short-lived cache; anything still fresh is injected either way so
    # follow-ups keep digging into the same data without a re-search.
    # Live web-activity events ({"type":"web","label":…}) narrate each stage in
    # the UI — searching → found sites, summarizing — Hermes-style, so a 2-4s
    # web turn never looks frozen.
    web_ctx, web_fresh, web_sources = None, False, []
    if s.get("web_search"):
        query = vc.web_gate(text, _history[-6:], _mem_llm_cfg(),
                            place=_read_json(PROFILE_FILE, {}).get("location", ""))
        if query:
            cached = _web_fresh_match(query)
            if cached:
                # actively being asked about — reuse and keep alive, no re-fetch
                cached["t"] = time.time()
                yield {"type": "web", "label": "going back over what it found"}
                print(f"[web] reusing cached results for {query!r}")
            else:
                yield {"type": "web", "label": f"searching the web · “{query}”"}
                results = vc.web_search(query)
                if results:
                    _web_remember(query, results)
                    web_fresh = True
                    doms = []
                    for r in results:
                        d = urlparse(r["url"]).netloc.replace("www.", "")
                        if d and d not in doms:
                            doms.append(d)
                    found = ", ".join(doms[:3]) or f"{len(results)} results"
                    # sites let the client rotate whimsical per-site wording
                    yield {"type": "web", "label": f"found {found} · summarizing",
                           "sites": doms[:5]}
                    print(f"[web] {len(results)} results for {query!r}")
                else:
                    yield {"type": "web", "label": "the search came up empty"}
                    print(f"[web] no results for {query!r}")
        web_ctx, web_sources = _web_block()

    voice, speed = s["tts_voice"], float(s["tts_speed"])
    if speak and _tts_engine == "chatterbox":
        _cb_select_voice(s.get("tts_cb_voice") or "builtin")   # one-time per voice
    # cadence settings: how the spoken reply is chunked + smoothed
    chunking = s["tts_chunking"]
    # Hybrid batches everything after sentence 1 into one big tail chunk. Kokoro
    # synthesizes ~10x real-time so that tail is ready instantly; Chatterbox runs
    # ~0.6x real-time, so the tail lands seconds after sentence 1 finishes — an
    # audible dead-air gap. Sentence streaming keeps synthesis ahead of playback.
    if _tts_engine == "chatterbox" and chunking == "hybrid":
        chunking = "sentence"
    trim, smoothing, gap_ms = bool(s["tts_trim"]), s["tts_smoothing"], int(s["tts_gap_ms"])
    gap = _silence(gap_ms)
    clause = chunking == "clause"
    stream_chunks = chunking in ("sentence", "clause", "hybrid")  # 'whole' waits for the end
    first_only = chunking == "hybrid"   # hybrid streams only the first sentence early
    # Long calls: inject only the recent tail of the conversation. The full
    # transcript lives in the session; older context is covered by memories +
    # recaps, and an unbounded history makes every turn's prefill slower.
    messages = vc.build_messages(
        _history[-40:], mem_text,
        documents=_doc_context(text), profile=_profile_context(), persona=s["persona"],
        recent=recent, forgotten=_forgotten, emotion=_emotion_active() and speak,
        web=web_ctx, web_fresh=web_fresh,
    )

    full, emitted_audio, emitted_tok, seq = "", 0, 0, 0
    reasoned = False   # have we told the client the model is reasoning (thinking, no output yet)?
    last_write = time.monotonic()   # heartbeat clock (see tick below)

    def _emit(seg: str):
        nonlocal seq
        audio = _synth_sentence(seg, voice, speed, trim=trim, smoothing=smoothing)
        if audio.size == 0:
            return None
        if gap.size:
            audio = np.concatenate([audio, gap])
        seq += 1
        return {"type": "audio", "seq": seq, "text": seg, "audio": _audio_b64(audio)}

    try:
        for tok in vc.llm_stream(messages, _llm_cfg()):
            full += tok
            clean = _spoken_region(full)   # strips [MEM_*] directives + <think>
            # heartbeat: a no-op line at most every 0.5s. Its real job is to
            # bound how long an abandoned turn survives — the single-threaded
            # server only notices a client abort when a write fails, and long
            # think-phases can otherwise go seconds without writing (hands-free
            # barge-in aborts the turn fetch and needs this slot freed fast).
            now = time.monotonic()
            if now - last_write > 0.5:
                last_write = now
                yield {"type": "tick"}
            # tokens are arriving but nothing speakable yet -> the model is
            # reasoning (inside <think>); tell the client once so it can show
            # "reasoning…" instead of looking frozen.
            if not reasoned and full.strip() and not clean.strip():
                reasoned = True
                yield {"type": "reasoning"}
            if emit_tokens and len(clean) > emitted_tok:
                yield {"type": "token", "text": clean[emitted_tok:]}
                emitted_tok = len(clean)
            if speak and stream_chunks:
                if first_only:
                    if emitted_audio == 0:
                        m = _SENT_BOUNDARY.search(clean)
                        if m:
                            emitted_audio = m.end()
                            msg = _emit(clean[:m.end()].strip())
                            if msg:
                                yield msg
                else:
                    sents, emitted_audio = _pop_sentences(clean, emitted_audio, clause=clause)
                    for seg in sents:
                        msg = _emit(seg)
                        if msg:
                            yield msg
    except Exception as e:
        print(f"[stream] llm error: {e}")
        yield {"type": "error", "error": str(e)}
        return

    if speak:
        # remainder: the trailing partial (sentence/clause), everything after the
        # first sentence (hybrid), or the whole reply ('whole') — as one smooth pass.
        remainder = _spoken_region(full)[emitted_audio:].strip()
        if remainder:
            msg = _emit(remainder)
            if msg:
                yield msg

    # parse directives (incognito: apply_directives still strips them from the
    # spoken text, but nothing is written because private turns emit none and the
    # persistence/extraction below is skipped)
    # Strip any expressive tags from the stored/displayed reply — they're a spoken
    # performance, so the transcript, memory, and caption stay clean prose.
    reply = vc.strip_emotion_tags(vc.apply_directives(_strata, full, mem, forgotten=_forgotten))
    _history.append({"role": "assistant", "content": reply})   # ephemeral

    if not private:
        _session["turns"].append({"role": "user", "content": text, "t": time.time()})
        _session["turns"].append({"role": "assistant", "content": reply, "t": time.time()})
        if not _session["title"]:
            _session["title"] = text[:60]
        _persist_session()
        # Memory writes go through the background smoothing layer: an explicit
        # "remember …" clause is polished (judged + rewritten) before storage,
        # and the LLM extraction pass distills everything else — with the last
        # few turns as context so fragments like "it's next Tuesday" resolve.
        # Nothing is stored verbatim, and none of it delays the next turn.
        _mem_jobs.put((text, event_id, _mem_llm_cfg(), vc.remember_candidate(text),
                       _recent_context()))

    memories = [m["text"] for m in vc.list_memories(_strata)]
    yield {"type": "done", "reply": reply, "memories": memories,
           "session": (_session["id"] if (_session and not private) else None),
           # sources the reply drew on (ephemeral, for the verification chip)
           **({"sources": web_sources} if web_ctx else {})}


def _handle_turn_stream(wav_bytes: bytes, private: bool = False):
    """Voice streaming turn: transcribe the mic audio, then run the shared core
    with audio synthesis on (and no token stream — the caption rides the audio)."""
    with _lock:
        tmp = HERE / ".turn_in.wav"
        tmp.write_bytes(wav_bytes)
        try:
            text = _asr.transcribe(str(tmp)).text.strip()
        finally:
            tmp.unlink(missing_ok=True)
        if not text:
            yield {"type": "empty"}
            return
        if not private and _session is None:
            _start_session()
        yield {"type": "meta", "transcript": text,
               "session": (_session["id"] if (_session and not private) else None)}
        yield from _stream_turn(text, speak=True, emit_tokens=False, private=private)


def _handle_chat_stream(text: str, speak: bool, private: bool = False):
    """Text streaming turn: same pipeline, fed a typed message. Streams the reply
    as text tokens; also speaks it via TTS when the user opted in."""
    text = (text or "").strip()
    with _lock:
        if not text:
            yield {"type": "empty"}
            return
        if not private and _session is None:
            _start_session()
        yield {"type": "meta", "transcript": text,
               "session": (_session["id"] if (_session and not private) else None)}
        yield from _stream_turn(text, speak=speak, emit_tokens=True, private=private)


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
            documents=_doc_context(question), profile=_profile_context(), persona=s["persona"])
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
        # The UI is a single file that changes with every update; never let a
        # browser serve a stale copy (that hides new Settings like the voice picker).
        if ctype.startswith("text/html"):
            self.send_header("Cache-Control", "no-store, must-revalidate")
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
            return self._json(200, {"name": _settings()["assistant_name"],
                                    "vad_port": VAD_PORT})
        if u.path == "/status":
            # live background work, for the UI's "updating memory…" pill
            mem = _bg["memory"] + _mem_jobs.qsize()
            return self._json(200, {"memory_jobs": mem, "recaps": _bg["recap"],
                                    "busy": bool(mem or _bg["recap"])})
        if u.path == "/settings":
            s = _settings()
            out = {k: s[k] for k in SETTINGS_FIELDS}
            out["has_api_key"] = bool(_get_api_key())
            out["persona_default"] = vc.PERSONA_PROMPT
            return self._json(200, out)
        if u.path == "/voices":
            return self._json(200, {"voices": _voice_list()})
        if u.path == "/tts/voices":
            return self._json(200, {"voices": _cb_voice_catalog(),
                                    "current": _settings().get("tts_cb_voice") or "builtin"})
        if u.path == "/asr/models":
            cur = _asr_key or (_settings()["asr_model"], _settings()["asr_backend"])
            return self._json(200, {"models": ASR_MODELS,
                                    "current": {"model": cur[0], "backend": cur[1]}})
        if u.path == "/ollama/models":
            url = (parse_qs(u.query).get("url", [None])[0] or _settings()["ollama_url"]).rstrip("/")
            try:
                import httpx
                r = httpx.get(f"{url}/api/tags", timeout=5)
                names = [m["name"] for m in r.json().get("models", []) if m.get("name")]
                return self._json(200, {"ok": True, "models": sorted(names)})
            except Exception as e:
                return self._json(200, {"ok": False, "error": str(e), "models": []})
        if u.path == "/openai/models":
            # Model list from the configured OpenAI-compatible endpoint (GET
            # {base}/models is part of the standard — LM Studio, vLLM, llama.cpp
            # server, OpenAI, DeepSeek all serve it). Feeds the memory-model picker.
            base = (parse_qs(u.query).get("base", [None])[0] or _settings()["openai_base"]).rstrip("/")
            if not base:
                return self._json(200, {"ok": False, "error": "no API endpoint configured", "models": []})
            try:
                import httpx
                headers = {}
                key = _get_api_key()
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                r = httpx.get(f"{base}/models", headers=headers, timeout=5)
                ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
                return self._json(200, {"ok": True, "models": sorted(ids)})
            except Exception as e:
                return self._json(200, {"ok": False, "error": str(e), "models": []})
        if u.path == "/memories":
            with _lock:
                mem = vc.list_memories(_strata)
            mem.sort(key=lambda m: m.get("t") or 0, reverse=True)   # newest first
            # ids are 64-bit ints; stringify so JS doesn't round them past 2^53.
            mem = [{"id": str(m["id"]), "text": m["text"], "t": m.get("t")} for m in mem]
            return self._json(200, {"memories": mem})
        if u.path == "/timeline":
            with _lock:
                tl = vc.build_timeline(_strata)
            return self._json(200, {"timeline": tl})
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
        if u.path == "/session/current":
            # the live conversation (for the sidebar's "Current" view)
            if _session and _session.get("turns"):
                return self._json(200, {"id": _session["id"], "turns": _session["turns"]})
            return self._json(200, {"id": _session["id"] if _session else None, "turns": []})
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
            # Speech-recognition model swap: eager + fail-safe. Only reload if the
            # pick actually changed; if the new model won't load, keep the current
            # one and don't persist the broken choice.
            asr_error = None
            if "asr_model" in body or "asr_backend" in body:
                cur = _settings()
                want = (body.get("asr_model") or cur["asr_model"],
                        body.get("asr_backend") or cur["asr_backend"])
                if _asr_key is None or tuple(_asr_key) != want:
                    with _lock:
                        asr_error = _reload_asr(want[0], want[1])
                    if asr_error:
                        body.pop("asr_model", None)
                        body.pop("asr_backend", None)
            # Voice-engine swap: same eager + fail-safe pattern. Chatterbox is a
            # ~1 GB first download, so the first switch can take a bit.
            tts_error = None
            if "tts_engine" in body:
                want_eng = body.get("tts_engine") or "kokoro"
                if want_eng != _tts_engine:
                    with _lock:
                        tts_error = _reload_tts(want_eng)
                    if tts_error:
                        body.pop("tts_engine", None)
            _save_settings(body)
            return self._json(200, {"ok": asr_error is None and tts_error is None,
                                    "asr_error": asr_error, "tts_error": tts_error})
        if u.path == "/tts/preview":
            body = json.loads(self._body() or b"{}")
            st = _settings()
            voice = body.get("voice") or st["tts_voice"]
            speed = float(body.get("speed", st["tts_speed"]))
            # cadence params come from the (possibly unsaved) Settings controls so
            # the user can tweak -> Preview -> adjust without saving first.
            chunking = body.get("chunking") or st["tts_chunking"]
            trim = bool(body["trim"]) if "trim" in body else st["tts_trim"]
            gap_ms = int(body.get("gap_ms", st["tts_gap_ms"]))
            smoothing = body.get("smoothing") or st["tts_smoothing"]
            sample = body.get("text") or ("Here's how I sound. I can pause between "
                "thoughts, breathe a little, and keep the rhythm natural.")
            with _lock:
                if _tts_engine == "chatterbox":   # audition the (maybe unsaved) pick
                    _cb_select_voice(body.get("cb_voice") or st.get("tts_cb_voice") or "builtin")
                audio = _synth_reply(sample, chunking=chunking, trim=trim, gap_ms=gap_ms,
                                     smoothing=smoothing, voice=voice, speed=speed)
            if audio.size == 0:
                return self._json(200, {"ok": False, "error": "synthesis failed"})
            buf = io.BytesIO()
            sf.write(buf, audio, vc.TTS_SR, format="WAV", subtype="PCM_16")
            return self._json(200, {"ok": True,
                                    "audio": base64.b64encode(buf.getvalue()).decode("ascii")})
        if u.path == "/tts/voice/upload":
            # raw audio bytes in the body; desired name in a header (any format
            # ffmpeg reads). Saved as a custom Chatterbox voice.
            raw = self._body()
            if not raw:
                return self._json(200, {"ok": False, "error": "no audio received"})
            name = unquote(self.headers.get("X-Voice-Name", "") or "My voice")
            try:
                with _lock:
                    entry = _ingest_voice_clip(raw, name)
            except Exception as e:
                return self._json(200, {"ok": False, "error": str(e)})
            return self._json(200, {"ok": True, "voice": entry})
        if u.path == "/tts/voice/delete":
            body = json.loads(self._body() or b"{}")
            with _lock:
                ok = _cb_delete_custom(body.get("id", ""))
            return self._json(200, {"ok": ok})
        if u.path == "/settings/test":
            body = json.loads(self._body() or b"{}")
            try:
                cfg = _llm_cfg(body)
                msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
                reply = vc.llm_complete(msgs, cfg)
                return self._json(200, {"ok": True, "reply": reply[:80]})
            except Exception as e:
                return self._json(200, {"ok": False, "error": str(e)})
        if u.path == "/turn/stream":
            wav = self._body()
            private = parse_qs(u.query).get("private", ["0"])[0] in ("1", "true")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for line in _handle_turn_stream(wav, private):
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
        if u.path == "/chat/stream":
            body = json.loads(self._body() or b"{}")
            msg = str(body.get("message", ""))
            speak = bool(body.get("speak", False))
            private = bool(body.get("private", False))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for line in _handle_chat_stream(msg, speak, private):
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
            # pre-embed chunks now so the first turn that uses this doc is snappy
            if len(text) > DOC_WHOLE_CAP:
                try: _doc_chunks_embedded(doc)
                except Exception as e: print(f"[docs] prewarm failed: {e}")
            return self._json(200, {"ok": True, "id": doc["id"], "name": name,
                                    "chars": len(text)})
        if u.path == "/document/delete":
            body = json.loads(self._body() or b"{}")
            did = str(body.get("id", ""))
            _doc_vec_cache.pop(did, None)   # evict cached chunk vectors
            docs = [d for d in _read_json(DOCS_FILE, []) if d["id"] != did]
            _write_json(DOCS_FILE, docs)
            return self._json(200, {"ok": True})
        if u.path == "/memory/polish":
            # Propose-only smoothing pass (Memories page button). The LLM drafts
            # cleanups off-lock; NOTHING is applied here — the user reviews each
            # suggestion and /memory/polish/apply commits the approved ones. A
            # months-old store never gets blind-rewritten by one click.
            with _lock:
                mems = vc.list_memories(_strata)
            changes = vc.polish_memory_store(mems, _mem_llm_cfg())   # slow LLM, off-lock
            # ids stringified: 64-bit ints round past 2^53 in JS
            out = [{"id": str(ch["id"]), "action": ch["action"], "old": ch["old"],
                    **({"text": ch["text"]} if ch.get("text") else {})} for ch in changes]
            return self._json(200, {"ok": True, "changes": out,
                                    "total": len(mems), "kept": len(mems) - len(out)})
        if u.path == "/memory/polish/apply":
            # Apply the user-approved smoothing suggestions. Each one re-checks
            # that the memory still reads exactly as it did at review time —
            # anything stale (edited/deleted since) is skipped, never guessed at.
            body = json.loads(self._body() or b"{}")
            applied = skipped = 0
            with _lock:
                current = {str(m["id"]): m["text"] for m in vc.list_memories(_strata)}
                for ch in body.get("changes", []):
                    mid = str(ch.get("id", ""))
                    if current.get(mid) != ch.get("old"):
                        skipped += 1
                        continue
                    try:
                        if ch.get("action") == "delete":
                            _strata.delete_memory(int(mid), mode="hard")
                        elif ch.get("action") == "rewrite" and (ch.get("text") or "").strip():
                            _strata.supersede_memory(int(mid), ch["text"].strip())
                        else:
                            skipped += 1
                            continue
                        applied += 1
                    except Exception as e:
                        print(f"[memory] polish apply failed on {mid}: {e}")
                        skipped += 1
            print(f"[memory] store polish: {applied} applied, {skipped} skipped")
            return self._json(200, {"ok": True, "applied": applied, "skipped": skipped})
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
    # Recap any past conversations missing a summary, in the background so the slow
    # summary LLM calls never delay startup or the first turn.
    threading.Thread(target=_backfill_recaps, name="recap-backfill", daemon=True).start()
    # Drain background fact-extraction jobs so that LLM call never stalls a turn.
    threading.Thread(target=_memory_worker, name="memory-worker", daemon=True).start()
    # Hands-free VAD channel: constructed here so a port conflict fails fast,
    # served on its own thread (Silero lazy-loads inside that thread on first use).
    vad_srv = HTTPServer((HOST, VAD_PORT), VadHandler)
    threading.Thread(target=vad_srv.serve_forever, name="vad-server", daemon=True).start()
    srv = HTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"\n  ✦ {ASSISTANT_NAME} is listening at {url}")
    print(f"    · hands-free VAD on :{VAD_PORT}")
    print("    Open it in your browser, click Start conversation, and just talk.\n")
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
