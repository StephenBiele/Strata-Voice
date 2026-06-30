"""Quick, reliable push-to-talk voice assistant for Apple Silicon.

A deliberately simple alternative to the vui streaming pipeline: no VAD,
no turn-endpointing guesswork, no WebRTC. You press Enter to talk, press
Enter again to stop. That removes every source of the flakiness we hit.

Pipeline:
    mic --> Parakeet V3 TDT (ASR) --> Ollama LLM --> Kokoro (TTS) --> speakers

Memory is Strata Memory (local-first SQLite canonical store). The assistant
sees your current memories every turn, and can store / supersede / forget
them via inline directives that we parse out of its reply. Memories persist
to ~/.vui/strata_memory.db — the same DB the vui integration uses, so they
carry across both.

Run:  python voicechat.py
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import tempfile
import threading
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

# ---- config (override via env) ----------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("VOICE_LLM_MODEL", "qwen3.5:4b")
ASR_MODEL = os.environ.get("VOICE_ASR_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
TTS_MODEL = os.environ.get("VOICE_TTS_MODEL", "prince-canuma/Kokoro-82M")
TTS_VOICE = os.environ.get("VOICE_TTS_VOICE", "af_heart")
DB_PATH = os.environ.get("VOICE_DB", str(Path.home() / ".vui" / "strata_memory.db"))
# Local embedding model for semantic recall (Strata's vector layer). Falls back
# to dump-all if this model isn't pulled, so the app never hard-depends on it.
EMBED_MODEL = os.environ.get("VOICE_EMBED_MODEL", "nomic-embed-text")
# Below this many active memories, inject them all (perfect recall, ~free).
# Above it, use Strata's recall() to select the most relevant for the turn.
RECALL_THRESHOLD = int(os.environ.get("VOICE_RECALL_THRESHOLD", "12"))
RECALL_TOP_K = int(os.environ.get("VOICE_RECALL_TOP_K", "8"))
MIC_SR = 16000   # Parakeet wants 16 kHz mono
TTS_SR = 24000   # Kokoro output rate

# The editable persona (exposed in Settings). Keep it about voice + tone.
PERSONA_PROMPT = """You are a warm, concise voice assistant. Keep replies short \
and natural: usually one or two sentences, since they are spoken aloud. Do not \
use markdown, lists, or emoji. Write the way people actually speak: short, plain \
sentences with simple punctuation. Avoid dashes, semicolons, and long chains of \
commas — they make the spoken voice sound choppy.

You know some things about the user — their profile and remembered facts are \
provided below. When the user asks about something you know (their name, where \
they live, a preference, anything in memory), answer directly and plainly from \
it — never refuse to recall it or tease about "not reciting." Otherwise, draw on \
what you know only when it is genuinely relevant to what they just said. Do not \
shoehorn their name or location into replies where they don't naturally belong."""

# Fixed instructions that make the memory feature work. Always appended after the
# (possibly user-edited) persona, so memory keeps working no matter what the user
# sets as their system prompt.
MEMORY_DIRECTIVES = """MEMORY DIRECTIVES — when warranted, append directive lines \
AFTER your spoken reply (the user never hears these; they are stripped):
- If the user states a durable fact about themselves (name, job, location, \
family, pets, preferences, allergies), append:  [MEM_ADD] <short fact>
- If that fact updates an existing memory, instead append:  [MEM_ADD] <new fact>
- If the user asks you to forget something, append:  [MEM_DEL] <keywords>
Only emit a directive for genuinely durable facts or explicit forget requests. \
Never emit one for small talk, questions, or transient events."""

# Always appended (not user-editable). Guarantees the model actually USES the
# profile + memories, regardless of the persona the user sets. Without this, a
# character-heavy persona (e.g. one that plays up fallibility) can make the model
# roleplay not-knowing the user even though the facts are right there in context.
RECALL_GUARD = """USING WHAT YOU KNOW: The user profile and current memories above \
are real, verified facts about this specific user that you genuinely have — not \
guesses or roleplay. When the user asks what you know about them, or asks anything \
the profile or memories cover (their name, where they live, their job, their pets, \
preferences, anything listed), answer directly and confidently from that \
information. Never claim you don't know them, can't remember, or might be \
misremembering something that is listed above — it is correct, so do not deny, \
hedge, or downplay it. Weave it in naturally instead of reciting a list."""

# Backwards-compatible default (persona + directives) for the CLI path.
SYSTEM_PROMPT = PERSONA_PROMPT + "\n\n" + MEMORY_DIRECTIVES + "\n\n" + RECALL_GUARD

_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---- audio I/O ---------------------------------------------------------------
def record_until_enter() -> np.ndarray:
    """Record mono 16 kHz audio until the user presses Enter again."""
    frames: list[np.ndarray] = []
    q: queue.Queue = queue.Queue()

    def cb(indata, frame_count, time_info, status):
        q.put(indata.copy())

    stop = threading.Event()

    def wait_enter():
        input()
        stop.set()

    threading.Thread(target=wait_enter, daemon=True).start()
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", callback=cb):
        while not stop.is_set():
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                pass
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).reshape(-1)


def play(audio: np.ndarray, sr: int) -> None:
    sd.play(audio, sr)
    sd.wait()


def patch_kokoro_tts() -> None:
    """Work around an mlx-audio Kokoro bug.

    In `SineGen.__call__`, `_f02sine` and `_f02uv` can return frame counts that
    differ by a few hundred samples, so `noise_amp * normal(sine_waves.shape)`
    fails to broadcast and the whole synthesis raises. This trips on plenty of
    ordinary sentences (not just long ones). We re-bind the method to a version
    that trims both to the shorter length (the f0-derived `uv` length is the
    correct target) before combining.
    """
    try:
        import mlx.core as mx
        from mlx_audio.tts.models.kokoro import istftnet

        def _sinegen_call(self, f0):
            fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
            sine_waves = self._f02sine(fn) * self.sine_amp
            uv = self._f02uv(f0)
            n = min(sine_waves.shape[1], uv.shape[1])
            sine_waves, uv = sine_waves[:, :n, :], uv[:, :n, :]
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * mx.random.normal(sine_waves.shape)
            sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise

        istftnet.SineGen.__call__ = _sinegen_call
        print("[tts] applied Kokoro SineGen length-alignment patch")
    except Exception as e:
        print(f"[tts] could not patch Kokoro ({e}); long sentences may fail")


# ---- LLM ---------------------------------------------------------------------
def build_messages(history, memories, documents=None, profile=None,
                   persona: str | None = None, recent=None) -> list[dict]:
    """Compose the full system prompt + history into a messages list.

    `persona` is the user-editable part; the fixed MEMORY_DIRECTIVES are
    always appended so the memory feature works regardless of edits. `recent` is
    a list of conversation recaps injected only when the retrieval layer judged
    them relevant this turn (so they're never volunteered on unrelated turns).
    """
    mem_block = "\n".join(f"- {m}" for m in memories) or "(none yet)"
    system = persona or PERSONA_PROMPT
    if profile:
        system += f"\n\n{profile}"
    system += f"\n\nCURRENT MEMORIES:\n{mem_block}"
    if recent:
        joined = "\n".join(f"- {r}" for r in recent)
        system += ("\n\nRECENT CONVERSATIONS (recaps of earlier sessions, surfaced because "
                   "they're relevant to what the user just said — use them to answer what you "
                   "talked about or to pick up where you left off):\n" + joined)
    if documents:
        joined = "\n\n".join(documents)
        system += (
            "\n\nREFERENCE DOCUMENTS the user shared (e.g. a resume or profile). "
            "Use them to answer questions accurately; quote only short snippets, "
            "never read a whole document aloud:\n" + joined
        )
    system += "\n\n" + MEMORY_DIRECTIVES
    system += "\n\n" + RECALL_GUARD
    return [{"role": "system", "content": system}, *history]


def llm_complete(messages: list[dict], cfg: dict | None = None) -> str:
    """Call the configured LLM backend (Ollama or any OpenAI-compatible API).

    cfg keys: backend ('ollama'|'openai'), thinking (bool),
      ollama_url, ollama_model, openai_base, openai_model, api_key.
    """
    cfg = cfg or {}
    backend = cfg.get("backend", "ollama")
    thinking = bool(cfg.get("thinking", False))
    if backend == "openai":
        base = (cfg.get("openai_base") or "").rstrip("/")
        model = cfg.get("openai_model") or "gpt-4o-mini"
        headers = {"Content-Type": "application/json"}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        # Cloud reasoning models can be slow — generous timeout.
        with httpx.Client(timeout=600) as client:
            r = client.post(f"{base}/chat/completions", headers=headers,
                            json={"model": model, "messages": messages,
                                  "stream": False, "temperature": 0.6})
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
    else:
        url = (cfg.get("ollama_url") or OLLAMA_URL).rstrip("/")
        model = cfg.get("ollama_model") or LLM_MODEL
        with httpx.Client(timeout=600) as client:
            r = client.post(
                f"{url}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "think": thinking, "options": {"temperature": 0.6}},
            )
            r.raise_for_status()
            content = r.json()["message"]["content"]
    if not thinking:
        content = _THINK_TAG.sub("", content)
    return content.strip()


def llm_reply(history: list[dict], memories: list[str],
              documents: list[str] | None = None,
              profile: str | None = None,
              persona: str | None = None,
              cfg: dict | None = None,
              recent=None) -> str:
    messages = build_messages(history, memories, documents, profile, persona, recent)
    return llm_complete(messages, cfg)


def llm_stream(messages: list[dict], cfg: dict | None = None):
    """Yield reply text token-by-token from the configured backend.

    Used by the streaming turn path so the server can synthesize and ship
    audio sentence-by-sentence instead of waiting for the whole reply.
    """
    cfg = cfg or {}
    backend = cfg.get("backend", "ollama")
    thinking = bool(cfg.get("thinking", False))
    if backend == "openai":
        base = (cfg.get("openai_base") or "").rstrip("/")
        model = cfg.get("openai_model") or "gpt-4o-mini"
        headers = {"Content-Type": "application/json"}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        with httpx.Client(timeout=600) as client:
            with client.stream("POST", f"{base}/chat/completions", headers=headers,
                               json={"model": model, "messages": messages,
                                     "stream": True, "temperature": 0.6}) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    tok = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if tok:
                        yield tok
    else:
        url = (cfg.get("ollama_url") or OLLAMA_URL).rstrip("/")
        model = cfg.get("ollama_model") or LLM_MODEL
        with httpx.Client(timeout=600) as client:
            with client.stream("POST", f"{url}/api/chat",
                               json={"model": model, "messages": messages, "stream": True,
                                     "think": thinking, "options": {"temperature": 0.6}}) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    tok = obj.get("message", {}).get("content")
                    if tok:
                        yield tok
                    if obj.get("done"):
                        break


# ---- memory directive parsing ------------------------------------------------
def apply_directives(strata, reply: str, memories: list[dict]) -> str:
    """Strip [MEM_ADD]/[MEM_DEL] lines from the reply and apply them to Strata.

    Returns the clean, speakable text.
    """
    spoken: list[str] = []
    for line in reply.splitlines():
        s = line.strip()
        if s.startswith("[MEM_ADD]"):
            fact = s[len("[MEM_ADD]"):].strip()
            if fact:
                _add_or_supersede(strata, fact, memories)
        elif s.startswith("[MEM_DEL]"):
            kw = s[len("[MEM_DEL]"):].strip()
            if kw:
                _forget(strata, kw, memories)
        else:
            spoken.append(line)
    return "\n".join(spoken).strip()


def _add_or_supersede(strata, fact: str, memories: list[dict]):
    """Write or update a fact. Returns the canonical record id of the written/
    updated fact, or None on an exact duplicate (nothing changed)."""
    fl = fact.lower()
    # Crude topical overlap: if an existing memory shares >=2 significant
    # words, treat this as an update (supersede) rather than a new fact.
    sig = {w for w in fl.split() if len(w) > 3}
    for m in memories:
        ml = m["text"].lower()
        if m["text"].lower() == fl:
            return None  # exact dup
        overlap = sig & {w for w in ml.split() if len(w) > 3}
        if len(overlap) >= 2:
            try:
                res = strata.supersede_memory(m["id"], fact)
                print(f"  · memory updated: {m['text']!r} -> {fact!r}")
                return res.get("id") if isinstance(res, dict) else None
            except Exception as e:
                print(f"  · supersede failed ({e}); writing as new")
                break
    res = strata.write_memory(fact)
    print(f"  · memory saved: {fact!r}")
    return res.get("id") if isinstance(res, dict) else None


def _forget(strata, keywords: str, memories: list[dict]) -> None:
    words = [w for w in keywords.lower().split() if len(w) > 2]
    hits = [m for m in memories
            if any(w in m["text"].lower() for w in words)]
    for m in hits:
        try:
            strata.delete_memory(m["id"], mode="hard")
            print(f"  · memory forgotten: {m['text']!r}")
        except Exception as e:
            print(f"  · forget failed ({e})")


# ---- memory listing ----------------------------------------------------------
def list_memories(strata) -> list[dict]:
    # Facts only (L1). Excludes L0 EPISODE events, which live on the same store but
    # are the raw episodic spine for the timeline, not distilled memories.
    from strata.canonical.records import Status, RecordType
    recs = strata.engine.store.query(record_type=RecordType.FACT, exclude_tombstoned=True)
    active = [r for r in recs if r.status in (Status.ACTIVE, Status.REINFORCED)]
    active.sort(key=lambda r: r.created_at)
    return [{"id": r.id, "text": r.content} for r in active]


# ---- semantic recall (Strata vector layer) -----------------------------------
class OllamaEmbedder:
    """Real embedding model served by the local Ollama, conforming to Strata's
    Embedder protocol (model_id, dim, embed). Lets recall() be actually semantic
    instead of the offline hash placeholder."""

    def __init__(self, model: str = EMBED_MODEL, url: str = OLLAMA_URL) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.model_id = f"ollama:{model}"
        self.dim = len(self.embed("dimension probe"))

    def embed(self, text: str) -> list[float]:
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{self.url}/api/embed",
                            json={"model": self.model, "input": text or " "})
            r.raise_for_status()
            data = r.json()
        # /api/embed returns {"embeddings":[[...]]}; tolerate the older shape too.
        if "embeddings" in data:
            return data["embeddings"][0]
        return data["embedding"]


def make_embedder():
    """Return an OllamaEmbedder if the embed model is available, else None so the
    caller falls back to the offline default (and the dump-all recall path)."""
    try:
        emb = OllamaEmbedder()
        print(f"  · semantic recall: ON ({emb.model_id}, dim={emb.dim})")
        return emb
    except Exception as e:
        print(f"  · semantic recall: OFF (embed model unavailable: {e})")
        return None


def warm_index(strata) -> int:
    """The vector store is in-memory and starts empty each run, so embed all active
    canonical records once at startup. Returns how many were indexed."""
    ids = [m["id"] for m in list_memories(strata)]
    if not ids:
        return 0
    eng = strata.engine
    try:
        eng.reindex_into(eng.active_generation, eng.embedder, ids)
    except Exception as e:
        print(f"  · index warm-up failed: {e}")
        return 0
    return len(ids)


def recall_memories(strata, query: str, top_k: int = RECALL_TOP_K) -> list[str]:
    """Return the most relevant memory claims for this query via Strata's belief
    bundle (vector + lexical + resolver). Falls back to listing on any error."""
    try:
        bundle = strata.recall(query or "", top_k=top_k, diversity=True)
        claims, seen = [], set()
        for cat in ("current_beliefs", "recent_context", "interaction_guidance",
                    "open_conflicts", "hypotheses"):
            for e in bundle.get(cat, []):
                c = (e.get("claim") or "").strip()
                if c and c not in seen:
                    seen.add(c)
                    claims.append(c)
        return claims[:top_k] if claims else [m["text"] for m in list_memories(strata)]
    except Exception as e:
        print(f"[recall] fell back to list: {e}")
        return [m["text"] for m in list_memories(strata)]


def select_memories(strata, query: str, *, semantic: bool = True,
                    threshold: int = RECALL_THRESHOLD) -> list[str]:
    """Pick the memory texts to inject this turn. Small store -> inject everything
    (perfect, ~free). Large store -> semantic recall of the most relevant few."""
    mems = list_memories(strata)
    if not semantic or len(mems) <= threshold:
        return [m["text"] for m in mems]
    return recall_memories(strata, query)


# ---- deterministic memory capture (no LLM) -----------------------------------
# Rule-based extraction from the user's own words. Runs every turn regardless of
# what the model does, so memory no longer depends on the LLM emitting directives.
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,!?;:'\"").strip()


def capture_memory(strata, user_text: str, event_id: int | None = None) -> list[str]:
    """Extract durable facts / forget-requests from a user utterance and apply
    them to Strata directly. Returns a list of change descriptions for logging.

    If ``event_id`` is given, each newly added fact is linked to that source
    event (so the timeline knows when/where the fact was learned)."""
    text = (user_text or "").strip()
    if not text:
        return []
    memories = list_memories(strata)
    changes: list[str] = []

    def add(fact: str) -> None:
        fact = _clean(fact)
        if len(fact) < 2:
            return
        fact = fact[0].upper() + fact[1:]
        before = {m["text"] for m in memories}
        rid = _add_or_supersede(strata, fact, memories)
        memories.append({"id": rid or -1, "text": fact})   # keep snapshot fresh for dedup
        if rid and event_id:
            _link_source(strata, rid, event_id)
        if fact not in before:
            changes.append(f"+ {fact}")

    # 1) Explicit forget — takes the whole turn.
    m = re.search(r"\b(?:forget|delete)\b(?:\s+(?:about|the\s+memory\s+about|that|my))?\s+(.+)", text, re.I)
    if m:
        kw = _clean(m.group(1))
        if kw:
            _forget(strata, kw, memories)
            return [f"- forget: {kw}"]

    # 2) Explicit remember — store the clause verbatim.
    m = re.search(r"\b(?:remember|make a note|note that|keep in mind|don'?t forget)\b(?:\s+that)?\s+(.+)", text, re.I)
    if m:
        add(m.group(1))
        return changes

    # 3) Occupation — preserve the preposition so it reads naturally.
    m = re.search(r"\bi work (as|at|in)\s+([^.!?]+)", text, re.I)
    if m:
        add(f"Works {m.group(1).lower()} {_clean(m.group(2))}")
    m = re.search(r"\bmy job is\s+([^.!?]+)", text, re.I)
    if m:
        add(f"Job: {_clean(m.group(1))}")

    # 4) Other high-confidence durable patterns (each independent).
    #    Location/allergy allow commas (e.g. "Austin, Texas"); preferences stop
    #    at a comma to avoid swallowing a following clause.
    pats = [
        (r"\b(?:my name is|i am called|i'?m called|you can call me|call me)\s+([A-Za-z][A-Za-z'’\-]+(?:\s+[A-Za-z][A-Za-z'’\-]+)?)", "Name is {}"),
        (r"\b(?:i live in|i'?m from|i am from|i'?m based in|i am based in|i'?m located in)\s+([^.!?]+?)(?:\s+and\b|[.!?]|$)", "Lives in {}"),
        (r"\b(?:i'?m allergic to|i am allergic to|allergic to)\s+([^.!?]+?)(?:\s+and\b|[.!?]|$)", "Allergic to {}"),
        (r"\bi (?:really )?like\s+([^.,!?]+)", "Likes {}"),
        (r"\bi (?:really )?love\s+([^.,!?]+)", "Loves {}"),
        (r"\bi (?:really )?enjoy\s+([^.,!?]+)", "Enjoys {}"),
        (r"\bi (?:really )?prefer\s+([^.,!?]+)", "Prefers {}"),
        (r"\bi (?:really )?(?:hate|dislike|don'?t like)\s+([^.,!?]+)", "Dislikes {}"),
    ]
    for pat, tmpl in pats:
        m = re.search(pat, text, re.I)
        if m:
            add(tmpl.format(_clean(m.group(1))))

    # 4) Pets / family — only for a known set of nouns to avoid false positives.
    m = re.search(r"\bi have (?:a |an )?(dog|cat|son|daughter|wife|husband|partner|"
                  r"brother|sister|kid|child|baby|pet)\b(?:\s+(?:named|called)\s+([A-Za-z]+))?",
                  text, re.I)
    if m:
        noun, name = m.group(1), m.group(2)
        add(f"Has a {noun}" + (f" named {name}" if name else ""))

    return changes


# ---- LLM extraction pass (covers natural speech the patterns miss) -----------
_EXTRACT_PROMPT = """You pull durable facts about the user out of their latest message, for long-term memory.
Return ONLY a JSON array of short factual strings written in the third person, e.g.
["Has a dog named Rex", "Works as a nurse", "Restoring a vintage motorcycle", "Allergic to shellfish"].
Include stable things: preferences, relationships, family, pets, job, location, hobbies, ongoing projects, health, and notable life facts.
Exclude: questions, comments about the assistant, fleeting events with no lasting significance, pure small talk, and anything already in existing memory.
If nothing durable is stated, return [].

Existing memory (do not duplicate):
{existing}

User message:
"{text}"

JSON array:"""


def extract_facts_llm(user_text: str, existing: list[str], cfg: dict | None = None) -> list[str]:
    """Ask the configured model to extract durable facts as a JSON array.
    Runs after the spoken reply, so it never adds latency to speech."""
    text = (user_text or "").strip()
    if not text:
        return []
    prompt = _EXTRACT_PROMPT.format(
        existing="\n".join(f"- {m}" for m in existing) or "(none)", text=text)
    try:
        raw = llm_complete([{"role": "user", "content": prompt}],
                           {**(cfg or {}), "thinking": False})
    except Exception as e:
        print(f"[memory] extraction call failed: {e}")
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for x in arr:
        if isinstance(x, str) and len(_clean(x)) > 2:
            out.append(_clean(x))
    return out[:5]


def add_facts(strata, facts: list[str], event_id: int | None = None) -> list[str]:
    """Persist extracted facts via the same dedup/supersede path. Returns new ones.
    Links each new fact to ``event_id`` (its source turn) when provided."""
    if not facts:
        return []
    memories = list_memories(strata)
    added = []
    for f in facts:
        before = {m["text"] for m in memories}
        rid = _add_or_supersede(strata, f, memories)
        memories.append({"id": rid or -1, "text": f})
        if rid and event_id:
            _link_source(strata, rid, event_id)
        if f not in before:
            added.append(f)
    return added


# ---- events & timeline (episodic spine) --------------------------------------
def record_event(strata, text: str, *, ts_ms: int | None = None) -> int | None:
    """Write a conversational turn as a raw L0 event (the episodic spine). Returns
    the event's canonical id, or None on failure. ``ts_ms`` backfills the time."""
    from strata.canonical.records import RecordType
    text = (text or "").strip()
    if not text:
        return None
    fields = {"record_subtype": "turn"}
    if ts_ms is not None:
        fields["created_at"] = ts_ms
    try:
        res = strata.write_event(text, record_type=RecordType.EPISODE, **fields)
        return res.get("id") if isinstance(res, dict) else None
    except Exception as e:
        print(f"[event] write failed: {e}")
        return None


def _link_source(strata, fact_id: int, event_id: int) -> None:
    try:
        strata.link_source(fact_id, event_id)
    except Exception as e:
        print(f"[event] link failed: {e}")


def list_events(strata) -> list[dict]:
    """All conversational-turn events, oldest first."""
    from strata.canonical.records import RecordType, Tier
    recs = strata.engine.store.query(tier=Tier.L0, record_type=RecordType.EPISODE,
                                     exclude_tombstoned=True)
    recs = [r for r in recs if r.record_subtype == "turn"]
    recs.sort(key=lambda r: r.created_at)
    return [{"id": r.id, "text": r.content, "t": r.created_at} for r in recs]


def build_timeline(strata, limit: int = 300) -> list[dict]:
    """Chronological moments (newest first). Each: the utterance and the facts that
    were learned from it. Facts hang off the event they were derived from."""
    events = list_events(strata)
    id_to_text = {m["id"]: m["text"] for m in list_memories(strata)}
    store = strata.engine.store
    out = []
    for e in events:
        facts = []
        try:
            for fid in store.derived_from(e["id"]):
                if fid in id_to_text:            # only still-active facts
                    facts.append(id_to_text[fid])
        except Exception:
            pass
        out.append({"id": str(e["id"]), "t": e["t"], "text": e["text"], "facts": facts})
    out.sort(key=lambda m: m["t"], reverse=True)
    return out[:limit]


# ---- conversation recaps + relevance-gated injection -------------------------
# Each finished conversation gets a 1-2 sentence recap stored in Strata's episodic
# layer (L0 EPISODE, subtype "summary"). A retrieval layer decides each turn whether
# any recap is relevant — by embedding similarity, not by asking the model to behave.
_SUMMARY_PROMPT = """Summarize this conversation in ONE or TWO sentences, third person, \
focused on what the user talked about, asked for, or planned — phrased so a future you can \
recall "what we talked about." No preamble; output only the summary.

Conversation:
{transcript}

Summary:"""


def summarize_session(turns: list[dict], cfg: dict | None = None) -> str:
    rows = [t for t in (turns or []) if t.get("role") in ("user", "assistant")]
    if not rows:
        return ""
    transcript = "\n".join(
        f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}" for t in rows[-40:])
    try:
        out = llm_complete(
            [{"role": "user", "content": _SUMMARY_PROMPT.format(transcript=transcript)}],
            {**(cfg or {}), "thinking": False})
        out = re.sub(r"\s+", " ", out).strip()
        if out:
            return out[:300]
    except Exception as e:
        print(f"[recap] summarize failed: {e}")
    first = next((t["content"] for t in rows if t.get("role") == "user"), "")
    return re.sub(r"\s+", " ", first).strip()[:140]


def record_summary(strata, text: str, *, ts_ms: int | None = None) -> int | None:
    """Store a conversation recap as an L0 EPISODE (subtype 'summary'). Kept out of
    the durable-fact path (list_memories is FACT-only) and the turn timeline."""
    from strata.canonical.records import RecordType
    text = (text or "").strip()
    if not text:
        return None
    fields = {"record_subtype": "summary"}
    if ts_ms is not None:
        fields["created_at"] = ts_ms
    try:
        res = strata.write_event(text, record_type=RecordType.EPISODE, **fields)
        return res.get("id") if isinstance(res, dict) else None
    except Exception as e:
        print(f"[recap] write failed: {e}")
        return None


def list_summaries(strata) -> list[dict]:
    """Conversation recaps, newest first."""
    from strata.canonical.records import RecordType, Tier
    recs = strata.engine.store.query(tier=Tier.L0, record_type=RecordType.EPISODE,
                                     exclude_tombstoned=True)
    recs = [r for r in recs if r.record_subtype == "summary"]
    recs.sort(key=lambda r: r.created_at, reverse=True)
    return [{"id": r.id, "text": r.content, "t": r.created_at} for r in recs]


# Canonical "asking to recall / continue" phrasings. The user's message is compared
# to these by embedding, so meta-recall is caught even with no shared topic words.
_RECALL_EXEMPLARS = [
    "what were we just talking about",
    "what did we talk about last time",
    "what were we discussing",
    "continue where we left off",
    "let's pick up where we left off",
    "remind me what we talked about",
    "catch me up on our last conversation",
    "what did we discuss earlier",
]
_CONTINUITY_RE = re.compile(
    r"\b(what (were|did) we (just )?(talk|talking|discuss|discussing)"
    r"|(continue|pick up) (where|from) we left off"
    r"|last (time|conversation|chat)|earlier (you|we)"
    r"|we were (talking|discussing)|catch me up|recap"
    r"|remind me what we|what did we talk about)\b", re.I)

_exemplar_vecs: dict = {}   # model_id -> [vec, ...]
_recap_vecs: dict = {}      # summary id -> vec


def wants_recent_context(text: str) -> bool:
    """Keyword fallback for meta-recall when no embedder is available."""
    return bool(_CONTINUITY_RE.search(text or ""))


def _cosine(a, b) -> float:
    from strata.vector.embedder import cosine
    try:
        return cosine(a, b)
    except Exception:
        return 0.0


def relevant_recaps(strata, query: str, embedder, *, max_n: int = 3,
                    meta_thresh: float = 0.62, topical_thresh: float = 0.50) -> list[str]:
    """The secondary trigger: return the conversation recaps worth injecting this
    turn — by embedding relevance, independent of the model. Meta-recall intent is
    matched against canonical exemplars; topical references against the recaps
    themselves. Returns [] (inject nothing) when nothing is relevant."""
    summaries = list_summaries(strata)   # newest first
    if not summaries:
        return []
    query = (query or "").strip()
    if embedder is None:
        return [s["text"] for s in summaries[:max_n]] if wants_recent_context(query) else []
    try:
        qv = embedder.embed(query)
    except Exception as e:
        print(f"[recap] query embed failed: {e}")
        return [s["text"] for s in summaries[:max_n]] if wants_recent_context(query) else []

    vecs = _exemplar_vecs.get(embedder.model_id)
    if vecs is None:
        try:
            vecs = [embedder.embed(x) for x in _RECALL_EXEMPLARS]
        except Exception:
            vecs = []
        _exemplar_vecs[embedder.model_id] = vecs
    meta = any(_cosine(qv, ev) >= meta_thresh for ev in vecs)

    selected = list(summaries[:max_n]) if meta else []
    for s in summaries:
        rv = _recap_vecs.get(s["id"])
        if rv is None:
            try:
                rv = embedder.embed(s["text"])
            except Exception:
                rv = None
            _recap_vecs[s["id"]] = rv
        if rv is not None and _cosine(qv, rv) >= topical_thresh:
            selected.append(s)

    seen, out = set(), []
    for s in selected:
        if s["id"] not in seen:
            seen.add(s["id"])
            out.append(s["text"])
        if len(out) >= max_n:
            break
    return out


# ---- conversation review -----------------------------------------------------
# Replays a saved transcript to (a) recover durable facts the live pass missed
# and fold them into memory, and (b) grade how the assistant behaved. The facts
# are what actually "improve the model": they become context on future turns.
_REVIEW_PROMPT = """You are grading a past voice conversation. Judge ONLY the assistant's behavior.
Return ONLY JSON, no prose:
{{"recall":"good|mixed|poor","persona":"good|mixed|poor","brevity":"good|mixed|poor","summary":"<=14 words","tips":["short fix", "..."]}}
- recall: did it correctly use what it knows about the user when relevant, and never deny a fact it should know?
- persona: did it stay natural and in character?
- brevity: were replies short enough for spoken voice (about 1-3 sentences)?
Give 0-3 concrete tips. If it did well, tips can be [].

Transcript:
{transcript}

JSON:"""


def _transcript_text(turns: list[dict], limit: int = 40) -> str:
    rows = [t for t in turns if t.get("role") in ("user", "assistant")][-limit:]
    return "\n".join(f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
                     for t in rows)


def review_session(strata, turns: list[dict], cfg: dict | None = None) -> dict:
    """Returns {added: [new facts], assessment: {...}}."""
    user_text = "\n".join(t["content"] for t in turns if t.get("role") == "user")
    existing = [m["text"] for m in list_memories(strata)]
    added = add_facts(strata, extract_facts_llm(user_text, existing, cfg))

    assessment = {}
    try:
        raw = llm_complete(
            [{"role": "user", "content": _REVIEW_PROMPT.format(
                transcript=_transcript_text(turns))}],
            {**(cfg or {}), "thinking": False})
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            assessment = json.loads(m.group(0))
    except Exception as e:
        print(f"[review] assessment failed: {e}")
    return {"added": added, "assessment": assessment}


# ---- memory hygiene (contradiction / duplicate audit) ------------------------
_AUDIT_PROMPT = """You clean up a user's long-term memory. Below are stored memory facts, each with an id.
Flag ONLY entries that should change, and classify each with a "kind":
- "contradiction": two facts that genuinely cannot both be true at the same time
  (e.g. "Lives in France" vs "Lives in Japan", "Has no pets" vs "Has a dog"). Keep the more recent/specific; remove the other.
- "redundant": one fact is fully implied by ANOTHER, more specific fact IN THIS SAME LIST
  (e.g. "Based in the United States" is redundant only if another listed fact says "Lives in Colorado").
  Keep the specific one; remove the general one. If no more-specific fact is present, do NOT flag it.
- "duplicate": the exact same fact written two ways. Keep the clearest; remove the rest.
- "junk": not a real durable fact — a fragment, a question, or a placeholder like "About me". keep is null.

CRITICAL RULE: facts that are all TRUE but at different levels of detail are NOT contradictions.
"Lives in the US" and "Lives in Colorado" do not conflict — at most the broader one is "redundant",
and ONLY when both are present in the list. Never call something a contradiction just because one
fact is more or less specific than another. Do not invent overlaps that aren't in the list.

Return ONLY a JSON array ([] if the memory is already clean). Each element:
{{"kind":"contradiction|redundant|duplicate|junk","remove":[<id>,...],"keep":<id or null>,"reason":"<short why>"}}

Memory facts:
{facts}

JSON array:"""


def audit_memories(strata, cfg: dict | None = None) -> list[dict]:
    """Flag contradictions / redundancy / duplicates / junk among the stored
    memories. Returns issues with resolved text, ready for the user to confirm.
    Compares memories only against each other — nothing is deleted here."""
    mems = list_memories(strata)
    if not mems:
        return []
    by_id = {m["id"]: m["text"] for m in mems}
    facts = "\n".join(f"[{m['id']}] {m['text']}" for m in mems)
    try:
        raw = llm_complete(
            [{"role": "user", "content": _AUDIT_PROMPT.format(facts=facts)}],
            {**(cfg or {}), "thinking": False})
    except Exception as e:
        print(f"[audit] call failed: {e}")
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    issues = []
    valid_kinds = {"contradiction", "redundant", "duplicate", "junk"}
    for it in arr if isinstance(arr, list) else []:
        try:
            remove = [int(i) for i in (it.get("remove") or []) if int(i) in by_id]
        except Exception:
            remove = []
        if not remove:
            continue
        keep = it.get("keep")
        keep = int(keep) if isinstance(keep, (int, str)) and str(keep).lstrip("-").isdigit() else None
        kind = str(it.get("kind", "")).lower().strip()
        issues.append({
            "kind": kind if kind in valid_kinds else "duplicate",
            "remove": [str(i) for i in remove],  # stringify: JS rounds 64-bit ints
            "remove_texts": [by_id[i] for i in remove],
            "keep": keep if keep in by_id else None,
            "keep_text": by_id.get(keep) if keep in by_id else None,
            "reason": str(it.get("reason", "")).strip(),
        })
    return issues


# ---- recall judge (semantic, entailment-aware) -------------------------------
_JUDGE_PROMPT = """A user has these TRUE facts about them (numbered):
{facts}

The assistant was asked what it knows about the user and replied:
"{answer}"

For EACH numbered fact, decide whether the reply conveys or clearly IMPLIES it.
Use common-sense entailment: a specific statement implies a broader one — mentioning
"Colorado" implies "based in the United States"; "has a dog named Molly" implies having a pet.
Return ONLY a JSON array of objects, one per fact: {{"n":<number>,"recalled":true|false}}."""


def judge_recall(facts: list[str], answer: str, cfg: dict | None = None) -> list[bool]:
    """Return a recalled/not list aligned to `facts`, judged semantically by the
    model (so entailment counts), rather than brittle keyword matching."""
    if not facts:
        return []
    numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    try:
        raw = llm_complete(
            [{"role": "user", "content": _JUDGE_PROMPT.format(
                facts=numbered, answer=answer or "")}],
            {**(cfg or {}), "thinking": False})
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        arr = json.loads(m.group(0)) if m else []
    except Exception as e:
        print(f"[eval] judge failed: {e}")
        return [False] * len(facts)
    verdict = {}
    for o in arr if isinstance(arr, list) else []:
        try:
            verdict[int(o.get("n"))] = bool(o.get("recalled"))
        except Exception:
            pass
    return [verdict.get(i + 1, False) for i in range(len(facts))]




# ---- main loop ---------------------------------------------------------------
def main() -> int:
    print("Loading models (first run downloads them)...")
    import parakeet_mlx
    from mlx_audio.tts.utils import load_model
    from strata.gateway.api import Strata

    asr = parakeet_mlx.from_pretrained(ASR_MODEL)
    tts = load_model(TTS_MODEL)
    patch_kokoro_tts()
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    strata = Strata.open(db_path=DB_PATH)
    print(f"Ready. LLM={LLM_MODEL}  ASR=Parakeet-V3  TTS=Kokoro  DB={DB_PATH}")

    mem = list_memories(strata)
    if mem:
        print(f"({len(mem)} memories loaded)")

    history: list[dict] = []
    tmp = Path(tempfile.gettempdir()) / "voicechat_turn.wav"

    print("\n--- Push to talk. Press Enter to start, Enter again to stop. "
          "Ctrl-C to quit. ---")
    try:
        while True:
            input("\n[Enter] to speak > ")
            print("Recording... (Enter to stop)")
            audio = record_until_enter()
            if audio.size < MIC_SR // 2:  # < 0.5s
                print("(too short, skipping)")
                continue

            sf.write(str(tmp), audio, MIC_SR)
            text = asr.transcribe(str(tmp)).text.strip()
            if not text:
                print("(no speech detected)")
                continue
            print(f"You: {text}")

            history.append({"role": "user", "content": text})
            mem = list_memories(strata)
            reply_raw = llm_reply(history, [m["text"] for m in mem])
            reply = apply_directives(strata, reply_raw, mem)
            history.append({"role": "assistant", "content": reply})
            print(f"Bot: {reply}")

            if reply:
                segs = list(tts.generate(reply, voice=TTS_VOICE, lang_code="a"))
                if segs:
                    out = np.concatenate([np.asarray(s.audio) for s in segs])
                    play(out, TTS_SR)
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        strata.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
