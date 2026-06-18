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
MIC_SR = 16000   # Parakeet wants 16 kHz mono
TTS_SR = 24000   # Kokoro output rate

# The editable persona (exposed in Settings). Keep it about voice + tone.
PERSONA_PROMPT = """You are a warm, concise voice assistant. Keep replies short \
and natural — one or two sentences, since they will be spoken aloud. Do not use \
markdown, lists, or emoji.

You have a persistent memory about the user. Use it naturally; never read it back \
as a list unless asked."""

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

# Backwards-compatible default (persona + directives) for the CLI path.
SYSTEM_PROMPT = PERSONA_PROMPT + "\n\n" + MEMORY_DIRECTIVES

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


# ---- LLM ---------------------------------------------------------------------
def build_messages(history, memories, documents=None, profile=None,
                   persona: str | None = None) -> list[dict]:
    """Compose the full system prompt + history into a messages list.

    `persona` is the user-editable part; the fixed MEMORY_DIRECTIVES are
    always appended so the memory feature works regardless of edits.
    """
    mem_block = "\n".join(f"- {m}" for m in memories) or "(none yet)"
    system = persona or PERSONA_PROMPT
    if profile:
        system += f"\n\n{profile}"
    system += f"\n\nCURRENT MEMORIES:\n{mem_block}"
    if documents:
        joined = "\n\n".join(documents)
        system += (
            "\n\nREFERENCE DOCUMENTS the user shared (e.g. a resume or profile). "
            "Use them to answer questions accurately; quote only short snippets, "
            "never read a whole document aloud:\n" + joined
        )
    system += "\n\n" + MEMORY_DIRECTIVES
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
              cfg: dict | None = None) -> str:
    messages = build_messages(history, memories, documents, profile, persona)
    return llm_complete(messages, cfg)


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


def _add_or_supersede(strata, fact: str, memories: list[dict]) -> None:
    fl = fact.lower()
    # Crude topical overlap: if an existing memory shares >=2 significant
    # words, treat this as an update (supersede) rather than a new fact.
    sig = {w for w in fl.split() if len(w) > 3}
    for m in memories:
        ml = m["text"].lower()
        if m["text"].lower() == fl:
            return  # exact dup
        overlap = sig & {w for w in ml.split() if len(w) > 3}
        if len(overlap) >= 2:
            try:
                strata.supersede_memory(m["id"], fact)
                print(f"  · memory updated: {m['text']!r} -> {fact!r}")
                return
            except Exception as e:
                print(f"  · supersede failed ({e}); writing as new")
                break
    strata.write_memory(fact)
    print(f"  · memory saved: {fact!r}")


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
    from strata.canonical.records import Status
    recs = strata.engine.store.query(exclude_tombstoned=True)
    active = [r for r in recs if r.status in (Status.ACTIVE, Status.REINFORCED)]
    active.sort(key=lambda r: r.created_at)
    return [{"id": r.id, "text": r.content} for r in active]


# ---- main loop ---------------------------------------------------------------
def main() -> int:
    print("Loading models (first run downloads them)...")
    import parakeet_mlx
    from mlx_audio.tts.utils import load_model
    from strata.gateway.api import Strata

    asr = parakeet_mlx.from_pretrained(ASR_MODEL)
    tts = load_model(TTS_MODEL)
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
