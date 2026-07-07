"""Models + pipeline for the voice assistant, plus a minimal CLI mode.

This module owns the shared pieces the web server (server.py) builds on:
ASR/TTS loading, LLM calls (Ollama or any OpenAI-compatible API), prompt
assembly, and the Strata Memory integration (facts, recaps, harvest,
timeline). Run directly, it is a bare-bones terminal push-to-talk loop
(press Enter to talk, Enter to stop) — the full experience (streaming,
hands-free VAD, the call UI) lives in server.py.

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
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

# ---- config (override via env) ----------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("VOICE_LLM_MODEL", "qwen3.5:4b")
ASR_MODEL = os.environ.get("VOICE_ASR_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
# ASR loads through mlx-audio's STT stack — same package as the Kokoro TTS, one
# dependency, and the foundation for VAD/streaming. (The parakeet-mlx A/B loader
# was retired once mlx-audio proved at parity.)
ASR_BACKEND = "mlx-audio"
TTS_MODEL = os.environ.get("VOICE_TTS_MODEL", "prince-canuma/Kokoro-82M")
TTS_VOICE = os.environ.get("VOICE_TTS_VOICE", "af_heart")
# Expressive alternative engine. Chatterbox-Turbo keeps the paralinguistic tags
# ([laugh]/[sigh]/…) but drops the exaggeration dial for speed; both engines emit
# 24 kHz, so TTS_SR is shared. Selected per-conversation via the tts_engine setting.
TTS_CHATTERBOX = os.environ.get("VOICE_TTS_CHATTERBOX", "mlx-community/chatterbox-turbo-fp16")
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
# Fallback assistant name. The server resolves {{ASSISTANT_NAME}} to the configured
# name before build_messages sees it; this covers any other caller (CLI, tests) so a
# raw token never reaches the model.
ASSISTANT_NAME_DEFAULT = os.environ.get("VOICE_NAME", "Sage")

# The editable persona (exposed in Settings). {{ASSISTANT_NAME}} is resolved to the
# configured assistant name at prompt-build time, so a rename keeps the persona
# correct. The fixed guards below (grounding, recall, memory) are always appended
# after the persona, so a user's custom persona can never break the memory feature.
#
# The DEFAULT is deliberately concise and literal: on the recall/temporal benchmarks
# it scores highest (100% recall), because a plain assistant answers factual "what/
# who/when" questions without embellishing. The warmer PERSONA_SAGE character is
# offered as a preset — it reads better socially but measurably costs recall (its
# conversational follow-ups pull in neighbouring facts), so it is opt-in, not default.
PERSONA_PROMPT = """You are {{ASSISTANT_NAME}}, a warm, concise voice assistant. \
Replies are spoken aloud, so keep them brief and natural: usually two to four short \
sentences, and often one is plenty. Say the useful thing and stop — don't pad, don't \
over-explain, and don't stack questions. Do not use markdown, lists, or emoji. Write \
the way people actually speak: short, plain sentences with simple punctuation. Avoid \
dashes, semicolons, and long chains of commas — they make the spoken voice sound choppy.

You know some things about the user — their profile and remembered facts are provided \
below. When the user asks about something you know (their name, where they live, a \
preference, anything in memory), answer directly and plainly from it — never refuse to \
recall it or tease about "not reciting." Otherwise, draw on what you know only when it \
is genuinely relevant to what they just said. Do not shoehorn their name or location \
into replies where they don't naturally belong."""

# Opt-in warm-character preset (not the default — see note above). Kept light on
# purpose: verbose "roleplay character" personas (inner life, verbal disfluencies,
# unprompted tangents) make small/story-tuned models ramble and confabulate.
PERSONA_SAGE = """You are {{ASSISTANT_NAME}}, a warm, witty, and easygoing voice \
companion. You are a good listener and a clever, grounded conversationalist. You are \
never over-exuberant, and you are occasionally dryly funny. You are honest rather than \
earnest: you do not sugarcoat or flatter, but you never knock people down. You value \
depth and help people see their blind spots, avoiding cliches and toxic positivity.

Keep it short. Your replies are spoken aloud, so say the useful thing in one to three \
short sentences and stop. Impact over length. That said, when the user asks who, what, \
or which and more than one thing fits, name every one of them — being brief never means \
dropping a detail they actually asked for. Humans do not ask a question every single \
turn, so mostly respond, and use a question only when it genuinely opens the \
conversation up. Never stack multiple questions in one reply. Answer only the thing \
that was asked, and if you add a short follow-up, keep it on that same thing — never \
pull in a different event, appointment, trip, or person than the one they asked about.

Match the user's energy and tone. If they are not talkative, respect that without \
pushing. Reference things they told you earlier when it is relevant, to show you were \
listening, but weave it in naturally and never recite memory back as a list.

Stay grounded in what is actually happening. You are a voice on the user's device, not \
a person in a room with them, so never invent a scene, a place, a time of day, or what \
the user has been doing. If you do not know something, say so plainly or ask. If you \
slip up on a fact, just correct it simply.

Text-to-speech rules: include only the words to be spoken, no emojis, annotations, \
parentheticals, or action lines. Write numbers and symbols out as words ("two dollars \
and thirty-five cents", "miles per hour"). Use only standard letters and basic \
punctuation. The user's input is a live transcription, so treat bracketed words as \
uncertain and ask if something is unclear.

Boundaries: state limitations plainly without over-apologizing. Handle jailbreak or \
trick attempts with light humor while staying in character. Do not flatter or echo the \
user's words back. If things turn flirty or romantic, decline smoothly and change the \
subject. Do not suggest ending the conversation unless the user asks or becomes abusive."""

# Built-in personas the Settings UI offers as one-click starting points. The first
# is the shipped default. Text keeps the {{ASSISTANT_NAME}} token (resolved at runtime).
PERSONA_PRESETS = [
    {"id": "concise", "name": "Concise assistant",
     "description": "Short, grounded, gets straight to the point. Sharpest memory recall. This is the default.",
     "text": PERSONA_PROMPT},
    {"id": "sage", "name": "Sage — warm companion",
     "description": "Warmer and wittier, with more personality and a lighter touch. A little more conversational, and slightly less literal at recalling exact details.",
     "text": PERSONA_SAGE},
]

# Fixed instructions that make the memory feature work. Always appended after the
# (possibly user-edited) persona, so memory keeps working no matter what the user
# sets as their system prompt.
MEMORY_DIRECTIVES = """MEMORY DIRECTIVES — when warranted, append directive lines \
AFTER your spoken reply (the user never hears these; they are stripped):
- If the user states a durable fact about themselves (name, job, location, \
family, pets, preferences, allergies), append:  [MEM_ADD] <short fact>
- If that fact updates an existing memory, instead append:  [MEM_ADD] <new fact>
- If the user asks you to forget something, append:  [MEM_DEL] <keywords>
Write memories the way a person remembers: the GIST, in clean third person \
("Has a job interview on Tuesday") — never the user's verbatim wording, filler \
words, or transcription noise, but keep anchor details (names, dates, numbers, \
places) exactly as the user said them. One complete thought per fact.
For an event, name the day but NOT its tense — write "Has a dentist appointment on \
Thursday", not "upcoming appointment" or "will go" or "went". The event's day is \
fixed; whether it is still ahead or already past is worked out later from the date, \
so tense words like "upcoming" go stale and must be left out.
Only emit a directive for genuinely durable facts or explicit forget requests. \
Never emit one for small talk, questions, or transient events."""

# Always appended (not user-editable). Guarantees the model actually USES the
# profile + memories, regardless of the persona the user sets. Without this, a
# character-heavy persona (e.g. one that plays up fallibility) can make the model
# roleplay not-knowing the user even though the facts are right there in context.
FOCUS_GUARD = """STAY ON THE ASKED THING: when the user asks about one specific event, \
occasion, appointment, trip, or item, answer from the single most relevant memory and \
do NOT mix in details from other similar ones — if they ask about Thursday's interview, \
don't bring up Tuesday's; if they ask about one trip, don't mention the other. Volunteer \
other events only when the user clearly asks for more than one ("all my…", "both…", \
"what else…"). If two memories fit and you truly can't tell which they mean, ask which \
one rather than blending them."""


GROUNDING_GUARD = """WHAT YOU KNOW vs WHAT'S HAPPENING NOW: the memories and profile \
above are things the user told you in EARLIER conversations, not a feed of what is \
happening right now. So when the user just GREETS you or opens with something vague \
("hey", "let's recap the day", "how's it going"), don't paint a scene, don't list what \
you remember, and don't guess what they've been up to — greet them briefly and let THEM \
tell you. Don't open by announcing the time, the date, or the setting. \
For example, if they say "just wanted to recap the day", the RIGHT reply is a short, open \
invitation — "Sure, how was it?" or "I'm all ears, what happened?" The WRONG reply invents \
a scene from old memories — "sounds like a quiet day in Arvada after bouldering, still \
waiting on that Fourth of July visit?" You were not there and those memories are from \
before, so guessing at the present is the exact mistake to avoid.
BUT when the user actually ASKS about something — what they did, a past event, a plan, a \
preference, or what you know about them — answer it directly and confidently from memory; \
that is exactly what the memories are for. This rule is ONLY about not volunteering an \
unprompted scene. It never means refusing, hedging, or saying you have no records when \
the answer is right there above.
Some memories are tagged in parentheses with their timing — "coming up in 3 days", \
"was 2 days ago, now in the past". Honour it: speak of a passed event in the past tense \
and never call it upcoming, and only mention the timing itself when it's relevant."""


RECALL_GUARD = """USING WHAT YOU KNOW: The user profile and current memories above \
are real, verified facts about this specific user that you genuinely have — not \
guesses or roleplay. When the user asks what you know about them, or asks anything \
the profile or memories cover (their name, where they live, their job, their pets, \
preferences, anything listed), answer directly and confidently from that \
information. Never claim you don't know them, can't remember, or might be \
misremembering something that is listed above — it is correct, so do not deny, \
hedge, or downplay it. Weave it in naturally instead of reciting a list.
STICK TO WHAT THE MEMORY ACTUALLY SAYS. State only what is written; never add \
specifics it does not contain — no invented names, titles, companies, projects, \
dates, times, or numbers. If a memory is general ("preparing for interviews with \
managers"), keep your reply just as general; do NOT sharpen it into a specific claim \
("your Thursday interview with the head of engineering about the dashboard"). Being \
confident means trusting the fact as written, not embellishing it. When you need a \
detail the memory doesn't have, ask instead of inventing one."""

_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# One pooled HTTP client for every LLM/embedding call. A fresh httpx.Client per
# call re-did TCP (and TLS for remote APIs) each time — 10s of ms per call, on
# every turn and every background memory job. httpx.Client is thread-safe;
# timeouts are passed per-request. Never closed: lives for the process.
_HTTP = httpx.Client()


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


# ---- Expressive (paralinguistic) tags ---------------------------------------
# Chatterbox-Turbo performs these inline; every other engine (Kokoro) can't, so
# they get stripped. Kept deliberately small — the events Turbo renders cleanly.
EMOTION_TAGS = ("laugh", "chuckle", "sigh", "gasp", "cough", "sniffle", "groan", "yawn")
_TAG_RE = re.compile(r"\[(" + "|".join(EMOTION_TAGS) + r")\]", re.IGNORECASE)
# Off-vocab guesses too, including multi-word ones ([clears throat]). Letters and
# spaces only, so [MEM_ADD] directives are never touched.
_ANY_TAG_RE = re.compile(r"\[[a-zA-Z][a-zA-Z ]{1,18}\]")


def strip_emotion_tags(text: str) -> str:
    """Remove [laugh]/[sigh]/… tags and tidy the spacing they leave behind.
    Used for engines that can't speak tags, and for anything stored (transcript,
    memory) so a fact never contains '[laugh]'."""
    text = _ANY_TAG_RE.sub("", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)   # close the gap before punctuation
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def sanitize_emotion_tags(text: str) -> str:
    """Keep only the official tags; drop off-vocab guesses ([whisper], [giggle],
    [clears throat]…). Verified by audition: Chatterbox-Turbo performs only the
    documented set — anything else it reads out loud as a word."""
    text = _ANY_TAG_RE.sub(lambda m: m.group(0) if _TAG_RE.fullmatch(m.group(0)) else "", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def fix_leading_tags(text: str) -> str:
    """A tag at the very start of a synthesis chunk doesn't render (observed on
    Chatterbox-Turbo — a front-of-line [sigh] is silent, mid-line works). Relocate
    a leading tag to just after the first word so it performs. Degenerate chunks
    that are only a tag are dropped."""
    m = re.match(r"\s*(\[[a-zA-Z]{2,12}\])\s*(.*)", text, re.DOTALL)
    if not m:
        return text
    tag, rest = m.group(1), m.group(2).strip()
    if not rest:
        return ""                                   # nothing but a tag → nothing to speak
    parts = rest.split(" ", 1)
    first, tail = parts[0], (parts[1] if len(parts) > 1 else "")
    return f"{first} {tag} {tail}".strip() if tail else f"{first} {tag}"


EMOTION_PROMPT = (
    "EXPRESSIVE DELIVERY: your voice can perform inline cues. When one genuinely "
    "fits the moment, drop it into the reply — use the whole palette, not just "
    "laughs: [laugh] for something truly funny, [chuckle] for light amusement or "
    "warmth, [sigh] for sympathy, reluctance, or relief, [gasp] for surprise or "
    "big news, [groan] for playful exasperation, [yawn] for sleepy or late-night "
    "moments, [cough]/[sniffle] almost never. Rules: at most one per reply and "
    "most replies need none — an unearned cue feels fake; match the user's mood, "
    "never perform excitement at bad news. NEVER start a sentence with a tag — "
    "place it after the first word or between clauses (\"Oh [laugh] that's "
    "great\", not \"[laugh] that's great\"). Tags are spoken performances, not "
    "words; only the exact tags listed work."
)


# ---- Web search (optional, off by default) -----------------------------------
# The free, keyless path: the `ddgs` package (DuckDuckGo). Search returns snippet
# metadata only — titles + descriptions are enough for spoken one-liners, and
# nobody wants a webpage read aloud. Results live in the server's memory for a
# few minutes (never on disk) so follow-ups can dig into the same data.

def web_search(query: str, limit: int = 5) -> list[dict]:
    """One keyless DuckDuckGo search. Returns [{title, url, description}]; an
    empty list on any failure — a broken search should never break the turn."""
    try:
        from ddgs import DDGS
        out = []
        with DDGS(timeout=8) as client:
            for i, hit in enumerate(client.text(query, max_results=limit)):
                if i >= limit:
                    break
                out.append({"title": str(hit.get("title", ""))[:120],
                            "url": str(hit.get("href") or hit.get("url") or ""),
                            "description": str(hit.get("body", ""))[:300]})
        return out
    except Exception as e:
        print(f"[web] search failed ({e}): {query!r}")
        return []


# Weather gets real data instead of search snippets: Open-Meteo is keyless (no
# API key, no signup) and returns actual forecast numbers — the fix for stale
# SEO pages answering "what's the weather" with last week's temperatures.
_WEATHER_RE = re.compile(
    r"\b(weather|forecast|temperature|temps?|rain(ing)?|snow(ing)?|humidity|"
    r"wind(y)?|sunny|cloudy|precipitation|heat ?wave|cold front)\b", re.I)
_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
        55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow",
        75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
        82: "heavy showers", 85: "snow showers", 86: "snow showers",
        95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail"}


def is_weather_query(query: str) -> bool:
    return bool(_WEATHER_RE.search(query or ""))


def weather_place_from_query(query: str, fallback: str = "") -> str:
    """Pull the place out of a gate query like "tokyo weather july 4 2026" —
    strip weather words, date words, and filler; what's left is the location.
    Falls back to the profile location for bare "weather today" questions."""
    q = _WEATHER_RE.sub(" ", query or "")
    q = re.sub(r"\b(today|tonight|tomorrow|now|current(ly)?|right now|this|week|weekend|morning|afternoon|evening|hourly|daily|in|for|the|at|like|near|around|what'?s?|will|it|is|be|going|to|gonna|do(es)?)\b",
               " ", q, flags=re.I)
    q = re.sub(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
               " ", q, flags=re.I)
    q = re.sub(r"\d+", " ", q)
    q = re.sub(r"[^\w\s,]", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ,")
    return q or fallback


def weather_lookup(place: str) -> list[dict]:
    """Live forecast from Open-Meteo (keyless): geocode the place, fetch current
    conditions + 3 days, and return it shaped like a search result so it flows
    through the same cache, prompt block, and sources chip as everything else."""
    if not place:
        return []
    try:
        # the geocoder matches bare names — "Denver, Colorado" and "denver co"
        # find nothing, "Denver" does — so retry with the part before the comma,
        # then with the trailing word (usually a state) dropped
        words = place.split(",")[0].strip().split()
        candidates = [place, place.split(",")[0].strip(),
                      " ".join(words[:-1]) if len(words) > 1 else ""]
        hit = None
        for name in dict.fromkeys(c for c in candidates if c):
            g = _HTTP.get("https://geocoding-api.open-meteo.com/v1/search",
                          params={"name": name, "count": 1}, timeout=8).json()
            hit = (g.get("results") or [None])[0]
            if hit:
                break
        if not hit:
            print(f"[web] weather: couldn't geocode {place!r}")
            return []
        loc = ", ".join(str(x) for x in [hit.get("name"), hit.get("admin1")] if x)
        f = _HTTP.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": hit["latitude"], "longitude": hit["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "auto", "forecast_days": 3}, timeout=8).json()
        cur, day = f.get("current", {}), f.get("daily", {})
        wmo = lambda c: _WMO.get(int(c or 0), "mixed conditions")
        parts = [f"Right now: {round(cur.get('temperature_2m', 0))}°F "
                 f"(feels like {round(cur.get('apparent_temperature', 0))}°F), "
                 f"{wmo(cur.get('weather_code'))}, wind {round(cur.get('wind_speed_10m', 0))} mph."]
        for i, date in enumerate(day.get("time", [])[:3]):
            name = ("Today", "Tomorrow")[i] if i < 2 else \
                   datetime.strptime(date, "%Y-%m-%d").strftime("%A")
            precip = (day.get("precipitation_probability_max") or [0, 0, 0])[i] or 0
            parts.append(f"{name}: {wmo(day['weather_code'][i])}, "
                         f"high {round(day['temperature_2m_max'][i])}°F, "
                         f"low {round(day['temperature_2m_min'][i])}°F, "
                         f"{round(precip)}% chance of precipitation.")
        return [{"title": f"Live weather for {loc}", "url": "https://open-meteo.com",
                 "description": " ".join(parts)}]
    except Exception as e:
        print(f"[web] weather lookup failed ({e}): {place!r}")
        return []


_WEB_GATE_PROMPT = """You decide whether answering the user's LATEST message needs a live web search.
RULE 1 (overrides everything): if the user says "look up", "look it up", "check", \
"double check", "verify", "google", or "search" — you MUST return a query, no matter \
the topic, even if you know the answer. "Look up when the opera house opened" -> \
["sydney opera house opening year"].
Also search when the answer needs fresh or checkable outside facts: current events, \
sports scores, store hours, prices, weather, release dates, "is that true?" (verify \
the assistant's previous claim against the web), or anything time-sensitive.
Do NOT search for: chit-chat, opinions, memories about the user, or anything the \
conversation itself already answers — unless RULE 1 applies.

Reply with ONLY a JSON array. If a search is needed: one short search-engine query, \
e.g. ["home depot store hours today"]. If the user is questioning a previous claim, \
build the query from that claim. If no search is needed: [].
Today is {today}. ONLY when the question is about CURRENT conditions or happenings — \
weather, news, events nearby, scores, "what's going on" — put today's actual date in \
the query (e.g. "denver weather {today_short}") so stale pages don't win. NEVER add \
the date or year to historical, timeless, or how-to questions — it skews the results.

Recent conversation:
{context}

User's latest message: {text}"""


def web_gate(text: str, recent_turns: list[dict] | None, cfg: dict | None = None,
             place: str = "") -> str | None:
    """Quick pre-turn check: does this turn want a web search? Returns a search
    query or None. Runs at temperature 0 on the memory model; sees recent turns
    so "can you double check that?" resolves against the previous claim.
    `place` (the profile location) grounds local queries — weather, store
    hours — instead of the model inventing a [location] placeholder."""
    ctx = "\n".join(f"{t['role']}: {t['content'][:200]}" for t in (recent_turns or [])[-6:]) or "(start of conversation)"
    if place:
        ctx = f"(user's location: {place})\n" + ctx
    now = datetime.now().astimezone()
    today = now.strftime("%A, %B ") + f"{now.day}, {now.year}"
    today_short = now.strftime("%B ") + f"{now.day} {now.year}"
    try:
        raw = llm_complete([{"role": "user", "content":
                             _WEB_GATE_PROMPT.format(context=ctx, text=text,
                                                     today=today, today_short=today_short)}],
                           _mem_cfg(cfg))
    except Exception as e:
        print(f"[web] gate failed ({e})")
        return None
    m = _first_json_array(raw)
    if not m:
        return None
    try:
        arr = json.loads(m)
    except Exception:
        return None
    if arr and isinstance(arr[0], str) and arr[0].strip():
        return arr[0].strip()[:200]
    return None


# ---- ASR ---------------------------------------------------------------------
class _ASR:
    """Thin wrapper so call sites use `asr.transcribe(path).text` — mlx-audio's
    STT models expose `.generate()` (returns an AlignedResult with `.text`)."""

    def __init__(self, backend: str, model):
        self.backend = backend
        self._model = model

    def transcribe(self, path):
        return self._model.generate(str(path))


def load_asr(model: str = ASR_MODEL, backend: str = ASR_BACKEND) -> _ASR:
    """Load an ASR model via mlx-audio's STT stack and return an `_ASR` wrapper.
    `backend` is retained in the signature/settings for compatibility; only
    "mlx-audio" ships."""
    from mlx_audio.stt.utils import load_model as _load_stt
    return _ASR("mlx-audio", _load_stt(model))


# ---- LLM ---------------------------------------------------------------------
def _now_context() -> str:
    """Current local date + time for the model. Local-first app: the server clock
    is the user's clock, so no timezone round-trip is needed. Lets the assistant
    greet by time of day and reason about 'today' / 'this morning' naturally."""
    now = datetime.now().astimezone()
    # portable (no %-d / %-I — those crash strftime on Windows)
    hour12 = now.hour % 12 or 12
    stamp = now.strftime("%A, %B ") + f"{now.day}, {now.year} at {hour12}:{now.minute:02d} " + now.strftime("%p")
    tz = now.strftime("%Z") or "local time"
    return f"Current date and time: {stamp} ({tz})."


# Below this many known facts, the assistant is "still getting to know" the user
# and leans slightly curious; past it, the nudge drops and it just converses.
CURIOSITY_MAX_FACTS = 8
CURIOSITY_PROMPT = (
    "GETTING TO KNOW THEM: you still know only a little about this person. When "
    "there's a natural opening, show genuine interest with AT MOST ONE light "
    "follow-up question — ideally tied to something they just said or something "
    "you already know (\"you mentioned Molly, is she your only pet?\"). Never "
    "interrogate, never stack questions, never let it feel like a survey — "
    "especially out loud. One warm, curious touch, then let the conversation "
    "breathe. This fades on its own as you learn more about them."
)


def build_messages(history, memories, documents=None, profile=None,
                   persona: str | None = None, recent=None,
                   forgotten=None, emotion: bool = False,
                   web: str | None = None, web_fresh: bool = False,
                   rules=None, getting_to_know: bool = False,
                   proactive: str | None = None) -> list[dict]:
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
    if rules:
        # L4 guardrails: standing rules the user set. Injected every turn,
        # unconditionally, ABOVE memories — these are hard boundaries, not
        # suggestions, and they always apply even if nothing recalls them.
        joined = "\n".join(f"- {r}" for r in rules)
        system += ("\n\nRULES the user has set — always follow these, in every reply, "
                   "even when nothing above reminds you to. They override your own "
                   "defaults but never the user's explicit request in the moment:\n" + joined)
    system += ("\n\n" + _now_context() +
               " Use this for time-aware replies (greetings, time of day, \"today\"); "
               "don't state the date or time unless it's relevant.")
    if getting_to_know:
        system += "\n\n" + CURIOSITY_PROMPT
    if proactive:
        system += ("\n\nGENTLE HEADS-UP (only right now, at the very start of this conversation): "
                   "the user has " + proactive + " coming up. Mention it in ONE short, natural line — "
                   "just the heads-up, nothing more. Do NOT narrate a scene, guess details, or pile on "
                   "other memories around it. If they open with a specific unrelated request, answer "
                   "that first, then optionally add the one-line heads-up. Mention it at most once, and "
                   "never raise it again later in the conversation.")
    system += f"\n\nCURRENT MEMORIES:\n{mem_block}"
    if forgotten:
        joined = "\n".join(f"- {f}" for f in forgotten)
        system += ("\n\nDELETED THIS CONVERSATION (the user asked to forget these — they may "
                   "still appear in the chat history above, but never mention or allude to "
                   "them again unless the user brings them up first):\n" + joined)
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
    if web:
        system += (
            "\n\nWEB RESULTS (fetched moments ago; they expire in a few minutes):\n" + web +
            "\nAnswer from these results in ONE or TWO short spoken sentences — just the "
            "answer, never URLs or lists read aloud. If they don't contain the answer, say "
            "you couldn't find it rather than guessing. If the user asks for more detail, "
            "draw deeper on these same results."
        )
        if web_fresh:
            system += (
                " You just searched for this, so open with a very short cue that it came "
                "from the web — vary the wording: \"Here's what I found:\", \"Just "
                "checked —\", \"From a quick search,\"."
            )
    system += "\n\n" + MEMORY_DIRECTIVES
    system += "\n\n" + RECALL_GUARD
    system += "\n\n" + FOCUS_GUARD
    system += "\n\n" + GROUNDING_GUARD
    if emotion:
        system += "\n\n" + EMOTION_PROMPT
    # Last-resort: resolve any {{ASSISTANT_NAME}} the caller didn't (CLI, tests).
    system = system.replace("{{ASSISTANT_NAME}}", ASSISTANT_NAME_DEFAULT)
    return [{"role": "system", "content": system}, *history]


# Voice replies are spoken aloud, so an uncapped reply just rambles. When the user
# hasn't set an explicit cap, fall back to this instead of the model's own (often
# thousands of tokens). ~200 tokens is roughly 150 words — plenty for two to four
# spoken sentences, and it only ever truncates a genuine runaway.
DEFAULT_MAX_TOKENS = int(os.environ.get("VOICE_MAX_TOKENS", "200"))


def _openai_payload(cfg: dict, model: str, messages: list[dict], stream: bool) -> dict:
    """OpenAI-compatible request body with user-tunable generation controls."""
    p = {"model": model, "messages": messages, "stream": stream,
         "temperature": float(cfg.get("temperature", 0.6)),
         "top_p": float(cfg.get("top_p", 1.0))}
    p["max_tokens"] = int(cfg.get("max_tokens", 0) or 0) or DEFAULT_MAX_TOKENS
    return p


def _ollama_opts(cfg: dict) -> dict:
    """Ollama `options` block with user-tunable generation controls."""
    o = {"temperature": float(cfg.get("temperature", 0.6)),
         "top_p": float(cfg.get("top_p", 1.0))}
    nc = int(cfg.get("num_ctx", 0) or 0)
    if nc > 0:
        o["num_ctx"] = nc               # context window (Ollama-specific)
    o["num_predict"] = int(cfg.get("max_tokens", 0) or 0) or DEFAULT_MAX_TOKENS
    return o


def llm_complete(messages: list[dict], cfg: dict | None = None) -> str:
    """Call the configured LLM backend (Ollama or any OpenAI-compatible API).

    cfg keys: backend ('ollama'|'openai'), thinking (bool),
      ollama_url, ollama_model, openai_base, openai_model, api_key,
      temperature, top_p, max_tokens, num_ctx (generation controls).
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
        r = _HTTP.post(f"{base}/chat/completions", headers=headers, timeout=600,
                       json=_openai_payload(cfg, model, messages, False))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    else:
        url = (cfg.get("ollama_url") or OLLAMA_URL).rstrip("/")
        model = cfg.get("ollama_model") or LLM_MODEL
        r = _HTTP.post(
            f"{url}/api/chat", timeout=600,
            json={"model": model, "messages": messages, "stream": False,
                  "think": thinking, "options": _ollama_opts(cfg)},
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
              recent=None, forgotten=None) -> str:
    messages = build_messages(history, memories, documents, profile, persona, recent,
                              forgotten=forgotten)
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
        with _HTTP.stream("POST", f"{base}/chat/completions", headers=headers, timeout=600,
                          json=_openai_payload(cfg, model, messages, True)) as r:
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
                    print(f"[stream] skipped malformed line: {line[:80]!r}")
                    continue
                tok = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                if tok:
                    yield tok
    else:
        url = (cfg.get("ollama_url") or OLLAMA_URL).rstrip("/")
        model = cfg.get("ollama_model") or LLM_MODEL
        with _HTTP.stream("POST", f"{url}/api/chat", timeout=600,
                          json={"model": model, "messages": messages, "stream": True,
                                "think": thinking, "options": _ollama_opts(cfg)}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    print(f"[stream] skipped malformed line: {line[:80]!r}")
                    continue
                tok = obj.get("message", {}).get("content")
                if tok:
                    yield tok
                if obj.get("done"):
                    break


# ---- memory directive parsing ------------------------------------------------
def apply_directives(strata, reply: str, memories: list[dict],
                     forgotten: list[str] | None = None) -> str:
    """Strip [MEM_ADD]/[MEM_DEL] lines from the reply and apply them to Strata.

    Returns the clean, speakable text. When ``forgotten`` is given, deleted
    keywords are appended to it so the caller can keep the model from
    referencing them for the rest of the session (they'd otherwise linger in
    the chat history and keep coming up after the user asked to forget them).
    """
    # Directives can appear mid-line ("Got it. [MEM_ADD] fact") — models don't
    # reliably put them on their own line. Split every line on the tags so the
    # spoken text never contains a directive and every directive is applied.
    tag_re = re.compile(r"\[MEM_(ADD|DEL)\]")
    spoken: list[str] = []
    for line in reply.splitlines():
        parts = tag_re.split(line)
        head = parts[0].strip()
        if head:
            spoken.append(head)
        # parts alternates after the head: kind, payload, kind, payload, …
        for kind, payload in zip(parts[1::2], parts[2::2]):
            payload = payload.strip()
            if not payload:
                continue
            if kind == "ADD":
                _add_or_supersede(strata, payload, memories)
            else:
                _forget(strata, payload, memories)
                if forgotten is not None:
                    forgotten.append(payload)
    return "\n".join(spoken).strip()


_DATEWORDS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
              "sunday", "january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december",
              "today", "tomorrow", "yesterday", "tonight"}


def _anchors(fact: str) -> set:
    """The distinguishing specifics of a fact — proper nouns, numbers, days,
    months. Two facts that differ on these are different facts, not restatements
    of one ("interview with Acme" vs "interview with Globex")."""
    out = set()
    for i, tok in enumerate(fact.split()):
        w = re.sub(r"[^\w:/-]", "", tok)
        if not w:
            continue
        lw = w.lower()
        if lw in _DATEWORDS or any(c.isdigit() for c in w):
            out.add(lw)
        elif i > 0 and w[0].isupper():            # proper noun (skip the sentence start)
            out.add(lw)
    return out


def _same_fact(a: str, b: str) -> bool:
    """Is `a` a restatement/refinement of `b` (merge) rather than a distinct fact
    (keep both)? Requires strong word overlap AND compatible anchors: if each has
    a distinguishing anchor the other lacks, they're different facts."""
    sa = {w for w in a.lower().split() if len(w) > 3}
    sb = {w for w in b.lower().split() if len(w) > 3}
    union = sa | sb
    jaccard = len(sa & sb) / len(union) if union else 0.0
    # strong overlap, OR one fact's words contained in the other (an elaboration)
    if jaccard < 0.5 and not (sa and (sa <= sb or sb <= sa)):
        return False
    aa, ab = _anchors(a), _anchors(b)
    if aa and ab and not (aa <= ab or ab <= aa):   # conflicting specifics → distinct
        return False
    return True


def _ground_facts(facts: list[str], source: str) -> list[str]:
    """Drop extracted facts whose distinguishing anchors (dates, numbers, proper
    nouns) aren't in the text the model actually saw. Catches invented specifics
    — the per-turn extractor hallucinating "interview on Tuesday" from a turn
    about nursing — without touching legitimately reworded facts, which carry no
    invented anchor. Harvest already grounds via quotes; this is the same
    discipline for the per-turn pass."""
    src = source.lower()
    kept = []
    for f in facts:
        missing = [a for a in _anchors(f) if a not in src]
        if missing:
            print(f"  · extract dropped (invented {missing}): {f!r}")
            continue
        kept.append(f)
    return kept


def _add_or_supersede(strata, fact: str, memories: list[dict]):
    """Write or update a fact. Returns the canonical record id of the written/
    updated fact, or None on an exact duplicate (nothing changed). Supersede only
    when the new fact is genuinely a restatement of an existing one — anchor-aware,
    so two distinct-but-similar facts (two interviews, two people) never clobber
    each other."""
    fl = fact.lower()
    for m in memories:
        if m["text"].lower() == fl:
            return None  # exact dup
        if _same_fact(fact, m["text"]):
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
    # are the raw episodic spine for the timeline, not distilled memories — and
    # L4 GUARDRAIL rules, which have their own always-on path (list_rules).
    from strata.canonical.records import Status, RecordType
    recs = strata.engine.store.query(record_type=RecordType.FACT, exclude_tombstoned=True)
    active = [r for r in recs if r.status in (Status.ACTIVE, Status.REINFORCED)]
    active.sort(key=lambda r: r.created_at)
    return [{"id": r.id, "text": r.content, "t": r.created_at} for r in active]


# ---- Rules (L4 guardrails) ---------------------------------------------------
# User-set standing rules. Stored as L4 GUARDRAIL records so they're structurally
# separate from facts: list_memories() (which drives the Memories page, smoothing,
# recall fallback, AND the "forget X" path) filters to FACT, so rules can never be
# smoothed, decayed, recall-gated, or forgotten by accident. They are injected into
# EVERY turn's prompt unconditionally — that unconditional-ness is the whole point.

def search_memories(strata, query: str, embedder=None, limit: int = 40) -> list[dict]:
    """Search the user's own memories. Word matches (substring) are precise, so
    when there are any, return just those. Only when nothing matches literally do
    we fall back to semantic recall — so a conceptual query ("job") still finds
    "Works as a nurse". Empty query returns everything. Returns memory dicts."""
    mems = list_memories(strata)
    q = (query or "").strip().lower()
    if not q:
        return mems
    hits = [m for m in mems if q in m["text"].lower()]
    if hits:
        return hits[:limit]
    if embedder is not None:                    # no literal match → conceptual fallback
        tmap = {m["text"]: m for m in mems}
        sem = [tmap[t] for t in recall_memories(strata, query, top_k=8) if t in tmap]
        return sem[:8]
    return []


def list_rules(strata) -> list[dict]:
    """Active standing rules, oldest first."""
    from strata.canonical.records import Status, RecordType
    recs = strata.engine.store.query(record_type=RecordType.GUARDRAIL, exclude_tombstoned=True)
    active = [r for r in recs if r.status in (Status.ACTIVE, Status.REINFORCED)]
    active.sort(key=lambda r: r.created_at)
    return [{"id": r.id, "text": r.content, "t": r.created_at} for r in active]


def add_rule(strata, text: str) -> dict | None:
    """Store one standing rule (L4). Deduplicates case-insensitively."""
    text = _clean(text)[:200]
    if len(text) < 2:
        return None
    for r in list_rules(strata):
        if r["text"].lower() == text.lower():
            return r
    try:
        res = strata.write_memory(text, tier="L4", record_type="guardrail")
    except Exception as e:
        print(f"[rules] add failed: {e}")
        return None
    return {"id": res.get("id"), "text": text} if isinstance(res, dict) else None


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
        r = _HTTP.post(f"{self.url}/api/embed", timeout=60,
                       json={"model": self.model, "input": text or " "})
        r.raise_for_status()
        data = r.json()
        # /api/embed returns {"embeddings":[[...]]}; tolerate the older shape too.
        if "embeddings" in data:
            return data["embeddings"][0]
        return data["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in ONE round trip — /api/embed accepts a list. One
        HTTP call instead of N; used wherever several texts need vectors at once
        (recap scoring, exemplar warmup)."""
        if not texts:
            return []
        r = _HTTP.post(f"{self.url}/api/embed", timeout=120,
                       json={"model": self.model, "input": [t or " " for t in texts]})
        r.raise_for_status()
        return r.json()["embeddings"]


def make_embedder(url: str = OLLAMA_URL):
    """Return an OllamaEmbedder if the embed model is available, else None so the
    caller falls back to the offline default (and the dump-all recall path).
    `url` lets the server pass the user's configured Ollama endpoint — which may
    be another machine on the network — instead of assuming localhost."""
    try:
        emb = OllamaEmbedder(url=url)
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


# Time-aware recall: embeddings ignore time, so "what did I do yesterday?" can't
# be answered by topical similarity. When a query names a relative time window,
# boost memories from that window to the front (a reorder, never a hard filter —
# the two-pass lesson). Env-gated for A/B against the benchmark.
TEMPORAL = os.environ.get("VOICE_TEMPORAL", "1") != "0"


# Proactive surfacing: at the START of a conversation, gently bring up a genuine
# upcoming commitment ("your interview with Globex is Thursday — want to prep?").
# The relative-date problem (a fact says "Thursday", but is it still ahead?) is
# solved by handing the model today's date + when each fact was mentioned and
# letting it resolve. Conservative by design — most turns surface nothing.
_COMMIT_RE = re.compile(
    r"\b(interview|appointment|meeting|deadline|due|wedding|flight|trip|exam|test|"
    r"reservation|party|concert|game|visit|conference|class|checkup|surgery|dentist|"
    r"doctor|presentation|graduation|birthday|anniversary|move|moving|closing|hearing)\b", re.I)

_UPCOMING_PROMPT = """Today is {today}. Below are things the user mentioned, each with WHEN they said it. \
Find the SINGLE most relevant SPECIFIC, DATED commitment that falls TODAY or within the next 3 days. \
Resolve relative dates from when it was said: "interview Thursday" said last Monday, with today being \
Wednesday, means this Thursday (tomorrow).
Be strict. Only answer if it is a concrete commitment (an appointment, interview, flight, deadline, \
event) with a clear day that is today or in the next 3 days. If so, reply with a SHORT description \
resolved to a concrete day — e.g. "a job interview with Globex this Thursday" or "a dentist appointment \
tomorrow". Nothing else.
Otherwise — anything more than 3 days out, already passed, a stable fact, a preference, a place, a \
person, a hobby, or at all vague — reply with exactly: NONE. When in doubt, reply NONE.

Mentioned:
{items}

Upcoming (or NONE):"""


def upcoming_nudge(memories: list[dict], cfg: dict | None = None) -> str | None:
    """A short resolved description of one genuine upcoming commitment to bring up
    at the start of a conversation, or None. Considers only recently-mentioned,
    commitment-type memories (so stable facts and old events never trigger it)."""
    now = datetime.now()
    cutoff = int(now.timestamp() * 1000) - 21 * 86_400_000   # only recent mentions
    cands = [m for m in memories
             if (m.get("t") or 0) >= cutoff and _COMMIT_RE.search(m.get("text", ""))]
    if not cands:
        return None
    today = now.strftime("%A, %B ") + f"{now.day}, {now.year}"
    items = "\n".join(
        f"- (mentioned {datetime.fromtimestamp((m['t'] or 0) / 1000).strftime('%A, %B ')}"
        f"{datetime.fromtimestamp((m['t'] or 0) / 1000).day}) {m['text']}"
        for m in cands)
    try:
        raw = llm_complete([{"role": "user",
                             "content": _UPCOMING_PROMPT.format(today=today, items=items)}],
                           _mem_cfg(cfg))
    except Exception as e:
        print(f"[proactive] check failed: {e}")
        return None
    line = (raw or "").strip().splitlines()[0].strip() if raw else ""
    if not line or line.upper().startswith("NONE") or len(line) < 6:
        return None
    return line[:160]


def _reltime(ts_ms: int, now_ms: int) -> str:
    """A human relative-time tag for a memory's timestamp — what the model reads
    to know which memory the asked timeframe means."""
    d = (now_ms - ts_ms) / 86_400_000
    if d < 1:   return "today"
    if d < 2:   return "yesterday"
    if d < 8:   return "in the past week"
    if d < 15:  return f"about {int(round(d))} days ago"
    if d < 45:  return "a few weeks ago"
    if d < 400: return f"about {max(1, int(round(d / 30)))} month{'s' if d >= 45 else ''} ago"
    return "a while ago"


def _temporal_window(query: str):
    """Parse a relative time expression into a (start_ms, end_ms) window, or None.
    Rolling windows anchored to now — fuzzy on purpose, since they only reorder
    recall, never drop anything."""
    q = (query or "").lower()
    now_ms = int(datetime.now().timestamp() * 1000)
    D = 86_400_000

    def w(newer_days, older_days):
        return (now_ms - older_days * D, now_ms - newer_days * D)

    if "yesterday" in q:
        return w(1, 2)
    if "this morning" in q or "today" in q or "tonight" in q:
        return w(0, 1)
    if "last week" in q:
        return w(7, 14)
    if "last month" in q:
        return w(31, 62)
    if any(p in q for p in ("this week", "this weekend", "past week", "past few days")):
        return w(0, 7)
    if "this month" in q:
        return w(0, 31)
    if any(p in q for p in ("recently", "lately", "these days", "the other day")):
        return w(0, 14)
    return None


# ---- event recency ------------------------------------------------------------
# A remembered event ("interview next Thursday", "outing for the Fourth of July")
# is FUTURE when you mention it and PAST once its day arrives. The store only knows
# WHEN you said it, not when the event is — so without this the model keeps calling
# a passed event "upcoming". We resolve the event's actual date from the fact text
# (anchored on when it was said, since "Thursday"/"next week" are relative to that),
# then tag it upcoming/past at recall time and push already-past events down. Fully
# deterministic (no LLM) so recall stays fast and the behaviour is benchmarkable.
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
_MON_ABBR = {m[:3]: i for m, i in _MONTHS.items()}
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
# fixed-date holidays (substring -> month, day). Longer keys first when matching.
_HOLIDAYS = {
    "new year's eve": (12, 31), "new years eve": (12, 31),
    "christmas eve": (12, 24), "christmas": (12, 25),
    "fourth of july": (7, 4), "4th of july": (7, 4), "july 4th": (7, 4),
    "independence day": (7, 4), "juneteenth": (6, 19), "halloween": (10, 31),
    "valentine": (2, 14), "new year": (1, 1),
}
_DAY_MS = 86_400_000


def _event_date(text: str, mention_ms: int):
    """Best-effort absolute date (ms) that a fact refers to, or None if it names no
    date. Relative words are anchored on ``mention_ms`` (when it was said); an
    absolute month/day picks the occurrence on or after that anchor, since people
    mention events shortly before they happen."""
    t = " " + (text or "").lower() + " "
    anchor = datetime.fromtimestamp(mention_ms / 1000).replace(hour=0, minute=0, second=0, microsecond=0)

    def mk(y, mo, d):
        try:
            return datetime(y, mo, d)
        except ValueError:
            return None

    def on_or_after(mo, d):
        # nearest occurrence of month/day that isn't well before the anchor
        cand = mk(anchor.year, mo, d)
        if cand is None:
            return None
        if cand < anchor - timedelta(days=1):
            return mk(anchor.year + 1, mo, d) or cand
        return cand

    dt = None
    # ISO date  (2026-07-04)
    m = re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        dt = mk(int(m[1]), int(m[2]), int(m[3]))
    # month name + day  ("July 4", "4th of July", "December 3, 2026")
    if dt is None:
        m = re.search(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b", t)
        if m and m[1] in _MONTHS or (m and m[1] in _MON_ABBR):
            mo = _MONTHS.get(m[1]) or _MON_ABBR.get(m[1])
            dt = mk(int(m[3]), mo, int(m[2])) if m[3] else on_or_after(mo, int(m[2]))
        if dt is None:
            m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([a-z]+)\b", t)
            if m and (m[2] in _MONTHS or m[2] in _MON_ABBR):
                mo = _MONTHS.get(m[2]) or _MON_ABBR.get(m[2])
                dt = on_or_after(mo, int(m[1]))
    # holidays
    if dt is None:
        for name, (mo, d) in _HOLIDAYS.items():
            if name in t:
                dt = on_or_after(mo, d)
                break
    # relative phrases
    if dt is None:
        if "day after tomorrow" in t:
            dt = anchor + timedelta(days=2)
        elif "tomorrow" in t:
            dt = anchor + timedelta(days=1)
        elif re.search(r"\bin (\d{1,2}) days?\b", t):
            dt = anchor + timedelta(days=int(re.search(r"\bin (\d{1,2}) days?\b", t)[1]))
        elif re.search(r"\bin (\d{1,2}) weeks?\b", t):
            dt = anchor + timedelta(weeks=int(re.search(r"\bin (\d{1,2}) weeks?\b", t)[1]))
        elif "next weekend" in t:
            dt = anchor + timedelta(days=(5 - anchor.weekday()) % 7 + 7)
        elif "this weekend" in t or "the weekend" in t:
            dt = anchor + timedelta(days=(5 - anchor.weekday()) % 7)
        elif "next week" in t:
            dt = anchor + timedelta(days=7)
        elif "next month" in t:
            dt = anchor + timedelta(days=30)
    # weekday name  ("Thursday", "next Friday") -> next such day at/after anchor
    if dt is None:
        for wd, idx in _WEEKDAYS.items():
            if wd in t:
                # nearest upcoming occurrence at/after the anchor. "next Thursday" is
                # genuinely ambiguous in speech; treat it as the imminent one.
                dt = anchor + timedelta(days=(idx - anchor.weekday()) % 7)
                break
    return int(dt.timestamp() * 1000) if dt else None


def _recency_tag(event_ms: int, now_ms: int) -> str:
    """Human tag telling the model whether a dated event is upcoming or already past.
    Counts whole CALENDAR days (events are date-level, so a noon 'now' vs a midnight
    event date must not round a next-day event down to zero)."""
    ev = datetime.fromtimestamp(event_ms / 1000).date()
    d = (ev - datetime.fromtimestamp(now_ms / 1000).date()).days
    if d == 0:
        return "happening today"
    if d > 0:
        n = d
        if n == 1:
            return "coming up tomorrow"
        if n <= 10:
            return f"coming up in {n} days"
        if n <= 45:
            return "coming up in a few weeks"
        if n <= 75:
            return "coming up in a couple of months"
        return "coming up later on"
    n = -d
    if n == 1:
        return "was yesterday, now in the past"
    if n <= 10:
        return f"was {n} days ago, now in the past"
    if n <= 45:
        return "was a few weeks ago, now in the past"
    if n <= 400:
        months = max(1, round(n / 30))
        return f"was about {months} month{'s' if months > 1 else ''} ago, now in the past"
    return "was a while ago, now in the past"


def _with_recency(m: dict, now_ms: int) -> str:
    """Tag a memory: an event date if the fact names one, else the mention time."""
    ev = _event_date(m["text"], m.get("t") or now_ms) if TEMPORAL else None
    if ev is not None:
        return f"{m['text']} ({_recency_tag(ev, now_ms)})"
    return f"{m['text']} ({_reltime(m.get('t') or now_ms, now_ms)})"


def _order_by_recency(mem_dicts: list[dict], now_ms: int) -> list[str]:
    """Tag dated events upcoming/past and push already-passed ones to the end — a
    passed event is less relevant than a live one. Undated memories keep their
    order and get no tag (so plain-fact recall is untouched)."""
    if not TEMPORAL:
        return [m["text"] for m in mem_dicts]
    fresh, past = [], []
    for m in mem_dicts:
        ev = _event_date(m["text"], m.get("t") or now_ms)
        if ev is None:
            fresh.append(m["text"])
        elif ev < now_ms - _DAY_MS // 2:          # >12h in the past
            past.append(f"{m['text']} ({_recency_tag(ev, now_ms)})")
        else:
            fresh.append(f"{m['text']} ({_recency_tag(ev, now_ms)})")
    return fresh + past


def select_memories(strata, query: str, *, semantic: bool = True,
                    threshold: int = RECALL_THRESHOLD, mems=None) -> list[str]:
    """Pick the memory texts to inject this turn. Small store -> inject everything
    (perfect, ~free). Large store -> semantic recall of the most relevant few. When
    the query names a time window, memories from that window are boosted to the
    front. Pass ``mems`` (a fresh list_memories snapshot) to avoid a duplicate query.

    Note: we deliberately do NOT scope the injected set to a single event
    (an earlier two-pass experiment did, and the benchmark showed it dropped
    recall by locking the wrong event). Keeping every candidate in context and
    letting FOCUS_GUARD stop the model from volunteering neighbours is the
    measured win — collisions to 0 with recall held at 100%."""
    if mems is None:
        mems = list_memories(strata)
    small = not semantic or len(mems) <= threshold
    if small:
        base = list(mems)
    else:
        order = recall_memories(strata, query)          # semantic, ordered by relevance
        tmap = {m["text"]: m for m in mems}
        base = [tmap[t] for t in order if t in tmap]

    now_ms = int(datetime.now().timestamp() * 1000)
    win = _temporal_window(query) if TEMPORAL else None
    if win:
        lo, hi = win
        in_win = sorted((m for m in mems if lo <= (m.get("t") or 0) <= hi),
                        key=lambda m: m.get("t") or 0, reverse=True)   # most recent first
        seen, ordered = set(), []
        # On a time-scoped query, tag EVERY injected memory with its time: a dated
        # event gets its real upcoming/past status, everything else its mention time.
        # This lets the model tell a similar memory happened at a DIFFERENT time.
        for m in in_win + base:
            if m["text"] not in seen:
                seen.add(m["text"])
                ordered.append(_with_recency(m, now_ms))
        return ordered[:max(RECALL_TOP_K, len(in_win))]

    out = base if small else base[:RECALL_TOP_K]
    return _order_by_recency(out, now_ms)


# ---- memory capture ------------------------------------------------------------
# Forgetting stays deterministic and immediate — deletion must be reliable.
# WRITING memories is never verbatim anymore: an explicit "remember …" becomes a
# candidate that the smoothing layer (polish_fact, background) judges and rewrites
# into a clean third-person fact; everything implicit is left to extract_facts_llm.
# The old regex writers (likes/lives/pets…) stored raw voice-transcript spans,
# which produced garbled memories ("Likes presented in that video enter that…")
# and captured questions as facts ("do you remember how to X" -> "How to X").
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,!?;:'\"").strip()


def _first_json_array(raw: str) -> str | None:
    """Return the first balanced JSON array in `raw`, or None. The old greedy
    re.search(r"\\[.*\\]", DOTALL) spanned from the FIRST '[' to the LAST ']' —
    if a chatty model emitted two arrays (or brackets in prose), the span was
    invalid JSON and the result was silently dropped."""
    start = raw.find("[")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(raw)):
            c = raw[i]
            if in_str:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': in_str = False
            elif c == '"': in_str = True
            elif c == "[": depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    seg = raw[start:i + 1]
                    try:
                        json.loads(seg)
                        return seg
                    except Exception:
                        break   # not valid JSON — try the next '['
        start = raw.find("[", start + 1)
    return None


def _mem_cfg(cfg: dict | None) -> dict:
    """Config for memory-writing LLM calls: temperature 0 and no thinking.
    Distillation should be deterministic — the chat temperature is tuned for
    personality, which is exactly what causes detail drift in stored facts.

    These calls emit JSON (fact lists, polish rewrites), not spoken replies, so
    they must NOT inherit the short spoken-reply cap — give them ample room."""
    base = cfg or {}
    return {**base, "temperature": 0.0, "thinking": False,
            "max_tokens": max(int(base.get("max_tokens", 0) or 0), 2048)}


def capture_memory(strata, user_text: str, event_id: int | None = None) -> list[str]:
    """Apply explicit forget-requests from a user utterance immediately.
    Returns a list of change descriptions for logging. (Fact WRITES happen in
    the background smoothing layer — see remember_candidate / polish_fact.)"""
    text = (user_text or "").strip()
    if not text:
        return []
    # "do you remember …" is a recall question, not a forget/remember command
    m = re.search(r"\b(?:forget|delete)\b(?:\s+(?:about|the\s+memory\s+about|that|my))?\s+(.+)", text, re.I)
    if m:
        kw = _clean(m.group(1))
        if kw:
            _forget(strata, kw, list_memories(strata))
            return [f"- forget: {kw}"]
    return []


def remember_candidate(user_text: str) -> str | None:
    """Detect an explicit 'remember that …' command and return the clause to be
    polished + stored by the background worker. Returns None for recall
    questions ('do you remember …?') and normal speech."""
    text = (user_text or "").strip()
    if not text:
        return None
    # questions about memory are recall, not storage commands
    if re.search(r"\b(?:do|did|can|could|will|would)\s+you\s+remember\b", text, re.I):
        return None
    m = re.search(r"\b(?:remember|make a note|note that|keep in mind|don'?t forget)\b(?:\s+that)?\s+(.+)", text, re.I)
    if not m:
        return None
    clause = _clean(m.group(1))
    return clause if len(clause) >= 2 else None


_POLISH_PROMPT = """The user explicitly asked their assistant to remember something. The request below \
comes from a voice transcript and may contain mis-transcriptions or filler words.
Rewrite it as ONE short, clean memory in the third person (e.g. "Has a job interview on Tuesday", \
"Prefers replies to be brief"). Fix obvious transcription garble; keep every real detail; add nothing. Copy anchor details \
(names, dates, days, times, numbers, places, titles) EXACTLY as said — paraphrase only the words \
around them. The request is the USER speaking in the first person — keep who-did-what correct.
Example: "remember that I gave them my report on Friday umm about the budget stuff" -> \
"Gave them a report about the budget on Friday".
If it is actually a question or a request to recall/delete (not something to store), reply with exactly SKIP.
Reply with ONLY the memory text (or SKIP) — no quotes, no explanation.

Request: "{text}" """


def polish_fact(candidate: str, cfg: dict | None = None) -> str | None:
    """Smoothing layer for explicit remember-commands: judge + rewrite into a
    clean third-person fact. Falls back to the cleaned verbatim clause if the
    LLM is unreachable (an explicit command must never be silently dropped);
    returns None when the model says it isn't a storable fact."""
    candidate = _clean(candidate or "")
    if not candidate:
        return None
    fallback = candidate.upper() if len(candidate) == 1 else candidate[0].upper() + candidate[1:]
    try:
        reply = llm_complete([{"role": "user", "content": _POLISH_PROMPT.format(text=candidate)}], _mem_cfg(cfg))
        reply = _THINK_TAG.sub("", reply).strip().strip('"').strip()
    except Exception as e:
        print(f"[memory] polish failed ({e}); storing verbatim")
        return fallback
    if not reply:
        return fallback
    if reply.upper().startswith("SKIP"):
        return None
    # guard against a chatty model: a memory is one short line
    line = reply.splitlines()[0].strip()
    return line if 2 <= len(line) <= 200 else fallback


# ---- LLM extraction pass (covers natural speech the patterns miss) -----------
_EXTRACT_PROMPT = """You pull durable facts about the user out of their latest message, for long-term memory.
The message is a voice transcript and may contain mis-transcriptions, filler, and run-ons — NEVER copy \
garbled wording into a fact. Write each fact cleanly in your own words; if a passage is too garbled to \
be sure what was meant, skip it rather than guess.
Copy anchor details — names, dates, days of the week, times, numbers, places, and titles — EXACTLY as the user said them; paraphrase only the wording around them, never the anchors.
Facts are read aloud by a voice, so spell everything out ("Arvada, Colorado" — never "Arvada, Co.").
Never begin a fact with "The user" — start with the verb or noun ("Has a dog named Molly", "Getting into bouldering").
For an event, record the DAY but not its tense: write "Has a dentist appointment on Thursday", never "upcoming appointment", "will go", or "went" — whether it is still ahead or already past is worked out later from the date, so words like "upcoming" go stale and must be left out.
A one-time outing or errand ("went to the park", "was at Cherry Creek today") is NOT durable — keep an activity \
only when it shows an ongoing interest or habit ("Getting into bouldering").
Return ONLY a JSON array of short factual strings written in the third person, e.g.
["Has a dog named Rex", "Works as a nurse", "Has a job interview on Tuesday", "Allergic to shellfish"].
Include stable things: preferences, relationships, family, pets, job, location, hobbies, ongoing projects, health, upcoming commitments (interviews, appointments), and notable life facts.
Exclude: questions (including questions about the assistant or its memory), comments about the assistant, fleeting events with no lasting significance, pure small talk, and anything already in existing memory.
If nothing durable is stated, return [].

Existing memory (do not duplicate):
{existing}
{context}
User message:
"{text}"

JSON array:"""


def extract_facts_llm(user_text: str, existing: list[str], cfg: dict | None = None,
                      context: str | None = None) -> list[str]:
    """Ask the configured model to extract durable facts as a JSON array.
    Runs after the spoken reply, so it never adds latency to speech.
    ``context`` is the last few conversation turns — it lets a fragment like
    "it's next Tuesday" resolve to the thing being discussed, instead of being
    skipped as meaningless on its own."""
    text = (user_text or "").strip()
    if not text:
        return []
    ctx = ""
    if context:
        ctx = ("\nRecent conversation (context ONLY — resolve references with it, "
               "but extract facts ONLY from the latest user message below):\n"
               + context + "\n")
    prompt = _EXTRACT_PROMPT.format(
        existing="\n".join(f"- {m}" for m in existing) or "(none)", text=text,
        context=ctx)
    try:
        raw = llm_complete([{"role": "user", "content": prompt}],
                           _mem_cfg(cfg))
    except Exception as e:
        print(f"[memory] extraction call failed: {e}")
        return []
    m = _first_json_array(raw)
    if not m:
        return []
    try:
        arr = json.loads(m)
    except Exception:
        return []
    out = []
    for x in arr:
        if isinstance(x, str) and len(_clean(x)) > 2:
            out.append(_clean(x)[:160])   # a memory is one short fact, never a paragraph
    # grounding: a fact's specific anchors must come from what the model saw
    return _ground_facts(out, text + " " + (context or ""))[:5]


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


_HARVEST_PROMPT = """You review a finished voice conversation and pull out the durable facts about the USER \
that should be kept in long-term memory. You see the whole transcript, so COMBINE details that were \
spread across turns into complete facts (e.g. an interview mentioned in one turn, its day in another, \
and the role in a third become one fact: "Has an interview next Tuesday with the VP for a builder role \
building internal tools").
The transcript is from speech recognition and may contain mis-transcriptions and clipped sentence \
starts — never copy garbled wording; write each fact cleanly, and skip what you can't confidently parse.
Copy anchor details — names, dates, days of the week, times, numbers, places, and titles — EXACTLY as the user said them; paraphrase only the wording around them, never the anchors.
Facts are read aloud by a voice, so spell everything out ("Arvada, Colorado" — never "Arvada, Co.").
Never begin a fact with "The user" — start with the verb or noun ("Has a dog named Molly", "Getting into bouldering").
For an event, record the DAY but not its tense: write "Has a dentist appointment on Thursday", never "upcoming appointment", "will go", or "went" — whether it is still ahead or already past is worked out later from the date, so words like "upcoming" go stale and must be left out.
A one-time outing or errand ("went to the park", "was at Cherry Creek today") is NOT durable — keep an activity \
only when it shows an ongoing interest or habit ("Getting into bouldering").
ONLY capture facts the USER stated about themselves or explicitly confirmed. This is critical:
- IGNORE anything the ASSISTANT said, suggested, guessed, or asked — the assistant's words are never facts about the user. (If the assistant says "you should grab a coffee for the drive" that is NOT "drinks coffee".)
- IGNORE anything the user negated, declined, corrected, doubted, or dismissed. Watch for negation: "that wouldn't be good", "no", "not really", "actually no", "I don't" all mean DON'T store it — and never store the opposite of what the user rejected.
- A fact only counts if you can point to the user's own words for it.
For EACH fact, return the short verbatim USER quote it comes from, so it can be checked.
Return ONLY a JSON array of objects: [{{"fact": "<clean third-person fact>", "quote": "<the user's own words it came from>"}}].
Include: preferences, relationships, family, pets, job, location, hobbies, ongoing projects, \
upcoming commitments (interviews, appointments, deadlines), notable experience and accomplishments.
Exclude: questions, tests of the assistant, requests to remember/forget, anything about the assistant \
itself, fleeting small talk, and anything already covered by existing memory.
If nothing new and durable was said, return [].

Existing memory (do not duplicate or restate):
{existing}

Transcript:
{transcript}

JSON array:"""


def harvest_session_facts(turns: list[dict], existing: list[str],
                          cfg: dict | None = None) -> list[dict]:
    """End-of-conversation fact harvest: one extraction pass over the WHOLE
    transcript, so facts scattered across turns get assembled into complete
    memories that per-turn extraction can't see. Returns [{fact, quote}] — the
    quote is the user's own words the fact is grounded in (the prompt only
    captures user-stated/confirmed facts, so the assistant's suggestions and
    anything the user negated never become facts). Pure function; the caller
    stores via add_harvested_facts, which uses the quote to source-link."""
    if not turns:
        return []
    prompt = _HARVEST_PROMPT.format(
        existing="\n".join(f"- {m}" for m in existing) or "(none)",
        transcript=_transcript_text(turns))
    try:
        raw = llm_complete([{"role": "user", "content": prompt}],
                           _mem_cfg(cfg))
    except Exception as e:
        print(f"[memory] harvest call failed: {e}")
        return []
    m = _first_json_array(raw)
    if not m:
        return []
    try:
        arr = json.loads(m)
    except Exception:
        return []
    out = []
    for x in arr:
        if isinstance(x, dict) and x.get("fact"):
            out.append({"fact": _clean(str(x["fact"]))[:160],
                        "quote": _clean(str(x.get("quote", "")))})
        elif isinstance(x, str):                      # tolerate the old flat shape
            out.append({"fact": _clean(x)[:160], "quote": ""})
    return [o for o in out if len(o["fact"]) > 2][:8]


def _match_event(events: list[dict], quote: str) -> int | None:
    """The L0 user-turn event a harvested quote came from: exact substring first,
    then word-overlap ≥0.6. None if nothing matches (fact still stores, unlinked)."""
    if not quote:
        return None
    ql = quote.lower()
    for e in events:
        if ql in e["text"].lower():
            return e["id"]
    qw = {w for w in re.findall(r"[a-z0-9]+", ql) if len(w) > 2}
    if not qw:
        return None
    best, best_score = None, 0.0
    for e in events:
        ew = {w for w in re.findall(r"[a-z0-9]+", e["text"].lower()) if len(w) > 2}
        if ew:
            score = len(qw & ew) / len(qw)
            if score > best_score:
                best, best_score = e["id"], score
    return best if best_score >= 0.6 else None


def add_harvested_facts(strata, harvested: list[dict]) -> list[str]:
    """Store harvested {fact, quote} items — but ONLY when the quote grounds to a
    real user turn. This is the guard against exactly what went wrong: a fact the
    user never said (the assistant's, a negation flipped positive, or the model
    parroting a prompt example) has no matching user utterance, so it's dropped
    rather than stored untraceable. Survivors are source-linked to that turn."""
    if not harvested:
        return []
    events = list_events(strata)
    memories = list_memories(strata)
    added = []
    for item in harvested:
        fact = _clean(item.get("fact", ""))[:160]
        if len(fact) < 3:
            continue
        ev_id = _match_event(events, item.get("quote", ""))
        if ev_id is None:
            print(f"[memory] harvest dropped (no user source): {fact!r}")
            continue
        before = {m["text"] for m in memories}
        rid = _add_or_supersede(strata, fact, memories)
        memories.append({"id": rid or -1, "text": fact})
        if rid:
            _link_source(strata, rid, ev_id)
        if fact not in before:
            added.append(fact)
    return added


_STORE_POLISH_PROMPT = """You are cleaning up a voice assistant's long-term memory store. Below are the \
stored memories, numbered. Some were written by older, sloppier versions of the system.
For each memory decide ONE of:
- keep     — already a clean, durable, third-person fact. Most memories should be kept.
- rewrite  — the fact is real but badly written: verbatim transcription garble, filler words, \
starts with "The user", contains abbreviations that sound wrong when read aloud \
("Arvada, Co." -> "Arvada, Colorado"), or is a fragment, or bakes tense into an event \
("upcoming outing for the Fourth of July" -> "Has an outing on the Fourth of July"; drop \
"upcoming"/"will"/"went" — the day is fixed, its past/future is worked out from the date). \
Rewrite as ONE clean third-person fact. \
Keep every anchor detail (names, dates, times, numbers, places) exactly. The rewritten text must \
NEVER begin with "The user" — start with the verb or noun ("Explained…", "Has a…", "Lives in…").
- delete   — not a durable fact at all: a question ("How to remove memories"), a test command \
("This: my favorite planet is..."), a one-time outing with no lasting significance \
("Went to the park today"), or an exact duplicate of another memory (delete the worse-written one).
Durable interests survive: "Getting into bouldering" is a keep even though a single park visit is not.

CONSOLIDATE FRAGMENTS: when two or more memories are pieces of the SAME fact \
("Has an interview Tuesday" + "Interviewing at Acme"), rewrite ONE of them into the \
single complete fact ("Has an interview Tuesday with Acme") and delete the other piece(s). \
Only combine memories that truly describe the same thing — NEVER merge memories that differ \
on a name, date, company, place, or number: two different interviews, two different trips, \
and two different people each stay as their own separate memory.

Return ONLY a JSON array of changes — memories to keep are simply omitted:
[{{"n": 3, "action": "rewrite", "text": "..."}}, {{"n": 7, "action": "delete"}}]
If everything is already clean, return [].

Memories:
{numbered}

JSON array:"""


def polish_memory_store(memories: list[dict], cfg: dict | None = None) -> list[dict]:
    """One-shot smoothing pass over the whole memory store (the Memories page
    button). Returns a list of {"id", "action": "rewrite"|"delete", "text"?} —
    the caller applies them. The LLM sees positional numbers, never raw 64-bit
    ids (models mangle long ids)."""
    if not memories:
        return []
    numbered = "\n".join(f"{i+1}. {m['text']}" for i, m in enumerate(memories))
    try:
        raw = llm_complete([{"role": "user",
                             "content": _STORE_POLISH_PROMPT.format(numbered=numbered)}],
                           _mem_cfg(cfg))
    except Exception as e:
        print(f"[memory] store polish failed: {e}")
        return []
    m = _first_json_array(raw)
    if not m:
        return []
    try:
        arr = json.loads(m)
    except Exception:
        return []
    out = []
    for ch in arr:
        if not isinstance(ch, dict):
            continue
        try:
            idx = int(ch.get("n", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(memories)):
            continue
        action = ch.get("action")
        if action == "delete":
            out.append({"id": memories[idx]["id"], "action": "delete",
                        "old": memories[idx]["text"]})
        elif action == "rewrite":
            text = _clean(ch.get("text") or "")
            if len(text) > 2 and text != memories[idx]["text"]:
                out.append({"id": memories[idx]["id"], "action": "rewrite",
                            "text": text, "old": memories[idx]["text"]})
    return out


def consolidation_proposals(strata) -> list[dict]:
    """L1.5 aggregation buffer (Strata's reflection engine): cluster near-duplicate
    L1 facts and return reviewable MERGE suggestions, shaped like the polish changes
    so both feed one review list. Propose-only — we reset the proposal store and lift
    the auto-accept threshold so even exact duplicates wait for the user's click,
    matching the app's 'nothing changes until you approve' contract."""
    try:
        from strata.reflection.proposals import ProposalKind, ProposalState, ProposalStore
    except Exception as e:
        print(f"[memory] L1.5 unavailable: {e}")
        return []
    refl = getattr(strata, "reflection", None)
    if refl is None:
        return []
    refl.proposals = ProposalStore()        # fresh — drop any prior run's proposals
    refl.auto_accept_threshold = 2.0        # nothing auto-merges; every cluster is reviewable
    try:
        refl.consolidate()
    except Exception as e:
        print(f"[memory] L1.5 consolidate failed: {e}")
        return []
    id2text = {m["id"]: m["text"] for m in list_memories(strata)}
    out = []
    for p in refl.proposals.list(state=ProposalState.USER_REVIEW_REQUIRED):
        if p.kind is not ProposalKind.MERGE:
            continue
        members = [{"id": str(rid), "text": id2text.get(rid, "")} for rid in p.record_ids]
        if len(members) < 2 or any(not m["text"] for m in members):
            continue                        # a member vanished since clustering — skip
        out.append({"action": "merge", "proposal_id": p.id, "members": members,
                    "text": p.target_content or members[0]["text"]})
    return out


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
            _mem_cfg(cfg))
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
            vecs = (embedder.embed_batch(_RECALL_EXEMPLARS)
                    if hasattr(embedder, "embed_batch")
                    else [embedder.embed(x) for x in _RECALL_EXEMPLARS])
        except Exception:
            vecs = []
        _exemplar_vecs[embedder.model_id] = vecs
    meta = any(_cosine(qv, ev) >= meta_thresh for ev in vecs)

    # embed all uncached recaps in ONE round trip (was one HTTP call per recap)
    missing = [s for s in summaries if s["id"] not in _recap_vecs]
    if missing:
        if len(_recap_vecs) > 512:   # bounded: re-embeddable cache, cheap to rebuild
            _recap_vecs.clear()
        try:
            if hasattr(embedder, "embed_batch"):
                for s, v in zip(missing, embedder.embed_batch([s["text"] for s in missing])):
                    _recap_vecs[s["id"]] = v
            else:
                for s in missing:
                    _recap_vecs[s["id"]] = embedder.embed(s["text"])
        except Exception as e:
            print(f"[recap] recap embed failed: {e}")
            for s in missing:
                _recap_vecs.setdefault(s["id"], None)

    selected = list(summaries[:max_n]) if meta else []
    for s in summaries:
        rv = _recap_vecs.get(s["id"])
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
            assessment = json.loads(m)
    except Exception as e:
        print(f"[review] assessment failed: {e}")
    return {"added": added, "assessment": assessment}


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
        m = _first_json_array(raw)
        arr = json.loads(m) if m else []
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
    from mlx_audio.tts.utils import load_model
    from strata.gateway.api import Strata

    asr = load_asr()
    tts = load_model(TTS_MODEL)
    patch_kokoro_tts()
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    strata = Strata.open(db_path=DB_PATH)
    print(f"Ready. LLM={LLM_MODEL}  ASR=Parakeet-V3 ({ASR_BACKEND})  "
          f"TTS=Kokoro  DB={DB_PATH}")

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
