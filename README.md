# Strata Voice

A local-first, push-to-talk voice assistant with real memory — built for Apple
Silicon. Speak to it, and it remembers what matters across sessions, knows your
profile, and can reference documents you upload (like a resume). Everything stays
on your device.

It pairs fast on-device speech models with [**Strata Memory**](https://github.com/StephenBiele/strata-memory),
a tiered, conflict-aware memory engine, so "remember…", "actually, change that…",
and "forget that" are first-class operations backed by a real canonical store —
not a flat text file.

```
mic → Parakeet V3 TDT (ASR) → LLM (Ollama or any OpenAI-compatible API) → Kokoro (TTS) → speakers
                                        ↕
                                Strata Memory (local SQLite)
```

## Why it feels good

- **Push-to-talk, not VAD.** Press and hold the orb (or the button) to talk, release
  to send — mouse or touch, no keyboard needed. The mic opens **only while you're
  holding**, so the OS "mic in use" indicator is off the rest of the time.
- **Or just type.** Prefer not to talk? "Send a message instead" opens a text chat
  (replies stream in, optionally spoken aloud). Typed and spoken turns share the same
  session, memory, and timeline — switch freely.
- **On-device speech.** [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx)
  for ASR and [Kokoro via `mlx-audio`](https://github.com/Blaizzy/mlx-audio) for
  TTS, both running on Apple's MLX. Fast and private.
- **Real memory.** Backed by Strata Memory: durable facts, supersession (updating
  a fact keeps history), semantic recall, and canonical-first deletion (forgetting
  actually forgets). Conversations even carry over ("what were we just talking about?").
- **Bring any model.** Local Ollama out of the box, or point it at any
  OpenAI-compatible endpoint (llama.cpp, LM Studio, vLLM, OpenAI, …), with tunable
  temperature / top-p / context.
- **A living, responsive UI.** A calm "call" interface with an orb that breathes,
  ripples, and blooms — and it adapts from desktop to phone (pages slide; the
  conversation becomes a bottom sheet on mobile).

## Features

- **Voice loop** — press & hold the orb (or the "Hold to talk" button) to speak; barge-in
  (press to talk while it's speaking to cut it off). Mouse or touch — no keyboard needed.
- **Text chat** — "Send a message instead" opens a typed chat; replies stream in and can
  optionally be spoken. Voice and text share one session, memory, and timeline.
- **Memory hub** — one place for **Timeline** (every turn, with the facts learned in that
  moment hanging off it), **Past chats** (saved transcripts), **Memories** (durable facts),
  and **Reference files**.
- **Memory tools** — find contradictions / duplicates / junk, run a recall test (does it
  actually remember?), and review a past conversation to fold in anything it missed.
- **Profile** — name, preferred name, location, gender; carried into every conversation.
  First-run onboarding asks once, then never again.
- **Reference files** — upload a PDF / DOCX / text file (resume, notes) and ask about it.
- **Incognito** — a ghost toggle for an off-the-record conversation: nothing is saved
  (no transcript, memory, event, or recap) while it still uses what it already knows.
- **Reasoning indicator** — with reasoning models, a live "reasoning… · 12s" timer so a
  long think-chain never looks frozen.
- **Settings** — assistant name, system prompt, "thinking" toggle, model backend,
  **LLM controls** (temperature, top-p, max tokens, and context window for Ollama), and
  **voice cadence** (chunking: sentence / clause / hybrid / whole, edge-silence trimming,
  inter-chunk gap, punctuation smoothing) — all A/B-able via the voice Preview with your
  own preview text.
- **Provider-agnostic** — Quick (local Ollama, pick or paste any model) or Advanced
  (any OpenAI-compatible endpoint + key). API keys are stored in the **macOS
  Keychain**, never on disk.
- **Responsive** — phone, tablet, or a small window: panels slide in/out, and the
  conversation becomes a bottom sheet on mobile.
- **Private by default** — profile, transcripts, memories, and uploads all live under
  `~/.vui/`; the mic opens only while you're holding to talk. The only network calls are
  to your chosen LLM (and the local embedder for recall).

## Requirements

- macOS on Apple Silicon (M1–M4)
- Python 3.12
- [ffmpeg](https://ffmpeg.org) — `brew install ffmpeg`
- An LLM backend — [Ollama](https://ollama.com) is easiest: `ollama pull qwen3.5:4b`
- An embedding model for semantic recall: `ollama pull nomic-embed-text` (optional —
  without it, memory falls back to injecting everything, which is fine at small scale)
- The [`strata-memory`](https://github.com/StephenBiele/strata-memory) package (installed below)

## Setup

```sh
git clone https://github.com/StephenBiele/Strata-Voice.git
cd Strata-Voice
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins `strata-memory` directly from its GitHub repo. To develop
against a local checkout instead:

```sh
.venv/bin/pip install -e ../strata-memory   # path to your strata-memory clone
```

## Run

```sh
ollama serve            # if not already running
.venv/bin/python server.py
```

Open **http://localhost:8765** and click **Start conversation** (the browser asks for
microphone access here the first time — allow it), then **press and hold the orb — or the
button — to talk**.

On first run the speech models download from Hugging Face (~1 GB total) and cache.

### CLI mode

A terminal-only push-to-talk loop is also included:

```sh
.venv/bin/python voicechat.py
```

## Configuration

Most things are set in the **Settings** page, but a few startup options are env vars:

| Var | Default | Notes |
| :--- | :--- | :--- |
| `VOICE_PORT` | `8765` | web server port |
| `VOICE_NAME` | `Sage` | initial assistant name (also editable in Settings) |
| `VOICE_LLM_MODEL` | `qwen3.5:4b` | default Ollama model |
| `OLLAMA_URL` | `http://localhost:11434` | URL endpoint for local Ollama server |

## How memory works

The assistant sees your profile and current memories every turn. When you state a
durable fact, the model appends a hidden `[MEM_ADD]` directive that the server
parses and writes to Strata (using supersession when it updates an existing fact).
"Forget …" appends `[MEM_DEL]`, which performs a canonical-first hard delete
(tombstone) in Strata. The directives are stripped before anything is spoken.

**Conversations carry over.** When a conversation ends it's recapped into Strata's
episodic layer. On a later turn, a retrieval layer embeds what you just said and — only
when it's relevant (asking to recall, "continue where we left off", or referring back to a
topic) — injects the matching recaps. So "what were we just talking about?" works, without
the recaps ever intruding on unrelated chat.

**Recall scales with your memory.** While the store is small, every fact is
injected each turn (perfect recall, essentially free). Once it grows past a
threshold, the assistant instead asks Strata's `recall()` for the most relevant
facts for what you just said — its **vector + lexical + resolver** stack, using a
local embedding model (`nomic-embed-text`). So "tell me about my pet" surfaces
"has a dog named Molly" even with no shared words.

**Incognito** turns skip all of this on the way *out* — no transcript, memory, event,
or recap is written — while still reading your existing profile and memories for context.

## Layout

```
server.py      web server: serves the UI + REST API (turns, profile, memories,
                 sessions, documents, settings); single-threaded for MLX's GPU stream
voicechat.py   models + pipeline: Parakeet ASR, Kokoro TTS, LLM calls (Ollama /
                 OpenAI-compatible), memory directive parsing; also the CLI loop
index.html     the entire front-end (single file): call UI, orb, and all panels
```

## Acknowledgements

Built on [Strata Memory](https://github.com/StephenBiele/strata-memory),
[parakeet-mlx](https://github.com/senstella/parakeet-mlx),
[mlx-audio](https://github.com/Blaizzy/mlx-audio) / [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M),
and [Ollama](https://ollama.com).
