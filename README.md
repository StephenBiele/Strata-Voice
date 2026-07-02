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
- **On-device speech.** Both ASR (Parakeet V3 TDT) and TTS (Kokoro) run through
  [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) on Apple's MLX — one library,
  one loader, and the engine behind the hands-free VAD. Fast and private.
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
- **Hands-free mode (experimental)** — flip the waveform toggle in a call and just talk:
  on-device voice-activity detection (Silero VAD via mlx-audio) notices when you start and
  stop speaking and sends your turn automatically. Start talking while it's mid-reply to
  **interrupt it** (barge-in). Tunable in plain language in Settings — voice sensitivity,
  the pause that ends your turn, lead-in padding, minimum speech length — and a "Show the
  in-call tuning panel" debug switch adds a live tuning panel during calls so you can dial
  in settings while actually talking. Hold-to-talk still works any time and takes
  precedence. Caveat: barge-in relies on your browser's echo cancellation — if it keeps
  interrupting itself through speakers, use headphones or turn "Interrupt while it's
  speaking" off.
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
  **LLM controls** (temperature, top-p, max tokens, and context window for Ollama),
  **voice cadence** (chunking: sentence / clause / hybrid / whole, edge-silence trimming,
  inter-chunk gap, punctuation smoothing) — all A/B-able via the voice Preview with your
  own preview text — and a **Speech recognition** picker to switch the ASR model live
  (Parakeet, Whisper Tiny/Small/Turbo, Qwen3-ASR). Switching swaps the model on the
  fly; a pick that can't load is caught and the previous model is kept.
- **Provider-agnostic** — Quick (local Ollama, pick or paste any model) or Advanced
  (any OpenAI-compatible endpoint + key). API keys are stored in the **macOS
  Keychain**, never on disk.
- **Responsive** — phone, tablet, or a small window: panels slide in/out, and the
  conversation becomes a bottom sheet on mobile.
- **Private by default** — profile, transcripts, memories, and uploads all live under
  `~/.vui/`; the mic opens only while you're holding to talk (in hands-free mode it stays
  open for the call — the OS indicator stays lit — but audio still never leaves the
  machine). The only network calls are to your chosen LLM (and the local embedder for
  recall).

## Install

One command, on a Mac with Apple Silicon:

```sh
git clone https://github.com/StephenBiele/Strata-Voice.git
cd Strata-Voice
./install.sh
```

The installer checks and installs the prerequisites (Python 3.12, ffmpeg, Ollama —
via Homebrew), sets up the Python environment, and asks which tier you want:

| Tier | Downloads | Fits | Chat + memory model |
| :--- | :--- | :--- | :--- |
| **Lightweight** | ~7 GB | 16 GB Macs | `qwen3.5:9b` |
| **Recommended** | ~24 GB | 32 GB+ Macs | `qwen3.6:latest` (36B) |

One model does both chat and memory: Ollama loads models one at a time, so a separate
memory model would evict the chat model on every background memory job (~8 s of reload
each way, measured). The Settings "Memory model" picker still exists for machines with
enough RAM to hold two models at once.

Both tiers use Parakeet V3 (0.6B) for speech-to-text, Kokoro for the voice, and
`nomic-embed-text` for semantic recall. Non-interactive installs:
`./install.sh --light` or `./install.sh --recommended`. Re-running the installer
is always safe — it never touches an existing profile or memories.

> While the repos are private, installing requires GitHub access to
> [`strata-memory`](https://github.com/StephenBiele/strata-memory) (pinned in
> `requirements.txt`). Once they're public, the commands above just work.

## Run

```sh
./start.sh
```

That starts Ollama if needed, launches the server, and opens
**http://localhost:8765** when it's ready. Click **Start conversation** (allow the
microphone prompt), then **press and hold the orb — or the button — to talk**, or
flip the waveform toggle for hands-free.

First launch downloads the speech models from Hugging Face (~1 GB) unless the
installer already prewarmed them.

## Uninstall

```sh
./uninstall.sh
```

Prompts before each step — the Python environment, your data (`~/.vui` — asks
twice), the Ollama models, and the cached speech models — then you delete the
folder. Nothing is removed without a yes.

<details>
<summary><strong>Manual install</strong> (what the script does)</summary>

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt   # strata-memory pins from GitHub
ollama pull qwen3.6:latest && ollama pull nomic-embed-text
ollama serve                                 # if not already running
.venv/bin/python server.py
```

To develop against a local strata-memory checkout:
`.venv/bin/pip install -e ../strata-memory`
</details>

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
| `VOICE_ASR_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | ASR model id (any mlx-audio STT model — or pick in Settings) |
| `VOICE_VAD_PORT` | `8766` | hands-free VAD channel (own port so speech detection keeps working mid-turn) |
| `OLLAMA_URL` | `http://localhost:11434` | URL endpoint for local Ollama server |

## How memory works

The assistant sees your profile and current memories every turn. When you state a
durable fact, the model appends a hidden `[MEM_ADD]` directive that the server
parses and writes to Strata (using supersession when it updates an existing fact).
"Forget …" appends `[MEM_DEL]`, which performs a canonical-first hard delete
(tombstone) in Strata. The directives are stripped before anything is spoken.

**Nothing is stored verbatim.** Voice transcripts are messy, so all memory writes go
through a smoothing layer: an explicit "remember…" is judged and rewritten into one
clean third-person fact, a background extraction pass (with recent turns as context)
distills implicit facts, and when a conversation ends a **whole-transcript harvest**
assembles facts that were scattered across turns ("I have an interview" … "it's next
Tuesday" … "for a builder role" → one complete memory). Deleting stays instant and
rule-based, and anything you forget mid-call won't be brought up again.

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
