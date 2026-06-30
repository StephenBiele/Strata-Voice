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

- **Push-to-talk, not VAD.** Hold **Space**, speak, release. No turn-detection
  guesswork, no mis-fires, no awkward latency — it responds the instant you let go.
- **On-device speech.** [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx)
  for ASR and [Kokoro via `mlx-audio`](https://github.com/Blaizzy/mlx-audio) for
  TTS, both running on Apple's MLX. Fast and private.
- **Real memory.** Backed by Strata Memory: durable facts, supersession (updating
  a fact keeps history), and canonical-first deletion (forgetting actually forgets).
- **Bring any model.** Local Ollama out of the box, or point it at any
  OpenAI-compatible endpoint (llama.cpp, LM Studio, vLLM, OpenAI, …).
- **A living UI.** A calm "call" interface with an orb that breathes at rest,
  ripples while it listens, and blooms while it speaks.

## Features

- **Voice loop** — hold Space to talk; barge-in (start talking while it speaks to interrupt).
- **Profile** — name, preferred name, location, gender; carried into every conversation.
  First-run onboarding asks once, then never again.
- **Memory** — state durable facts and they're stored; "forget X" hard-deletes them.
  A Memories page lists everything it knows, each removable.
- **Past chats** — every call's transcript is saved and re-readable.
- **Reference files** — upload a PDF / DOCX / text file (resume, profile, notes) and
  ask about it by voice.
- **Settings** — edit the assistant's name, edit the system prompt, toggle "thinking",
  and choose your model backend.
- **Provider-agnostic** — Quick (local Ollama, pick or paste any model) or Advanced
  (any OpenAI-compatible endpoint + key). API keys are stored in the **macOS
  Keychain**, never on disk.
- **Local-first** — profile, transcripts, memories, and uploads all live under
  `~/.vui/`. The only network call is to your chosen LLM.

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

Open **http://localhost:8765**, click **Start conversation**, and **hold Space to
talk**. macOS will ask for microphone access the first time — allow it.

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

**Recall scales with your memory.** While the store is small, every fact is
injected each turn (perfect recall, essentially free). Once it grows past a
threshold, the assistant instead asks Strata's `recall()` for the most relevant
facts for what you just said — its **vector + lexical + resolver** stack, using a
local embedding model (`nomic-embed-text`). So "tell me about my pet" surfaces
"has a dog named Molly" even with no shared words.

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
