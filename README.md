<div align="center">

# ✦ Strata Voice

**A voice assistant that actually remembers you.**

Push-to-talk or hands-free. On-device speech. Real, tiered memory.
Runs entirely on your Mac — nothing leaves your machine.

[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black)](#install)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#install)
[![MLX](https://img.shields.io/badge/speech-MLX%20on--device-orange)](#the-models)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Install](#install) · [Features](#features) · [How memory works](#how-memory-works) · [The models](#the-models) · [Configuration](#configuration) · [Uninstall](#uninstall)

</div>

---

Most voice assistants forget you the moment the window closes. Strata Voice is built
around the opposite idea: **your conversations accumulate into memory you own** — durable
facts, an episodic timeline, conversation recaps — stored in a local SQLite database by
[**Strata Memory**](https://github.com/StephenBiele/strata-memory), a tiered,
conflict-aware memory engine. "Remember…", "actually, change that…", and "forget that"
are first-class operations against a real canonical store, not lines in a text file.

Speech never leaves your machine: ASR, TTS, and voice-activity detection all run
on-device via Apple's MLX. The only network calls are to the LLM you choose — local
Ollama by default.

## Install

```sh
git clone https://github.com/StephenBiele/Strata-Voice.git
cd Strata-Voice
./install.sh
```

The installer checks the prerequisites (Python 3.12, ffmpeg, Ollama — installed via
Homebrew if missing), builds the Python environment, and asks which tier you want:

| Tier | Downloads | Fits | Chat + memory model |
| :--- | :--- | :--- | :--- |
| **Lightweight** | ~10 GB | 16 GB Macs | `gemma4:e4b` (fastest) |
| **Recommended** | ~24 GB | 32 GB+ Macs | `qwen3.6:latest` (36B) |

Then:

```sh
./start.sh
```

Ollama starts if needed, the server launches, and your browser opens to
**http://localhost:8765**. Click **Start conversation**, allow the microphone, and
**just start talking** — it hears when you speak. (Prefer push-to-talk? Flip "Manual
turns" in the conversation menu.)

> **Requirements:** macOS on Apple Silicon (M1 or newer). Non-interactive installs:
> `./install.sh --light` or `./install.sh --recommended`. Re-running the installer is
> always safe — it never touches an existing profile or memories. While the repos are
> private, installing needs GitHub access to `strata-memory`; once public, the commands
> above just work.

## Features

**Hands-free by default** — just talk: on-device Silero VAD detects when you start and
stop speaking and sends your turn automatically. Talking over a reply **interrupts it**
(barge-in). A **mute button** (red when muted) releases the mic entirely — the OS
indicator goes dark until you unmute. Every knob is tunable in plain language — voice
sensitivity, the pause that ends your turn, lead-in padding, minimum speech length — and
a debug switch adds an **in-call live tuning panel** so you can dial it in while actually
talking. Barge-in leans on your browser's echo cancellation: if it interrupts itself
through speakers, use headphones.

**Manual turns** — prefer push-to-talk? Flip "Manual turns" in the conversation menu (or
Settings) and it waits until you release before thinking: press & hold the orb or the
button to speak, and the mic opens *only while you hold*. One mode's controls at a time —
live shows the mute button, manual shows the hold button, never both. Ending a
conversation keeps you on the call page — one tap starts the next.

**Real memory** — nothing is stored verbatim from a voice transcript. A smoothing layer
rewrites explicit "remember…" requests into clean third-person facts, a background
extraction pass distills implicit ones, and an end-of-call harvest assembles facts
scattered across turns into complete memories. Deletion is instant and rule-based, and
anything you forget mid-call won't be brought up again. See
[How memory works](#how-memory-works).

**Memory hub** — Timeline (every turn, with the facts learned in that moment hanging off
it), Past chats, Memories, and Reference files, in one place. Memory tools find
contradictions and junk, run recall tests, and re-review past conversations.

**Text chat** — prefer typing? Replies stream in and can optionally be spoken. Voice and
text share one session, one memory, one timeline.

**Incognito** — a ghost toggle for off-the-record conversations: nothing is saved, while
it still uses what it already knows.

**Reference files** — upload a PDF / DOCX / text file (a resume, notes) and ask about it.

**Bring any model** — local Ollama out of the box, or any OpenAI-compatible endpoint
(llama.cpp, LM Studio, vLLM, OpenAI …). API keys live in the **macOS Keychain**, never
on disk. Full LLM controls (temperature, top-p, max tokens, context window), a live
speech-recognition picker (Parakeet, Whisper, Qwen3-ASR), voice cadence tuning with
A/B preview, and a background-work pill so memory processing is never invisible.

**A living, responsive UI** — a calm call interface with an orb that breathes, ripples,
and blooms; adapts from desktop to phone (panels slide, the conversation becomes a
bottom sheet).

## How memory works

The assistant sees your profile and current memories every turn. Memory writes flow
through three layers, none of them verbatim:

- **Explicit** — "remember that…" becomes a candidate that a polishing pass judges and
  rewrites into one clean third-person fact ("Has a job interview on Tuesday"). Anchor
  details — names, dates, times, numbers, places — are copied exactly as you said them.
- **Implicit** — a background extraction pass (with recent turns as context, temperature
  0) distills durable facts out of natural speech and skips what it can't confidently
  parse.
- **End-of-call harvest** — when a conversation ends, one pass over the whole transcript
  assembles facts that were scattered across turns ("I have an interview" … "it's next
  Tuesday" … "building internal tools" → one complete memory), and the conversation is
  recapped into the episodic layer so "what were we just talking about?" works next time.

Every fact is source-linked to the verbatim turn it came from (that's the Timeline), so
the ground truth is always one link away. While the store is small, every fact is
injected each turn; past a threshold, Strata's vector + lexical recall selects the most
relevant ones. Forgetting is deterministic and immediate — deletion never depends on an
LLM's judgment — and incognito turns write nothing at all.

## The models

| Role | Model | Runs on |
| :--- | :--- | :--- |
| Speech-to-text | Parakeet V3 TDT (0.6B) — swappable to Whisper / Qwen3-ASR in Settings | MLX, on-device |
| Voice | Kokoro 82M | MLX, on-device |
| Hands-free VAD | Silero VAD | MLX, on-device |
| Semantic recall | nomic-embed-text | Ollama, local |
| Chat + memory | your pick — tiers above, or any model in Settings | Ollama / any OpenAI-compatible API |

One model handles both chat and memory by default: Ollama loads models one at a time,
so a separate memory model would evict the chat model on every background job (~8 s of
reload each way, measured). The Settings "Memory model" picker exists for machines with
enough RAM to hold two models resident.

## Architecture

```
┌─────────────────────────── Browser (one page) ────────────────────────────┐
│   call UI · orb · hands-free capture · text chat · memory hub · settings  │
└───────────────┬────────────────────────────────────────────┬──────────────┘
                │ /turn/stream · /chat/stream (NDJSON)        │ /vad/feed (PCM)
┌───────────────▼─────────────────────────────┐  ┌────────────▼─────────────┐
│        main server :8765 (one thread)       │  │    VAD server :8766      │
│   Parakeet ASR → LLM → Kokoro TTS (MLX)     │  │  Silero VAD (mlx-audio)  │
│   background: memory worker · recap+harvest │  │  speech start/stop       │
└───────────────┬─────────────────────────────┘  └──────────────────────────┘
                ▼
┌─────────────────────────────────────────────┐
│         Strata Memory (local SQLite)        │
│    verbatim turns ← facts ← recaps          │
│    supersession · semantic recall · links   │
└─────────────────────────────────────────────┘
```

The main server is single-threaded on purpose (MLX's GPU stream lives in the thread
that loaded the models); everything slow — memory writes, recaps, fact harvest — runs on
background threads, and the VAD channel rides its own port so barge-in detection keeps
working while a reply is streaming.

## Project structure

```
Strata-Voice/
├── server.py       web server: UI + REST API (turns, memory, sessions, settings),
│                     the VAD micro-server, and the background memory workers
├── voicechat.py    models + pipeline: ASR/TTS loading, LLM calls, prompt assembly,
│                     the Strata Memory integration; also a minimal CLI mode
├── index.html      the entire front-end, one file
├── install.sh      one-command installer (tiers, models, prerequisites)
├── start.sh        start Ollama + the server, open the browser
└── uninstall.sh    guarded uninstaller — prompts before every destructive step
```

Your data lives in `~/.vui/` (profile, transcripts, memories, uploads) — outside the
repo, never committed, and untouched by reinstalls.

## Configuration

Most things live in **Settings**. Startup options are env vars:

| Var | Default | Notes |
| :--- | :--- | :--- |
| `VOICE_PORT` | `8765` | web server port |
| `VOICE_VAD_PORT` | `8766` | hands-free VAD channel |
| `VOICE_NAME` | `Sage` | initial assistant name |
| `VOICE_LLM_MODEL` | `qwen3.5:4b` | default Ollama model (the installer seeds your tier's pick) |
| `VOICE_ASR_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | ASR model id (or pick in Settings) |
| `OLLAMA_URL` | `http://localhost:11434` | local Ollama endpoint |

## Uninstall

```sh
./uninstall.sh
```

Prompts before each step — the Python environment, your data (`~/.vui`, asks twice),
the Ollama models, the cached speech models — then you delete the folder. Nothing is
removed without a yes.

<details>
<summary><strong>Manual install</strong> (what the script does)</summary>

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt   # strata-memory pins from GitHub
ollama pull gemma4:e4b && ollama pull nomic-embed-text
ollama serve                                 # if not already running
.venv/bin/python server.py
```

CLI-only mode (terminal push-to-talk, no UI): `.venv/bin/python voicechat.py`

To develop against a local strata-memory checkout:
`.venv/bin/pip install -e ../strata-memory`
</details>

## Acknowledgements

Built on [Strata Memory](https://github.com/StephenBiele/strata-memory),
[mlx-audio](https://github.com/Blaizzy/mlx-audio) /
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) /
[Parakeet](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3),
[Silero VAD](https://github.com/snakers4/silero-vad), and [Ollama](https://ollama.com).

## License

[MIT](LICENSE)

<div align="center">
<sub>✦ your conversations, your memory, your machine</sub>
</div>
