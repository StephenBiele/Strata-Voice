# Strata Voice — feature inventory

A voice assistant that runs on your Mac and actually remembers you. First the
short version; below it, the complete inventory.

## The short version

- **Just talk** — no buttons to hold. It hears when you start and stop speaking,
  and you can interrupt it mid-sentence just by talking over it.
- **It remembers you** — facts about your life, past conversations, upcoming
  plans. Say "remember…" or "forget that" and it does — permanently, across
  every conversation.
- **Private by default** — speech recognition, the voice, and memory all live on
  your Mac. Nothing leaves your machine unless you deliberately connect a cloud
  model — and it warns you when you do.
- **Real voices, even yours** — ten preset voices from real human recordings, or
  clone any voice from a 15-second clip. Two engines: one fast and clean, one
  expressive.
- **It can laugh and sigh** — with the expressive engine, it adds natural touches
  — a laugh at a joke, a sigh of sympathy — when the moment genuinely fits.
- **Type or talk, same brain** — switch between voice and typing any time. One
  conversation, one memory. Mute it and it answers in text; unmute and it speaks.
- **Off the record when you want** — an incognito toggle: it still knows you,
  but saves nothing from the conversation.
- **It reads your files** — upload a resume, notes, or a document and ask about
  it; it pulls up just the relevant parts.
- **Bring any AI model** — works out of the box with a free local model, or plug
  in any provider — OpenAI-compatible, local or on another machine on your
  network. Memory works with all of them.
- **Memory you can see** — a hub with a timeline of every conversation, every
  saved fact (and where it came from), plus tools to clean up, test recall, and
  review.
- **Simple until you want depth** — settings show just the essentials; a gear
  reveals the full toolkit for people who like to tune the engine.
- **One command to install, one to remove** — an installer that checks
  everything and picks models to fit your Mac; an uninstaller that asks before
  touching anything.

## The complete inventory

### Talking & conversation

- **Hands-free turn taking** — on-device voice-activity detection senses when you
  start and stop speaking and sends your turn automatically.
- **Barge-in** — talking over a reply interrupts it instantly (toggleable; leans
  on browser echo cancellation).
- **True mute** — releases the microphone at the OS level (the indicator light
  goes dark; other apps like Discord get it back) and pauses the call timer,
  while a reply in progress finishes speaking.
- **Voice tuning** — sensitivity, pause-before-it-replies, lead-in padding, and
  minimum speech length, all in plain language.
- **In-call live tuning panel** — adjust those knobs mid-conversation while
  testing.
- **Unified text chat** — type instead of talking; same session, same memory.
  Replies stream in live.
- **Smart muted typing** — sending a message while muted auto-opens the
  conversation panel (the reply arrives as text); unmuted typing gets a spoken
  reply.
- **Live captions** — see what it's saying as it speaks, plus a "reasoning…"
  indicator for thinking models.
- **Call timer** that pauses while muted and resumes where it left off.
- **Incognito** — a ghost toggle: nothing saved, while it still uses what it
  already knows.
- **Graceful endings** — ending a call keeps you on the call page; one tap starts
  the next conversation.
- **A living orb** — breathes when idle, ripples when listening, blooms when
  speaking, dims when muted.
- **Responsive UI** — adapts from desktop to phone; the conversation becomes a
  bottom sheet, touch targets sized for fingers.
- **Robust turn handling** — abandoned or interrupted turns abort cleanly
  (heartbeats keep the server from hanging on a closed tab).

### Speech recognition

- **On-device transcription** — Parakeet V3 by default; audio never leaves the
  machine.
- **Model picker** — swap to Whisper Turbo / Small / Tiny or Qwen3-ASR in
  Settings, described in plain language.
- **Fail-safe switching** — a new model must load and pass a test transcription
  before it replaces the working one; a bad pick can't brick a call.

### Voice & speech output

- **Two engines** — Kokoro (fast, clean, many voices) and Chatterbox-Turbo
  (expressive, cloning); switch live in Settings with download progress shown.
- **Expressive emotion** — the model itself drops in `[laugh]`, `[chuckle]`,
  `[sigh]`, `[gasp]`, `[groan]`, `[yawn]` where the moment fits, coached to
  match your mood.
- **Tag guardrails** — only the eight tags the engine genuinely performs are
  spoken; made-up ones are silently dropped, none may start a sentence (they
  don't render there), and they never appear in transcripts or memories.
- **Ten preset voices** — real public-domain human recordings (CC0, LibriVox),
  five women and five men, first-name labels with honest descriptions.
- **Voice cloning** — upload any short clip and it speaks in that voice;
  normalized automatically, removable any time, cached so switching is instant.
- **Kokoro voice packs** — American and British voices with a speed dial.
- **Cadence controls** — phrasing (sentence / clause / hybrid / whole), trim
  silence, exact gap between phrases, punctuation smoothing.
- **No mid-reply gaps** — each sentence synthesizes while the previous one
  plays, tuned per engine.
- **Voice preview** — hear your current picks (even unsaved) with your own test
  line.
- **Speaks like a person** — persona tuned for short spoken replies; punctuation
  shaped so the voice doesn't sound choppy.
- **Time awareness** — knows the date and time of day, so greetings and "today"
  make sense.

### Memory

- **Nothing stored verbatim** — a smoothing layer rewrites speech into clean
  third-person facts; anchor details (names, dates, numbers, places) are kept
  exactly as you said them.
- **Explicit remember** — "remember that…" is judged and polished into one clean
  fact before storage.
- **Implicit extraction** — a background pass distills durable facts out of
  natural conversation, with recent turns as context so fragments like "it's
  next Tuesday" resolve.
- **End-of-call harvest** — one pass over the whole transcript assembles facts
  scattered across turns into complete memories.
- **Instant, rule-based forgetting** — deletion never depends on an AI's
  judgment, and anything forgotten mid-call is never brought up again.
- **Supersession** — "actually, change that…" replaces the old fact instead of
  piling up contradictions.
- **Source-linked** — every fact links back to the verbatim moment it came from;
  the ground truth is one click away.
- **Semantic recall** — small stores inject everything (perfect recall); past a
  threshold, vector + keyword search selects what's relevant to this turn.
- **No cross-wiring similar events** — when you ask about one specific thing
  ("the Thursday interview", "that appointment yesterday"), it answers from the
  single most relevant memory instead of blending in a similar one — while still
  listing them all when you ask for "all" or "both". Measured by a disambiguation
  benchmark (`tests/memory_benchmark.py`) that plants close-in-time events and
  scores recall vs. collisions.
- **Conversation recaps** — each session is summarized into an episodic layer,
  surfaced only when relevant, so "what were we talking about last time?" works.
- **Standing rules (L4 guardrails)** — plain-language rules the assistant always
  follows ("call me Alex", "never suggest alcohol", "keep replies short"). Stored
  separately from facts and injected into every reply unconditionally — they're
  never smoothed away, forgotten, or subject to relevance gating.
- **"Here's what I've learned"** — after a few new facts accumulate, a quiet,
  tappable card offers to show you what it's picked up (opening the Memories
  list). Low-frequency by design: it won't reappear until even more is learned,
  and viewing the list clears it. Memory is never hidden — this just surfaces it.
- **Memory hub** — Timeline (every turn with the facts learned in that moment),
  Rules, Past chats, Memories (dated, newest first), and Reference files in one place.
- **Memory tools** — "Smooth memories" proposes cleanups (garbled phrasing,
  duplicates, junk, and combining fragments of the same fact into one) that you
  approve or dismiss one by one; plus recall tests and re-review of past
  conversations.
- **Time-aware recall** — "what did I do yesterday?", "what have I been up to
  lately?" surface memories from that actual time window, each tagged with when
  it happened, so two similar events from different times don't get confused.
- **Works with any backend** — the full memory system runs through whatever
  model you connect, local or remote (verified end-to-end).
- **Separate memory model** — optionally use a different model for background
  memory work, on either backend.
- **Background, and visible** — memory writing never delays your next reply, and
  a status pill shows when it's working.

### Models & connections

- **Local by default** — Ollama models chosen to fit your Mac (Lightweight
  ~16 GB, Recommended 32 GB+).
- **Any OpenAI-compatible endpoint** — any provider, hosted locally or on
  another machine on your network.
- **Privacy notice** — choosing a third-party endpoint shows exactly what leaves
  your machine; a local endpoint stays private.
- **Keys in the keychain** — API keys live in the macOS Keychain, never on disk.
- **Test connection** — verify a backend before committing to it.
- **Model lists** — installed Ollama models, or whatever your API endpoint
  serves, listed in the pickers.
- **Generation controls** — temperature, top-p, max response tokens, context
  window.
- **Thinking toggle** — allow or skip step-by-step reasoning (off is faster for
  voice).
- **Editable persona** — rewrite the system prompt freely; memory instructions
  are always appended so memory keeps working.

### Web lookups (optional)

- **Off by default** — search queries leave your machine (DuckDuckGo, no API
  key), so it's a deliberate opt-in with an honest explanation in Settings.
- **Gated, not constant** — a quick pre-turn check decides whether the question
  actually needs fresh info (scores, hours, weather, news); everyday
  conversation never triggers a search.
- **"Can you double check that?"** — questioning a previous claim makes it
  verify that claim against the web.
- **Live weather data** — weather questions skip search snippets and pull real
  forecast numbers from Open-Meteo (keyless, no signup), for your profile
  location or any place you name.
- **Location- and date-aware queries** — "what's the weather tomorrow?" uses
  your profile location and today's date; timeless questions stay undated.
- **Verifiable sources** — a small chip under each web answer shows site icons
  and a count; hover or tap lists the actual links it read.
- **Voice-shaped answers** — one or two spoken sentences with just the answer;
  it says when the results don't contain one instead of guessing.
- **Five-minute working memory** — results stay in RAM for follow-ups ("tell me
  more"), refresh while actively discussed, then vanish. Never written to disk,
  never enters the transcript or long-term memory.
- **"Checking the web…"** status in the call UI while it searches.

### Files & profile

- **Reference files** — upload PDF / DOCX / text (a resume, notes, a story) and
  ask about it.
- **Smart retrieval** — small files go to the model whole; large ones are
  chunked and embedded so each question pulls only the relevant passages.
- **Profile** — name, preferred name, location, gender; seen every turn so it
  greets and refers to you correctly.
- **Onboarding** — a gentle first-run flow for profile and model setup.

### Updates

- **In-app updates** — Settings → Updates shows your version, checks what's new
  (with the list of changes), and updates in one click: pulls the latest,
  refreshes dependencies, rebuilds the Mac app if you have one, restarts itself,
  and the page reconnects when it's back.
- **Safe by design** — the update applies only if it merges cleanly
  (`--ff-only`); local edits are never overwritten, and a failure keeps the
  running version.

### Settings & polish

- **Advanced-mode gear** — settings show just the essentials by default; one
  toggle reveals the full toolkit inside the same sections. The choice persists.
- **Save from anywhere** — a Save button pinned in the header plus one at the
  bottom; saving never kicks you off the page.
- **Honest loading states** — engine and model switches show a spinner with
  what's happening and how long it might take; failures keep your previous
  working choice.
- **Plain-language copy** — every control says what it does in words a person
  uses.
- **Detail care** — slider fills that actually meet the knob, accessible touch
  targets, reduced-motion support, always-fresh pages after updates.
- **A/B preview culture** — voices and cadence are auditioned, not guessed at.
- **Background-work pill** — memory processing is never invisible.

### Install & under the hood

- **One-command install** — checks prerequisites (Python, ffmpeg, Ollama —
  installed if missing), builds the environment, offers hardware tiers, and can
  build the Mac app for you.
- **A real Mac app** — `./make_app.sh` builds Strata Voice.app: a native window
  (WKWebView, no Electron) with its own Dock icon and a proper microphone
  permission prompt. It starts Ollama and the server if needed, and only shuts
  down what it started.
- **Guarded uninstall** — prompts before every destructive step; your data is
  asked about twice.
- **Your data, outside the repo** — everything lives in `~/.vui`, untouched by
  reinstalls.
- **Network-ready** — `VOICE_HOST` exposes the app to other devices;
  `OLLAMA_URL` can point at another machine (embeddings follow it).
- **CLI mode** — a minimal terminal push-to-talk mode with no UI.
- **Considered architecture** — a single-threaded model server (the GPU likes it
  that way), a separate channel for interrupt detection, and background workers
  for everything slow.
- **Performance care** — pooled connections, batched embeddings, cached reads,
  capped context injection on long calls.
- **Cross-platform path** — a researched plan for NVIDIA / AMD on Windows and
  Linux ([docs/CROSS-PLATFORM.md](CROSS-PLATFORM.md)).
