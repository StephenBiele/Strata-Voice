# Duplex prototype — PersonaPlex feasibility experiments

Working notes for evaluating [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)
(full-duplex speech-to-speech, Moshi architecture, 7B — the only size released)
as an optional conversation mode. The model collapses VAD + ASR + LLM + TTS
into one continuously-streaming model, which is exactly what fixes pause
detection, barge-in, and emotional delivery — and exactly what removes the
per-turn prompt assembly our memory system rides on.

The design bet under test: **memory can still work** via
(1) session-start injection of the role prompt, and
(2) background re-prefill ("hot swap") — rebuild the KV cache as
`[updated memory prompt] + [replayed session token history]` and swap it in
at a step boundary, driven by the existing Ollama LLM running recall in the
background. Memory *writes* never involve the speech model: transcripts flow
into the existing harvest path, so the DATA-SAFETY contract is untouched.

## Setup (Apple Silicon Mac, Python 3.12)

```bash
git clone https://github.com/mu-hashmi/personaplex-mlx && cd personaplex-mlx
pip install -e .
export HF_TOKEN=...   # needs access to nvidia/personaplex-7b-v1
```

~5.3 GB download at 4-bit. Wear headphones — the port has no echo
cancellation. English only; 16 preset voices (`NATF0-3`, `NATM0-3`,
`VARF0-4`, `VARM0-4`).

## Experiment 1 — session-start memory injection (cheap, run first)

Does a role prompt built from real Strata memories actually change what the
model knows and how it behaves? Below `RECALL_THRESHOLD` (12) memories this
is the *entire* memory-read story, so if it works, small stores need nothing
more.

```bash
# from the Strata-Voice repo root
python duplex/personaplex_prompt.py             # inspect the generated prompt
python duplex/personaplex_prompt.py --launch    # web mode with your memories
python duplex/personaplex_prompt.py --launch --bare   # control: no memories
```

Probe checklist (ask out loud, compare injected vs `--bare`):

- [ ] Name recall: "do you remember my name?" — uses profile name, doesn't invent one
- [ ] Unprompted relevance: steer near a stored fact without naming it — does it surface naturally?
- [ ] Negative probe: ask about something NOT in memory — does it admit not knowing, or confabulate? (7B Moshi-family models confabulate readily; measure how badly)
- [ ] Rules: set a standing rule in the prompt (e.g. "never call them buddy") — does it hold for a whole session?
- [ ] Prompt budget: default 2000 chars — does a maxed-out prompt degrade responsiveness or leak into speech (reading memories aloud)?
- [ ] Feel: interruptions, pauses, backchannels vs our pipeline — the reason we're here

## Experiment 2 — prefill speed (the load-bearing number for re-prefill)

The hot-swap design only works if rebuilding a cache over a few thousand
tokens takes ~a second, not minutes. Offline mode measures this without any
new code: time-to-first-audio ≈ prefill time of the text prompt.

```bash
say --data-format=LEI16@24000 -o probe.wav "What do you remember about me?"
time python -m personaplex_mlx.offline --voice NATF2 \
  --text-prompt "$(python duplex/personaplex_prompt.py 2>/dev/null)" \
  --input-wav probe.wav --output-wav out.wav --output-text out.json \
  --seed 42424242
```

Run once with a ~200-char prompt and once with the full ~2000-char prompt;
the delta is the prefill cost. Also note steady-state step time (the port
reported ~68 ms/step ≈ RTF 0.87 on M2 Max — confirm on this machine).
`--seed` makes offline runs repeatable, so this doubles as a deterministic
A/B harness for prompt wording later.

- [ ] Prefill rate: ______ tokens/sec → 2k-token history rebuild ≈ ______ s
- [ ] Steady-state: ______ ms/step (must stay < 80 ms for real-time)
- [ ] RAM high-water mark with model resident: ______ GB

## Experiment 3 — naive mid-stream text injection (only if 1 & 2 look good)

Force text tokens into the stream mid-session *without* re-prefilling.
Out-of-distribution — the model never trained on system text appearing
mid-dialogue — so expect weirdness (reading the memory aloud, ignoring it).
Requires poking at the port's generation loop; worth one afternoon at most,
and only to see whether we can skip the full re-prefill machinery sometimes.

## What a real integration would look like (if the bets hold)

- PersonaPlex as a sidecar process (same pattern as the VAD micro-server on
  :8766) speaking WebSocket audio to the page; the main server stays as-is.
- Parakeet running in parallel on the user's audio for a clean user-side
  transcript (PersonaPlex emits its own text stream for its side).
- The Ollama LLM as background "cortex": watches transcripts, runs Strata
  recall, triggers re-prefill when the relevant memory set changes, and runs
  the existing harvest for memory writes.
- Duplex is a *mode*, not a replacement: the 7B backbone is far weaker than
  the recommended qwen3.6 36B on knowledge and instruction-following, so the
  classic pipeline remains the smart mode.

## Findings

(fill in after the Mac runs)

| Date | Machine | Experiment | Result |
|------|---------|------------|--------|
|      |         |            |        |
