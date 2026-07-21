# Duplex prototype — PersonaPlex feasibility experiments

Working notes for evaluating [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)
(full-duplex speech-to-speech, Moshi architecture, 7B — the only size released)
as an optional conversation mode. The model collapses VAD + ASR + LLM + TTS
into one continuously-streaming model, which is exactly what fixes pause
detection, barge-in, and emotional delivery — and exactly what removes the
per-turn prompt assembly our memory system rides on.

The design bet under test: **memory can still work** via
(1) session-start injection of the role prompt, and
(2) mid-session updates driven by the existing Ollama LLM running recall in the
background. Memory *writes* never involve the speech model: transcripts flow
into the existing harvest path, so the DATA-SAFETY contract is untouched.

### Cache constraint (from reading the port's source — reshapes the design)

`personaplex-mlx` runs moshi_mlx's `LmGen`: a **causal** KV cache stepped **one
80ms audio frame per `gen.step()`**, with the system prompt prefilled at the
front via `gen.text_prompt_tokens = tokenizer.encode(wrap_with_system_tags(p))`
then `gen.step_system_prompts()`; `reset_streaming()` clears it.

Because the cache is front-anchored, replacing the system prompt means
recomputing everything after it — the whole conversation's audio history,
replayed one step per frame. That is ~conversation-length wall time: **~750
steps (~50s) per minute of history** at the port's ~68ms/step. So the original
"re-prefill = rebuild `[new prompt] + [replayed history]`" idea is **not viable
for a live turn** — same cache-prefix constraint that prompt caching lives under.

The two cheap moves that survive, both implemented in `duplex/personaplex_sink.py`:

- **Append the delta** (default): inject only the *newly-relevant* memories as
  text into the ongoing stream at a pause — no reset, no replay, O(delta). It's
  out-of-distribution (the model never saw system text mid-dialogue), so its
  behaviour is what **experiment 3** must check.
- **Boundary reset**: at a genuine topic break, `reset_streaming()` + fresh
  prompt. Cheap, but drops in-session short-term memory (what was just said).

Session-start injection (1) stays the primary, always-cheap mechanism and fully
covers the ≤`RECALL_THRESHOLD` "perfect recall" case; (2) is these two deltas.

## What's built so far (runs off a Mac too)

- `duplex/role_prompt.py` — pure role-prompt assembly (head + profile + L4
  rules + budgeted memories). No I/O; shared by everything below.
- `duplex/personaplex_prompt.py` — experiment-1 CLI. Builds the prompt from the
  WHOLE store and (`--launch`) starts the MLX port with it. `--bare` control.
- `duplex/cortex.py` — the background brain: the WRITE + RECALL orchestration
  that would run alongside PersonaPlex. Records each user turn as an L0 event,
  decides when newly-relevant recall warrants a mid-session update
  (`plan_reprefill`), passing the sink both the full prompt and the *delta*, and
  harvests durable facts at session end — all through the *existing* memory
  functions (`select_memories`, `record_event`, `harvest_session_facts`,
  `add_harvested_facts`).
- `duplex/personaplex_sink.py` — the `ReprefillSink` adapter, written against the
  port's real API (append-delta + boundary-reset per the cache constraint
  above). On-device scaffolding: every port symbol is marked `VERIFY` and must
  be confirmed against the installed package; a `LoggingSink` fallback lets the
  wiring run with no MLX.
- `tests/test_duplex_cortex.py` — dependency-free tests (no strata/Ollama/MLX,
  so they run in CI and here) covering prompt budgeting, the update decision,
  delta computation, small-store-never-swaps, topic-shift-triggers-one-update,
  and harvest-on-finish. `python tests/test_duplex_cortex.py` → 27 pass.

The cortex is transport-agnostic: `StrataBackend` wraps the real pipeline for
the Mac, `FakeBackend` (in the test) scripts recall/harvest for verification.
What remains for a live duplex mode is the transport itself — a PersonaPlex
sidecar speaking WebSocket audio, Parakeet on the user side, and the sink's
`VERIFY` symbols confirmed + its `drain()` wired into the port's step loop.

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

- [x] Name recall: **works but ~75% reliable** (6/8 seeds, deterministic offline
      harness `scratchpad/exp2/seed_sweep.py`; live session missed — variance +
      ASR noise). When it hits it's often messy: confuses itself with the user
      ("*this is* Steven") and flips the profile's two name fields (Steven↔Stephen).
- [x] Unprompted relevance: **passes** — 2/8 seeds spontaneously opened with the
      stored *coaching* memory ("let's start the coaching") without being asked.
- [x] Negative probe: bare control **confabulates hard** ("the name of the actor
      who played the role in the movie…"). Injected condition is safer (admits/asks).
- [~] Rules (brief, no lists): held acceptably offline; live felt loose ("dumb as
      bricks" per operator). Not rigorously measured yet.
- [ ] Prompt budget: not yet stressed (ran at 1402/2000 chars).
- [~] Feel: live session — novel (real barge-in/backchannel) but the 7B is weak
      and inconsistent; operator's overall read was underwhelmed.

Promptcraft finding (matters for `role_prompt.py`): the **natural 3rd-person
persona prompt beats explicit instructions**. "The user's name is Steven" made
the model refuse ("I can't access your name"); the current
"You are Sage, Steven's personal voice assistant… Their full name is…" phrasing
is what recalled correctly. Do **not** rewrite role_prompt to be more directive.
Reliability lever (tested, `scratchpad/exp2/temp_sweep.py`): lowering
`--text-temp` monotonically raises name recall — **0.7→75%, 0.5→88%, 0.3→88%,
0.1→100%** (8 seeds each). But two caveats the hit-rate hides: (1) low temp
induces repetition/looping ("Hi, this is Steven. Hi, this is Steven…") — costs
the naturalness duplex exists for; ~0.3 is the sweet spot. (2) A
**temp-independent persona-confusion wart** persists at every temp — the model
introduces *itself* as Steven ("this is Steven") instead of addressing the user,
flips Steven↔Stephen, and misattributes ("It's Dan"). Verdict: reliability is
tunable, polish is 7B-limited. Needs a live feel-check at temp 0.3.

## Experiment 2 — step time + RAM (confirms the cache constraint, sizes the machine)

Source-reading already told us full mid-session re-prefill is impractical (one
step per frame of history). This experiment just quantifies the step cost that
implies, and confirms the model fits. Offline mode needs no new code:

```bash
say --data-format=LEI16@24000 -o probe.wav "What do you remember about me?"
time python -m personaplex_mlx.offline --voice NATF2 \
  --text-prompt "$(python duplex/personaplex_prompt.py 2>/dev/null)" \
  --input-wav probe.wav --output-wav out.wav --output-text out.json \
  --seed 42424242
```

`--seed` makes offline runs repeatable, so this doubles as a deterministic A/B
harness for prompt wording later.

- [x] Steady-state: **32.9 ms/step** (p50 over 295 steps; mean 32.9, p90 33.4,
      max 38.9) on an M4 Pro / 48 GB — RTF 0.41, ~47 ms headroom. Half the
      port's ~68 ms on M2 Max, so real-time is comfortable.
- [x] Implied replay cost: 750 frames/min × 32.9 ms = **~24.7 s to replay 1 min**
      of history → mid-turn full re-prefill still a non-starter (better than the
      ~50 s the 68 ms figure implied, still unusable live). Boundary reset must
      land at breaks, not mid-turn — confirmed.
- [x] RAM high-water mark: **10.4 GB** MLX peak footprint (max RSS 4.0 GB),
      4-bit. Fits alongside Ollama on 48 GB; a 16 GB Mac must evict the chat
      model first. Session-start prefill of the 321-token role prompt: 1.29 s
      (one-time).

## Experiment 3 — mid-stream delta injection (the real update mechanism)

Now load-bearing, not optional: this is how memory updates reach a live session
(`personaplex_sink.py`, append strategy). Inject a short memory delta as text
into the ongoing stream without a reset, and watch what the model does with it.
Out-of-distribution, so the failure modes are the point:

- [ ] Does it silently absorb the new fact and use it when relevant?
- [ ] Does it read the injected text aloud (leak)? Reword `append_template` if so.
- [ ] Does it ignore it entirely? If so, injection-only can't do live updates and
      we fall back to boundary reset at pauses (or session-start injection only).
- [ ] Latency of a `drain()` append at a frame boundary — any audible hitch?

First confirm the sink's `VERIFY` symbols against the installed port (how it
feeds extra text tokens mid-stream may differ from the session-start path), then
wire `drain()` into the step loop.

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

## Verdict (2026-07-20, M4 Pro)

**No-go on PersonaPlex-7B as a memory-bearing conversation mode.** The plumbing
all works — two-venv bridge, session-start injection, deterministic timing — and
the memory *mechanism* is sound (injection lands; recall tunable to 100% via
text-temp). But the 7B backbone is simply **too weak**: even at max reliability
it introduces itself as the user, flips the user's name, misattributes stored
facts, and turns robotic/loopy at the low temps that buy reliability. It never
crosses the bar of feeling like it knows you. The blocker is model capability,
not integration — so no transport was built (correctly gated on exp-1 passing,
which it didn't). Revisit only if a materially stronger full-duplex model ships
(PersonaPlex is 7B-only today) or a larger backbone can be run elsewhere.

What survives for a future attempt: the timing/RAM numbers, the promptcraft
finding (natural persona > explicit instructions), the text-temp reliability
curve, and all the transport-agnostic cortex/prompt scaffolding (untouched,
tests green).

## Findings

(fill in after the Mac runs)

| Date | Machine | Experiment | Result |
|------|---------|------------|--------|
| 2026-07-20 | M4 Pro / 48 GB | Setup | Port needs its own venv — Strata pins (numpy 2.5, mlx 0.31, torch 2.12, hf-hub 1.21) conflict hard with the port's (numpy <2.3, mlx <0.27, torch <2.8, hf-hub <0.29). Two-venv bridge: Strata venv builds the prompt, port venv (3.12, personaplex-mlx 0.3.0) runs the model. `PERSONAPLEX_PYTHON` env var added to `personaplex_prompt.py --launch`. |
| 2026-07-20 | M4 Pro / 48 GB | 2 (timing/RAM) | **32.9 ms/step** p50 (RTF 0.41, 47 ms headroom); replay ~24.7 s/min-history (mid-turn re-prefill still dead); **10.4 GB** peak footprint (fits w/ Ollama on 48 GB); session-start prefill 1.29 s / 321 tok. Offline run produced coherent audio + text. |
| 2026-07-20 | M4 Pro / 48 GB | 1 (memory injection) | **Mixed / yellow light.** Session-start injection *works* — name recalled **6/8 seeds (75%)**, coaching memory surfaced unprompted 2/8; bare control confabulates hard. But recall is flaky + messy (self/user confusion, Steven↔Stephen), and the live session felt weak ("dumb as bricks"). Promptcraft: natural 3rd-person persona > explicit "user's name is X" (the latter triggers refusals). **Reliability lever confirmed: text-temp 0.7→0.1 lifts recall 75%→100%** (0.3=88%, sweet spot); but low temp induces looping and a temp-independent persona-confusion wart ("introduces itself as Steven") remains. Reliability tunable, polish 7B-limited. Not a clean pass. |
| 2026-07-20 | M4 Pro / 48 GB | 1 live @ temp 0.3 | Operator feel-check: **NO-GO — "model is just way too dumb."** Reliability lever helped the metric but not the felt experience; 7B backbone is the ceiling. Transport not built. |
