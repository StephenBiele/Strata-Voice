# Duplex mode — handoff to a Mac session

> **CONCLUDED 2026-07-20 (M4 Pro): no-go.** Experiments 1–2 ran; the mechanism
> works but PersonaPlex-7B is too weak to feel like it knows the user, so no
> transport was built. Full verdict + numbers in
> [DUPLEX-PROTOTYPE.md](DUPLEX-PROTOTYPE.md). The steps below are kept for the
> record / a future stronger-model retry.

You're a Claude Code session running locally on the Mac (Apple Silicon, MLX +
Ollama available). This picks up feasibility work on adding **PersonaPlex**
(NVIDIA full-duplex speech-to-speech) as an optional conversation mode. All the
prep that didn't need hardware is already done, on branch
`claude/personaplex-integration-feasibility-zoap10`. Your job is the part that
needs a real Mac: run the experiments, then wire the transport **only if they
pass**.

**Read first:** [DUPLEX-PROTOTYPE.md](DUPLEX-PROTOTYPE.md) (full design +
experiments) and the root `CLAUDE.md` (project rules — single-threaded server,
data-safety, `./tests/run.sh` before any data-touching change).

## The one finding you must not re-litigate

Full mid-session "re-prefill" (rebuild the KV cache as `[new prompt] +
[replayed history]`) is **not viable** and was already discarded. PersonaPlex's
`LmGen` cache is causal + front-anchored, stepped one 80ms frame per
`gen.step()`, so replacing the front recomputes all history ≈ **50s of compute
per minute of conversation**. Don't rebuild it. The design uses only the cheap
moves: **session-start injection** (primary) and **mid-stream delta append** /
**boundary reset** (`duplex/personaplex_sink.py`).

## What's already built (no hardware needed, tested green here)

- `duplex/role_prompt.py` — pure prompt assembly (profile + L4 rules + budgeted memories)
- `duplex/personaplex_prompt.py` — experiment-1 CLI (`--launch`, `--bare`)
- `duplex/cortex.py` — background WRITE+RECALL brain over the existing memory pipeline
- `duplex/personaplex_sink.py` — `ReprefillSink` adapter (append + boundary-reset), symbols marked `VERIFY`
- `tests/test_duplex_cortex.py` — `python tests/test_duplex_cortex.py` → 27 pass (run this first to confirm your checkout)

## Do these in order — stop and report if one fails

1. **Setup.** `git clone https://github.com/mu-hashmi/personaplex-mlx && cd
   personaplex-mlx && pip install -e .`, `export HF_TOKEN=...` (needs access to
   `nvidia/personaplex-7b-v1`). ~5.3 GB at 4-bit. Headphones — no echo cancel.

2. **Experiment 1 — does injected memory work?** From the Strata-Voice repo root:
   `python duplex/personaplex_prompt.py --launch` vs `--bare`. Run the probe
   checklist in DUPLEX-PROTOTYPE.md (name recall, unprompted relevance,
   **confabulation on a negative probe**, rule adherence, feel). This is the
   go/no-go for the whole idea. Record results in the doc's findings table.

3. **Experiment 2 — step time + RAM.** Offline timing (command in the doc).
   Confirm < 80 ms/step and that it fits in RAM alongside (or instead of) the
   Ollama chat model. Fills in the cache-constraint numbers.

4. **Experiment 3 — confirm the sink, test delta injection.** Open the installed
   port's source and check every `VERIFY` symbol in `duplex/personaplex_sink.py`
   against reality (how it feeds extra text tokens mid-stream may differ from the
   session-start path). Then test whether an appended memory delta is absorbed,
   read aloud (leak), or ignored. That outcome decides whether live updates use
   append, boundary-reset, or session-start-only.

## Only after 1–3 pass — the transport

Build the sidecar (mirror the VAD micro-server pattern in `server.py`: separate
process/port, since the main server is single-threaded MLX): PersonaPlex over
WebSocket audio, Parakeet on the user side for a clean transcript, the `Cortex`
driven from those transcripts with `StrataBackend` + `PersonaPlexSink`, and
`sink.drain()` called from the port's step loop. Duplex is a **mode toggle**,
not a replacement — the classic pipeline stays the smart brain (7B backbone is
much weaker than qwen3.6 on knowledge/instructions). Keep all memory writes on
the existing Strata paths; run `./tests/run.sh` before committing data changes.

## Guardrails

- Don't push to other branches; stay on `claude/personaplex-integration-feasibility-zoap10`.
- The speech model never touches storage — writes go through the existing
  harvest/dedup/supersede paths only (DATA-SAFETY.md).
- Any UI you add must be checked in light AND dark mode (CLAUDE.md).
