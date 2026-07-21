# Duplex fine-tune — could we make PersonaPlex not-dumb?

Follow-up to [DUPLEX-PROTOTYPE.md](DUPLEX-PROTOTYPE.md), which ended **no-go**:
PersonaPlex-7B works as plumbing but is too weak/flaky to feel like it knows the
user. This doc asks the next question — **can fine-tuning fix it, on what
hardware, with what data?** — and sketches the pipeline we'd actually build.

TL;DR: the failures that made it feel broken are mostly a **distribution
mismatch** (it was tuned to be a scripted *banking service agent*), which is
LoRA-fixable. The raw-intelligence ceiling of the Helium-7B backbone is not.
Unsloth is the wrong tool; a lone 3090 is marginal; the real cost is generating
paired stereo audio, not the GPU time.

## The architecture (confirmed from the port source)

PersonaPlex = the **Moshi** stack, not a standard HF text model:

- **Temporal transformer** (the backbone): `d_model=4096, 32 layers, 32 heads`,
  RoPE, RMSNorm, gating — i.e. **Helium-7B**. Text vocab 32k (SentencePiece).
  This is where "smarts" live, and where the weakness is.
  (`personaplex_mlx/models/lm.py::config_personaplex_7b_v1`)
- **Depformer**: a small `d_model=1024, 6-layer` transformer that predicts the
  **8 Mimi audio codebooks** per frame, conditioned on the backbone.
- **Mimi** neural codec: 8 codebooks @ **12.5 Hz** (80 ms/frame). Text and audio
  are interleaved streams — that interleaving *is* the full-duplex behaviour.
- Full weights 16 GB bf16 (~7B).

Consequence: you can't treat this like "a 7B — just QLoRA it." Any training must
respect the dual-transformer + Mimi-token + two-stream structure.

## Why it's "dumb" — it's partly distribution, not just capability

What PersonaPlex was trained on (this reframes everything):

| Phase | Data |
|---|---|
| Moshi pretrain | unsupervised audio, backbone init from **Helium** |
| Moshi duplex | **Fisher** — 2 000 h phone calls (8 kHz → 24 kHz) |
| Moshi instruct | **20 000 h synthetic TTS** "Moshi↔user" dialogues |
| **PersonaPlex** | **< 5 000 h**: Fisher + **synthetic *banking* service dialogues** |

Re-read the exp-1 transcripts against that: *"Hello, this is Steven. **Thanks
for calling.** … **Welcome to coaching.**"* — that's **call-center register.**
The model was tuned to be a *service agent reading an account script*, and we
asked it to be a warm companion with persistent memory. So it treats the Strata
role-prompt like a banking script: reads names off it, flips Steven↔Stephen,
greets like a support line, and adopts the user's identity ("*this is* Steven")
because "agent inhabits the account" is in its prior.

**That mismatch is exactly what LoRA fixes** — NVIDIA *built* PersonaPlex this way
with < 5 k hours. Re-skinning the persona with our own data is in-distribution
for the method.

## What LoRA can and cannot buy

| Fixable (behavioral / distribution) | NOT fixable (base capability) |
|---|---|
| Persona attribution (stop "*this is* Steven") | Raw world knowledge |
| Call-center register → warm, brief companion | Multi-step reasoning |
| Faithful use of the injected role-prompt | Complex instruction chains |
| **Abstention** — "you haven't told me that" (anti-confabulation) | Making Helium-7B ≈ qwen-36B |
| Spoken brevity, no lists-aloud | |

The exp-1 failures that felt damning are almost all in the left column. The
right-column ceiling matters *less* for a memory companion, because most turns
are about the user's own life (which is in the prompt), not general knowledge.

## Feasibility — tools & hardware

- **Unsloth: no.** It accelerates standard HF transformers (Llama/Qwen/Gemma)
  and single-stream TTS (Orpheus/Sesame). It cannot load Moshi's dual-transformer
  duplex graph. Category mismatch.
- **Real tool: [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)**
  — LoRA rank 128, scaling 2.0, all linear layers (optional embedding FT).
  Caveat: it targets Kyutai **Moshiko/Moshika**, *not* PersonaPlex — using it on
  the NVIDIA checkpoint needs config/weight-layout porting. Alternative: fine-tune
  Moshiko/Moshika directly (supported) and drop PersonaPlex.
- **3090 / 24 GB: marginal.** Published peak is **39.6 GB** single-GPU (batch 16,
  ~100 s clips); 8×H100 shards to 23.7 GB each. To fit 24 GB: batch 1–2 + short
  `duration_sec` (~20–40 s), which their docs warn *degrades quality*. Fine for a
  short-clip PoC; a rented **A100/H100 (40–80 GB)** or 2×3090 is the sane rig.
- **The real bottleneck is data**, not compute (below).

## Datasets that move it most (ranked)

Target format (moshi-finetune): **stereo WAV — left = assistant, right = user** —
plus a `.jsonl` manifest `{path, duration}` and per-file timestamped `.json`
transcript. So every item must become two-channel audio (TTS both sides).

1. **Self-synthesized grounded-companion dialogues (by far #1).** From real +
   synthetic Strata role-prompts, scripted by qwen3.6, TTS'd to stereo. Must
   include: 2nd-person memory recall, **memory-*absent* → abstention**,
   corrections ("no, it's Steven"), warm/brief spoken style. Overwrites the
   banking prior with our behaviour and guarantees train==inference prompt format.
2. **Abstention / negative-probe contrast sets.** (profile, question-about-an-
   absent-fact → "you haven't mentioned that"). Small, high-impact vs confabulation.
3. **Multi-Session Chat (MSC)** + **Synthetic-Persona-Chat** — cross-session
   "remember what you told me last time" continuity + persona consistency.
4. **DailyDialog / DailyTalk** (moshi-finetune's example is DailyTalkContiguous)
   — natural 2-speaker everyday talk to protect *naturalness*, so persona-tuning
   doesn't overfit into stiffness.
5. **A little Fisher/Switchboard-style 2-channel audio** — not to add duplex
   skill (base has it) but as a regularizer so barge-in/backchannel don't drift.

Mix: heavy on (1)+(2), seasoned with (3)/(4), a dash of (5). The whole game is
replacing the service-agent distribution with a grounded-companion one *without*
breaking the duplex feel.

## The data-generation pipeline (what we'd actually build)

Most of this is code we already have pieces of. Everything reads the real store
**read-only** and writes only to scratch — no `~/.vui` writes (DATA-SAFETY).

**Stage A — script dialogues (text, uses the classic brain).**
1. Build role-prompts with the existing `duplex/role_prompt.py` — same assembly
   the model sees at inference. Seed from the real store (à la
   `personaplex_prompt.py`, read-only) **and** synthesize many fake
   profiles/memory-sets (varied names, sizes, domains) so it generalizes past one
   user.
2. For each prompt, have **qwen3.6 (Ollama)** write a multi-turn spoken dialogue:
   assistant grounded in the prompt, user improvising. Bake in the behaviour
   targets — recall, abstention on absent facts, a correction, brevity, no lists.
   Emit labeled turns `(speaker, text)`.
3. Add explicit abstention/contrast items (dataset #2).

**Stage B — render stereo audio.**
1. Assistant turns → TTS in the **target PersonaPlex voice** (the port's own
   `run_tts.py` / voices, so timbre matches inference).
2. User turns → TTS in **many varied voices** (multi-speaker TTS: the project's
   `mlx-audio`, or an external many-speaker set) so the user channel isn't one voice.
3. Lay turns on a timeline with natural gaps/short overlaps → write a **24 kHz
   2-channel WAV** (L=assistant, R=user). Because we placed the turns, we can emit
   the timestamped transcript `.json` directly (cleaner than re-running
   `annotate.py`).

**Stage C — manifest & train.**
1. Emit `.jsonl` + per-file `.json`.
2. moshi-finetune: LoRA r128; batch 1–2 + short `duration_sec` on a 3090, or
   rent an A100/H100. Port PersonaPlex config *or* train Moshiko.

**Stage D — eval (reuse what exp-2 already built).** The scratchpad harnesses
(`seed_sweep.py`, `temp_sweep.py`, `name_diag.py`) are exactly the eval loop:
name-recall hit-rate across seeds, abstention rate on negative probes,
persona-attribution correctness. Promote them into `duplex/eval/` so pre/post-FT
A/Bs are one command. (They're currently in session scratch and will be lost —
worth copying in if we pursue this.)

## Bottom line

- **Possible?** Yes via moshi-finetune-style LoRA — *not* Unsloth. Budget the
  PersonaPlex-porting glue, or just fine-tune Moshiko/Moshika.
- **On a 3090?** Marginal — short-clip PoC yes, real run wants A100/H100.
- **Biggest cost:** the stereo companion-dialogue corpus (TTS + timing), not GPU.
- **Realistic payoff:** the persona-confusion / name-flip / confabulation /
  call-center-vibe failures are addressable; the "can't reason about the world"
  ceiling is not — likely tolerable for a memory companion, not for a general
  assistant. This stays a **mode**, never the smart brain.

## Sources

- [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) (LoRA config, stereo data format, VRAM table)
- [Moshi paper (kyutai.org)](https://kyutai.org/Moshi.pdf) (training phases, Fisher, 20k h synthetic)
- [NVIDIA PersonaPlex — ADLR](https://research.nvidia.com/labs/adlr/personaplex/) (<5k h, Fisher + banking service dialogues, Helium backbone)
- [nu-dialogue/moshi-finetune](https://github.com/nu-dialogue/moshi-finetune) (J-Moshi — precedent for finetuning on your own spoken-dialogue data)
- [Unsloth TTS fine-tuning docs](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) (scope = single-stream TTS, not duplex Moshi)
