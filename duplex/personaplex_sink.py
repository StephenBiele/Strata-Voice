"""ReprefillSink adapter for the personaplex-mlx port — the seam between the
cortex's memory decisions and PersonaPlex's live generation state.

STATUS: on-device scaffolding, NOT yet run. It is written against the port's
real API as read from mu-hashmi/personaplex-mlx (moshi_mlx LmGen), but this repo
has no MLX/strata, so every symbol marked "VERIFY" must be confirmed against the
installed port before trusting it — the port may rename or move these.

Why this shape (see docs/DUPLEX-PROTOTYPE.md "cache constraint"): LmGen holds a
causal KV cache stepped one 80ms audio frame per `gen.step()`. The system prompt
sits at the front, so replacing it means recomputing the whole cache — ~one step
per frame of history, i.e. roughly the conversation's length. Full mid-session
re-prefill is therefore impractical live. Two cheap moves remain, and this
adapter implements both:

  - APPEND (default): inject only the delta (newly-relevant memories) as text
    tokens into the ongoing stream at a pause — no reset, no replay, O(delta).
    Out-of-distribution (the model never trained on system text mid-dialogue),
    so its behaviour is the thing experiment 3 must check.
  - BOUNDARY_RESET: at a genuine break, reset_streaming() + set the full new
    prompt + step_system_prompts(). Cheap, but drops in-session short-term
    memory (what was just said this call). Use only between topics, not mid-turn.

MLX is single-threaded and the GPU stream lives on the model-loading thread, so
reprefill() MUST be called on that thread (hand it to the port's step loop, e.g.
via a queue it drains between frames) — never from the cortex's background
thread directly. `pending()`/`drain()` support that hand-off.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class SinkConfig:
    # "append" = inject the delta into the live text stream (cheap, OOD-risk).
    # "boundary_reset" = full reset+reprompt (cheap, loses in-session memory).
    strategy: str = "append"
    # How a memory delta is phrased when appended into the text stream. Kept
    # short and aside-like; the model reads the text stream as its own "inner
    # monologue", so this nudges what it now knows without a spoken line.
    append_template: str = "(You now also remember: {facts})"


class PersonaPlexSink:
    """Bridges cortex.ReprefillSink to a live personaplex-mlx generator.

    Pass the port's generator (`LmGen`), its text tokenizer, and the
    `wrap_with_system_tags` helper — the exact objects local_web.py builds. The
    adapter never blocks the audio loop: reprefill() just enqueues an action;
    the port's step loop calls drain() between frames to apply it on the model
    thread.
    """

    def __init__(self, gen, text_tokenizer, wrap_with_system_tags,
                 cfg: SinkConfig | None = None) -> None:
        self._gen = gen                              # VERIFY: LmGen instance
        self._tok = text_tokenizer                   # VERIFY: .encode(str) -> list[int]
        self._wrap = wrap_with_system_tags           # VERIFY: str -> str (system tags)
        self._cfg = cfg or SinkConfig()
        self._q: list[tuple[str, list[str]]] = []
        self._lock = threading.Lock()

    # ---- cortex side (any thread) -------------------------------------------
    def reprefill(self, prompt: str, added: list[str]) -> None:
        """Called by the cortex (background thread). Non-blocking: enqueue only."""
        with self._lock:
            self._q.append((prompt, list(added)))

    def pending(self) -> bool:
        with self._lock:
            return bool(self._q)

    # ---- model side (MUST be the model-loading / step thread) ---------------
    def drain(self) -> None:
        """Apply queued updates. Call from the port's step loop at a frame
        boundary (ideally while the user is speaking, so a boundary reset never
        clips the assistant mid-word)."""
        with self._lock:
            batch, self._q = self._q, []
        for prompt, added in batch:
            if self._cfg.strategy == "boundary_reset":
                self._boundary_reset(prompt)
            else:
                self._append_delta(added)

    def _append_delta(self, added: list[str]) -> None:
        if not added:
            return
        text = self._cfg.append_template.format(facts="; ".join(added))
        try:
            tokens = self._tok.encode(text)          # VERIFY: encode signature
        except Exception as e:
            print(f"[sink] append encode failed: {e}")
            return
        # VERIFY: how the port feeds extra text tokens into the live stream.
        # local_web sets `gen.text_prompt_tokens` + calls `gen.step_system_prompts()`
        # at session start; the append path reuses that queue mid-stream so the
        # tokens ride the text channel over the next frames without a reset.
        try:
            existing = getattr(self._gen, "text_prompt_tokens", None) or []
            self._gen.text_prompt_tokens = list(existing) + list(tokens)
            self._gen.step_system_prompts()          # VERIFY: safe to call mid-session
            print(f"[sink] appended delta ({len(added)} memories, {len(tokens)} tokens)")
        except Exception as e:
            print(f"[sink] append inject failed: {e}")

    def _boundary_reset(self, prompt: str) -> None:
        # VERIFY: reset_streaming clears cache but keeps loaded weights/voice.
        try:
            self._gen.reset_streaming()
            self._gen.text_prompt_tokens = self._tok.encode(self._wrap(prompt))
            self._gen.step_system_prompts()
            print(f"[sink] boundary reset with fresh prompt ({len(prompt)} chars)")
        except Exception as e:
            print(f"[sink] boundary reset failed: {e}")


class LoggingSink:
    """No-MLX fallback: records calls and prints them. Lets the sidecar and
    cortex be wired and driven end-to-end (with a stub generator) before the
    real generator is available — and is what the cortex tests use in spirit."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def reprefill(self, prompt: str, added: list[str]) -> None:
        self.calls.append((prompt, list(added)))
        print(f"[sink:log] reprefill: +{added} (prompt {len(prompt)} chars)")
