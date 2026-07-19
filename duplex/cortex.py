"""Background "cortex" for duplex (PersonaPlex) mode.

PersonaPlex is the mouth and ears: it listens and speaks full-duplex, handling
pauses, barge-in, and delivery — but it's a 7B backbone with no per-turn prompt
assembly, so it can't do Strata's recall or memory writes. This module is the
compensating brain that runs alongside it, driven by the conversation
transcript (PersonaPlex emits its own text stream; Parakeet transcribes the
user side). It does two jobs, both reusing the EXISTING memory code paths:

  1. WRITE  — every user turn is recorded as an L0 event; at session end (and
     optionally mid-session) the standard harvest extracts durable facts and
     stores them source-linked. The speech model never touches the store, so
     the DATA-SAFETY contract is unchanged (writes still go through Strata's
     dedup/supersede + quote-grounding, same as the classic pipeline).

  2. RECALL — after each user turn it recomputes the memory set that SHOULD be
     in context (vc.select_memories) and, when new relevant memories have
     entered that the current role prompt can't see, asks the transport to
     re-prefill: rebuild the KV cache as [fresh role prompt] + [replayed
     session token history] and swap it in at a step boundary. That swap is the
     one piece that needs PersonaPlex's internals — it lives behind the
     ReprefillSink interface, so this orchestration is testable without MLX.

Below RECALL_THRESHOLD memories, select_memories returns everything every turn,
so the candidate set never grows past the session-start prompt and no
re-prefill ever fires — small stores are fully served by injection alone. Above
it, recall selects different memories as the topic shifts, and those shifts are
exactly what triggers a swap.

Nothing here imports MLX or runs audio; it's pure orchestration over `vc`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import role_prompt


class ReprefillSink(Protocol):
    """The transport's hook for swapping the role prompt mid-session. The real
    implementation (a PersonaPlex sidecar adapter) rebuilds the KV cache from
    the new prompt plus the replayed conversation history and swaps it in at a
    step boundary; the prototype/tests use a recording fake."""

    def reprefill(self, prompt: str) -> None: ...


class Backend(Protocol):
    """Everything the cortex needs from Strata + the memory pipeline, narrowed
    to a handful of calls so tests can supply a fake without strata or Ollama.
    The default (StrataBackend) wraps voicechat/`vc` directly."""

    def record_event(self, text: str) -> None: ...
    def list_memories(self) -> list[str]: ...
    def list_rules(self) -> list[str]: ...
    def select_memories(self, query: str) -> list[str]: ...
    def harvest(self, turns: list[dict], existing: list[str]) -> list[dict]: ...
    def store_facts(self, harvested: list[dict]) -> list[str]: ...


@dataclass
class CortexConfig:
    assistant_name: str = "Sage"
    user_name: str = "the user"
    profile: list[str] = field(default_factory=list)   # pre-formatted profile lines
    budget: int = role_prompt.DEFAULT_BUDGET
    # Re-prefill only when at least this many genuinely-new relevant memories have
    # entered the recall set since the last prefill. 1 = swap as soon as anything
    # new becomes relevant; raise it to trade freshness for fewer swaps.
    min_new_memories: int = 1


def plan_reprefill(current: list[str], candidate: list[str],
                   min_new: int) -> list[str] | None:
    """Pure decision: given the memories currently visible to the model and the
    freshly-recalled candidate set, return the new set to prefill, or None to
    leave the prompt alone.

    Fire only when the candidate brings in memories the model can't currently
    see (candidate - current). Memories merely dropping out of the top-K don't
    force a swap — the model just stops referencing them — which keeps the
    prompt from thrashing and its size bounded to the recalled set."""
    cur = set(current)
    new = [m for m in candidate if m not in cur]
    if len(new) < max(1, min_new):
        return None
    return candidate


class Cortex:
    """Drives WRITE + RECALL for one duplex session. Feed it transcript turns as
    they arrive; call finish() when the conversation ends.

    Threading note: in the live app the recall/harvest LLM calls are slow and
    must run off the audio path — mirror the server's recap pattern and call
    on_user_turn/finish from a background thread. The class itself holds only
    in-memory session state and does no locking."""

    def __init__(self, backend: Backend, sink: ReprefillSink,
                 cfg: CortexConfig | None = None) -> None:
        self._be = backend
        self._sink = sink
        self._cfg = cfg or CortexConfig()
        self._turns: list[dict] = []
        self._in_context: list[str] = []      # memories currently in the role prompt
        self._prefills = 0

    # ---- session lifecycle ---------------------------------------------------
    def start_prompt(self) -> str:
        """Build and record the session-start role prompt (the full recalled
        set). Return it so the transport can launch PersonaPlex with it."""
        mems = self._be.list_memories()
        # Session start has no query yet; seed with the whole store (small) or a
        # generic recall (large) so the opening prompt isn't empty.
        candidate = mems if len(mems) <= _threshold() else self._be.select_memories("")
        prompt, dropped = self._render(candidate)
        self._in_context = candidate
        if dropped:
            print(f"[cortex] start prompt: dropped {dropped} oldest memories to fit "
                  f"{self._cfg.budget} chars")
        return prompt

    def on_user_turn(self, text: str) -> bool:
        """Record the user turn and re-prefill if newly-relevant memories entered
        the recall set. Returns True iff a re-prefill was issued."""
        text = (text or "").strip()
        if not text:
            return False
        self._turns.append({"role": "user", "content": text})
        self._be.record_event(text)            # L0 event — harvest source-links to it
        candidate = self._be.select_memories(text)
        target = plan_reprefill(self._in_context, candidate, self._cfg.min_new_memories)
        if target is None:
            return False
        prompt, dropped = self._render(target)
        self._in_context = target
        self._prefills += 1
        if dropped:
            print(f"[cortex] re-prefill #{self._prefills}: dropped {dropped} oldest "
                  f"memories to fit budget")
        self._sink.reprefill(prompt)
        return True

    def on_assistant_turn(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._turns.append({"role": "assistant", "content": text})

    def finish(self) -> list[str]:
        """End of conversation: harvest durable facts from the whole transcript
        and store them (source-linked). Returns the newly-stored facts. Mirrors
        server._recap_session's harvest, minus the episodic recap."""
        if not self._turns:
            return []
        existing = self._be.list_memories()
        harvested = self._be.harvest(self._turns, existing)
        added = self._be.store_facts(harvested) if harvested else []
        if added:
            print("[cortex] harvested:", added)
        return added

    # ---- internals -----------------------------------------------------------
    @property
    def prefill_count(self) -> int:
        return self._prefills

    def _render(self, memories: list[str]) -> tuple[str, int]:
        return role_prompt.build(
            assistant_name=self._cfg.assistant_name,
            user_name=self._cfg.user_name,
            profile=self._cfg.profile,
            rules=self._be.list_rules(),
            memories=memories,
            budget=self._cfg.budget)


def _threshold() -> int:
    """RECALL_THRESHOLD, read lazily so this module imports with no heavy deps."""
    try:
        import voicechat as vc
        return vc.RECALL_THRESHOLD
    except Exception:
        return 12


# ---- default backend: wraps voicechat/Strata --------------------------------
class StrataBackend:
    """Production Backend: the real memory pipeline. Imports `vc` lazily so the
    cortex module (and its tests) load without voicechat's audio dependencies.

    ``cfg`` is the memory-LLM config the server already builds (_mem_llm_cfg):
    backend/model/temperature for the background extraction calls."""

    def __init__(self, strata, cfg: dict | None = None) -> None:
        import voicechat as vc
        self._vc = vc
        self._strata = strata
        self._cfg = cfg

    def record_event(self, text: str) -> None:
        self._vc.record_event(self._strata, text)

    def list_memories(self) -> list[str]:
        return [m["text"] for m in self._vc.list_memories(self._strata)]

    def list_rules(self) -> list[str]:
        return [r["text"] for r in self._vc.list_rules(self._strata)]

    def select_memories(self, query: str) -> list[str]:
        return self._vc.select_memories(self._strata, query)

    def harvest(self, turns: list[dict], existing: list[str]) -> list[dict]:
        return self._vc.harvest_session_facts(turns, existing, self._cfg)

    def store_facts(self, harvested: list[dict]) -> list[str]:
        return self._vc.add_harvested_facts(self._strata, harvested, self._cfg)
