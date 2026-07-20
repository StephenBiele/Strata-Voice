#!/usr/bin/env python3
"""Unit tests for the duplex cortex — the WRITE + RECALL orchestration that runs
alongside PersonaPlex (duplex/cortex.py) and the shared role-prompt builder
(duplex/role_prompt.py).

Deliberately dependency-free: a FakeBackend stands in for Strata + the memory
pipeline and a recording sink stands in for the PersonaPlex transport, so this
runs anywhere (no strata, no Ollama, no MLX) and exercises the actual decision
logic — when events get recorded, when a re-prefill fires, when it doesn't, and
that harvest routes through the store. The real memory functions the production
backend wraps are covered by the existing write/recall benchmarks.

Usage:  python tests/test_duplex_cortex.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from duplex import role_prompt                       # noqa: E402
from duplex.cortex import Cortex, CortexConfig, plan_reprefill  # noqa: E402


# ---- fakes ------------------------------------------------------------------
class RecordingSink:
    def __init__(self):
        self.prompts = []
        self.deltas = []

    def reprefill(self, prompt, added):
        self.prompts.append(prompt)
        self.deltas.append(added)


class FakeBackend:
    """In-memory stand-in. `select_returns` is a list of memory-sets to hand back
    from successive select_memories() calls, so a test can script a topic shift.
    Recording of events and stored facts lets tests assert the write path ran."""

    def __init__(self, memories=None, rules=None, select_returns=None,
                 harvest_returns=None):
        self._memories = list(memories or [])
        self._rules = list(rules or [])
        self._select_returns = list(select_returns or [])
        self._harvest_returns = list(harvest_returns or [])
        self.events = []
        self.harvest_calls = []
        self.stored = []

    def record_event(self, text):
        self.events.append(text)

    def list_memories(self):
        return list(self._memories)

    def list_rules(self):
        return list(self._rules)

    def select_memories(self, query):
        if self._select_returns:
            return self._select_returns.pop(0)
        return list(self._memories)

    def harvest(self, turns, existing):
        self.harvest_calls.append((list(turns), list(existing)))
        return self._harvest_returns.pop(0) if self._harvest_returns else []

    def store_facts(self, harvested):
        facts = [h["fact"] for h in harvested]
        self.stored.extend(facts)
        self._memories.extend(facts)
        return facts


# ---- test helpers -----------------------------------------------------------
_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}")


# ---- role_prompt builder ----------------------------------------------------
def test_role_prompt_sections():
    prompt, dropped = role_prompt.build(
        assistant_name="Sage", user_name="Alex",
        profile=["They live in Denver."],
        rules=["never call them buddy"],
        memories=["Has a dog named Molly", "Works as a nurse"])
    check("prompt names assistant and user", "Sage" in prompt and "Alex" in prompt)
    check("prompt includes profile", "Denver" in prompt)
    check("prompt includes rules", "never call them buddy" in prompt)
    check("prompt includes memories", "Molly" in prompt and "nurse" in prompt)
    check("nothing dropped under budget", dropped == 0)


def test_role_prompt_budget_drops_oldest():
    mems = [f"Fact number {i} about the user with some length" for i in range(50)]
    prompt, dropped = role_prompt.build(
        assistant_name="Sage", user_name="Alex", memories=mems, budget=600)
    check("budget enforced (prompt within ~budget)", len(prompt) <= 700)
    check("some memories dropped", dropped > 0)
    # newest (highest index) kept, oldest dropped
    check("newest memory kept", "Fact number 49" in prompt)
    check("oldest memory dropped", "Fact number 0 " not in prompt)


# ---- plan_reprefill (pure decision) -----------------------------------------
def test_plan_reprefill():
    check("no change -> no prefill",
          plan_reprefill(["a", "b"], ["a", "b"], 1) is None)
    check("dropout only -> no prefill",
          plan_reprefill(["a", "b"], ["a"], 1) is None)
    check("one new relevant -> prefill to candidate",
          plan_reprefill(["a"], ["a", "c"], 1) == ["a", "c"])
    check("min_new gate holds back a single new item",
          plan_reprefill(["a"], ["a", "c"], 2) is None)
    check("min_new met by two new items",
          plan_reprefill(["a"], ["a", "c", "d"], 2) == ["a", "c", "d"])


# ---- cortex: small store never re-prefills ----------------------------------
def test_small_store_no_reprefill():
    be = FakeBackend(memories=["Has a dog named Molly", "Works as a nurse"])
    sink = RecordingSink()
    cx = Cortex(be, sink, CortexConfig(assistant_name="Sage", user_name="Alex"))
    cx.start_prompt()
    # select always returns the full (unchanged) set for a small store
    cx.on_user_turn("hey how's it going")
    cx.on_user_turn("what's new with you")
    check("small store: user turns recorded as events", be.events ==
          ["hey how's it going", "what's new with you"])
    check("small store: no re-prefill ever", sink.prompts == [] and cx.prefill_count == 0)


# ---- cortex: topic shift in a large store triggers one re-prefill -----------
def test_large_store_reprefill_on_topic_shift():
    baseline = [f"Fact {i}" for i in range(20)]      # > threshold -> recall regime
    be = FakeBackend(
        memories=baseline,
        select_returns=[
            ["Fact 1", "Fact 2"],                    # start_prompt seed (query="")
            ["Fact 1", "Fact 2"],                    # turn 1: same -> no swap
            ["Fact 1", "Fact 2", "Kids are named Sam and Jo"],  # turn 2: new -> swap
            ["Fact 1", "Fact 2", "Kids are named Sam and Jo"],  # turn 3: same -> no swap
        ])
    sink = RecordingSink()
    cx = Cortex(be, sink, CortexConfig())
    cx.start_prompt()
    fired1 = cx.on_user_turn("morning")
    fired2 = cx.on_user_turn("remind me about the kids' schedule")
    fired3 = cx.on_user_turn("thanks")
    check("no swap when recall set stable", fired1 is False and fired3 is False)
    check("swap when a new relevant memory enters", fired2 is True)
    check("exactly one re-prefill issued", cx.prefill_count == 1 and len(sink.prompts) == 1)
    check("re-prefill prompt contains the newly relevant memory",
          "Kids are named Sam and Jo" in sink.prompts[0])
    check("re-prefill delta is just the newly-entered memory",
          sink.deltas == [["Kids are named Sam and Jo"]])
    check("all three user turns recorded as events", len(be.events) == 3)


# ---- cortex: finish() harvests and stores -----------------------------------
def test_finish_harvests_and_stores():
    be = FakeBackend(
        memories=["Works as a nurse"],
        harvest_returns=[[{"fact": "Has an interview next Tuesday",
                           "quote": "I've got an interview next Tuesday"}]])
    sink = RecordingSink()
    cx = Cortex(be, sink, CortexConfig())
    cx.start_prompt()
    cx.on_user_turn("I've got an interview next Tuesday")
    cx.on_assistant_turn("Oh nice, what's the role?")
    added = cx.finish()
    check("finish harvested over the full transcript", len(be.harvest_calls) == 1)
    check("harvested turns include user and assistant",
          len(be.harvest_calls[0][0]) == 2)
    check("finish stored the durable fact",
          added == ["Has an interview next Tuesday"] and
          "Has an interview next Tuesday" in be.stored)


def test_finish_empty_conversation_is_noop():
    be = FakeBackend(memories=["Works as a nurse"])
    cx = Cortex(be, RecordingSink(), CortexConfig())
    check("empty conversation harvests nothing", cx.finish() == [])
    check("no harvest call for empty conversation", be.harvest_calls == [])


def main():
    for t in (test_role_prompt_sections, test_role_prompt_budget_drops_oldest,
              test_plan_reprefill, test_small_store_no_reprefill,
              test_large_store_reprefill_on_topic_shift,
              test_finish_harvests_and_stores,
              test_finish_empty_conversation_is_noop):
        print(t.__name__)
        t()
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
