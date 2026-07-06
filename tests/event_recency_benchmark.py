#!/usr/bin/env python3
"""Event-recency benchmark.

A remembered event is FUTURE when you mention it and PAST once its day arrives.
This checks two layers:

  1. The deterministic resolver (_event_date + _recency_tag): given a fact and
     WHEN it was said, does it work out the event's real date and label it
     upcoming/past correctly? Fully deterministic — no model, no network.
  2. End-to-end: with a now-past event in the store, does the model actually use
     the past tense instead of calling it "upcoming"? (needs Ollama)

Usage:  .venv/bin/python tests/event_recency_benchmark.py
        BENCH_MODEL=qwen3.5:4b .venv/bin/python tests/event_recency_benchmark.py
"""
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voicechat as vc  # noqa: E402

D = 86_400_000
# Fixed "now" so the whole benchmark is deterministic: Monday, July 6 2026.
NOW = int(datetime(2026, 7, 6, 12, 0).timestamp() * 1000)


def days_before_now(n):
    return NOW - n * D


# ---- Part 1: deterministic resolver -----------------------------------------
# (fact, when_it_was_said_ms, expected_class, must_appear_in_tag)
#   class: "past" | "future" | "today" | "none"
RESOLVER_CASES = [
    ("Has an outing with aunt and uncle for the Fourth of July", days_before_now(5), "past", "past"),
    ("Watching fireworks on July 4th", days_before_now(3), "past", "past"),
    ("Has a dentist appointment on July 20", days_before_now(0), "future", "coming up"),
    ("Interview next Thursday", days_before_now(7), "past", "past"),       # said last Mon -> Jul 2
    ("Interview next Thursday", days_before_now(0), "future", "coming up"),  # said today  -> Jul 9
    ("Has a flight on 2026-12-25", days_before_now(0), "future", "coming up"),
    ("Party this weekend", days_before_now(4), "past", "past"),            # said Thu -> Sat Jul 4
    ("Doctor visit tomorrow", days_before_now(0), "future", "tomorrow"),
    ("Wedding in 3 days", days_before_now(0), "future", "coming up in 3 days"),
    ("Moving on the 15th of August", days_before_now(0), "future", "coming up"),
    ("Lives in Arvada, Colorado", days_before_now(30), "none", ""),
    ("Has a dog named Molly", days_before_now(30), "none", ""),
    ("Works as a nurse", days_before_now(30), "none", ""),
]


def classify(fact, said_ms):
    ev = vc._event_date(fact, said_ms)
    if ev is None:
        return "none", ""
    tag = vc._recency_tag(ev, NOW)
    d = (ev - NOW) / D
    cls = "today" if -0.5 <= d < 0.5 else ("future" if d >= 0.5 else "past")
    return cls, tag


def run_resolver():
    print("── Part 1: deterministic resolver ──")
    ok = 0
    for fact, said, want_cls, want_sub in RESOLVER_CASES:
        cls, tag = classify(fact, said)
        good = (cls == want_cls) and (want_sub.lower() in tag.lower())
        ok += good
        mark = "✓" if good else "✗"
        print(f"  {mark} [{cls:>6}] {fact!r} -> {tag!r}"
              + ("" if good else f"   (wanted {want_cls} / '{want_sub}')"))
    print(f"\n  resolver: {ok}/{len(RESOLVER_CASES)} correct\n")
    return ok, len(RESOLVER_CASES)


def run_ordering():
    """Past events must be pushed below still-live ones."""
    print("── Part 1b: past events demoted ──")
    mems = [
        {"text": "Has an outing for the Fourth of July", "t": days_before_now(5)},   # past
        {"text": "Lives in Arvada, Colorado", "t": days_before_now(40)},             # undated
        {"text": "Has a dentist appointment on July 20", "t": days_before_now(0)},   # future
    ]
    out = vc._order_by_recency(mems, NOW)
    july4_idx = next(i for i, s in enumerate(out) if "Fourth of July" in s)
    dentist_idx = next(i for i, s in enumerate(out) if "dentist" in s)
    ok = july4_idx > dentist_idx and july4_idx == len(out) - 1
    print("  order:", [s.split(" (")[0] for s in out])
    print(f"  {'✓' if ok else '✗'} passed event sits last\n")
    return int(ok), 1


# ---- Part 2: end-to-end (model uses past tense for a passed event) ----------
LLM_CFG = {"backend": "ollama", "ollama_model": os.environ.get("BENCH_MODEL", "qwen3.5:4b"),
           "temperature": 0.0}
# each: (facts, query, must NOT say, should feel past/future)
LLM_CASES = [
    {"id": "passed-event",
     "facts": [{"text": "Has an outing with aunt and uncle for the Fourth of July", "days_ago": 5}],
     "query": "Did I already do that Fourth of July thing with my aunt and uncle?",
     "avoid": ["upcoming", "coming up", "will be", "you have an outing", "you'll"],
     "want_any": ["did", "went", "already", "past", "yes"]},
    {"id": "still-upcoming",
     "facts": [{"text": "Has a dentist appointment on July 20", "days_ago": 0}],
     "query": "Is that dentist thing still coming up?",
     "avoid": ["already happened", "in the past", "you went"],
     "want_any": ["coming up", "yes", "still", "upcoming", "on the 20th", "july 20"]},
]


def run_llm():
    print(f"── Part 2: end-to-end · model={LLM_CFG['ollama_model']} ──")
    emb = vc.make_embedder(vc.OLLAMA_URL)
    ok = 0
    for sc in LLM_CASES:
        with tempfile.TemporaryDirectory() as d:
            from strata.gateway.api import Strata
            st = Strata.open(db_path=str(Path(d) / "e.db"), embedder=emb)
            for f in sc["facts"]:
                st.write_memory(f["text"], created_at=NOW - f["days_ago"] * D)
            vc.warm_index(st)
            ctx = vc.select_memories(st, sc["query"], semantic=True)
            msgs = vc.build_messages([{"role": "user", "content": sc["query"]}], ctx)
            try:
                reply = vc.llm_complete(msgs, LLM_CFG)
            except Exception as e:
                reply = f"(llm error: {e})"
            reply = re.sub(r"\[MEM_(ADD|DEL)\][^\n]*", "", reply).strip()
            low = reply.lower()
            clean = not any(a in low for a in sc["avoid"])
            hit = any(w in low for w in sc["want_any"])
            good = clean and hit
            ok += good
            print(f"  {'✓' if good else '✗'} {sc['id']}")
            print(f"     ctx  : {ctx}")
            print(f"     reply: {reply[:150]!r}\n")
    print(f"  end-to-end: {ok}/{len(LLM_CASES)} correct\n")
    return ok, len(LLM_CASES)


def main():
    print(f"Event-recency benchmark · now = {datetime.fromtimestamp(NOW/1000):%A, %B %d %Y}\n")
    r_ok, r_n = run_resolver()
    o_ok, o_n = run_ordering()
    print("═══ deterministic core ═══")
    print(f"  resolver + ordering: {r_ok + o_ok}/{r_n + o_n}\n")
    if "--no-llm" not in sys.argv:
        try:
            run_llm()
        except Exception as e:
            print(f"  (skipped end-to-end: {e})")


if __name__ == "__main__":
    main()
