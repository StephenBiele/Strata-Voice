#!/usr/bin/env python3
"""Memory WRITE benchmark — how good is the system at deciding what to store?

The recall benchmark measures reading; this measures writing. Each scenario is a
mini conversation replayed through the REAL write pipeline (per-turn extraction +
end-of-call harvest, same functions the live app calls). Then we check what
landed in the store against ground truth, scored by plain keyword matching — no
LLM judges anything:

  - Recall     — were the durable facts captured? (should_store present)
  - Precision  — was junk kept out? (should_not_store absent — fleeting outings,
                 the assistant's own suggestions, things the user negated)
  - Coexist    — do two distinct-but-similar facts BOTH survive dedup, or does one
                 clobber the other?

Needs Ollama. Scenarios live in write_scenarios.json.
Usage:  .venv/bin/python tests/write_benchmark.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import voicechat as vc  # noqa: E402

LLM_CFG = {"backend": "ollama", "ollama_model": os.environ.get("BENCH_MODEL", "qwen3.5:4b"),
           "temperature": 0.0}


def _present(key, texts):
    blob = " ".join(texts).lower()
    return key.lower() in blob


def run_scenario(sc, embedder) -> dict:
    with tempfile.TemporaryDirectory() as d:
        from strata.gateway.api import Strata
        st = Strata.open(db_path=str(Path(d) / "w.db"), embedder=embedder)
        turns = sc["turns"]
        # replay the conversation through the real per-turn write path
        for i, t in enumerate(turns):
            if t["role"] != "user":
                continue
            eid = vc.record_event(st, t["content"])
            existing = [m["text"] for m in vc.list_memories(st)]
            ctx = "\n".join(f"{x['role']}: {x['content']}" for x in turns[max(0, i - 6):i])
            facts = vc.extract_facts_llm(t["content"], existing, LLM_CFG, context=ctx)
            vc.add_facts(st, facts, eid, LLM_CFG)   # cfg -> exercises write-side adjudication
        # end-of-call harvest (grounded, source-linked)
        existing = [m["text"] for m in vc.list_memories(st)]
        harvested = vc.harvest_session_facts(turns, existing, LLM_CFG)
        vc.add_harvested_facts(st, harvested, LLM_CFG)

        stored = [m["text"] for m in vc.list_memories(st)]
        st.close()

    missed = [k for k in sc["should_store"] if not _present(k, stored)]
    junk = [k for k in sc["should_not_store"] if _present(k, stored)]
    clobbered = [pair for pair in sc.get("coexist", [])
                 if not all(_present(k, stored) for k in pair)]
    return {"id": sc["id"], "stored": stored, "missed": missed,
            "junk": junk, "clobbered": clobbered,
            "n_store": len(sc["should_store"]), "n_avoid": len(sc["should_not_store"])}


def main():
    data = json.loads((Path(__file__).parent / "write_scenarios.json").read_text())
    embedder = vc.make_embedder()   # ok if None; extraction doesn't need it
    print(f"Memory WRITE benchmark · {len(data['scenarios'])} scenarios · "
          f"model={LLM_CFG['ollama_model']}\n")
    rows = [run_scenario(sc, embedder) for sc in data["scenarios"]]

    for r in rows:
        print(f"── {r['id']}")
        for s in r["stored"]:
            print(f"     stored: {s!r}")
        if r["missed"]:    print(f"   ✗ MISSED (should have stored): {r['missed']}")
        if r["junk"]:      print(f"   ✗ JUNK (should not have stored): {r['junk']}")
        if r["clobbered"]: print(f"   ✗ CLOBBERED (a distinct fact was lost): {r['clobbered']}")
        if not (r["missed"] or r["junk"] or r["clobbered"]):
            print("   ✓ clean")
        print()

    n = len(rows)
    store_tot = sum(r["n_store"] for r in rows)
    store_hit = sum(r["n_store"] - len(r["missed"]) for r in rows)
    junk_tot = sum(len(r["junk"]) for r in rows)
    clob = sum(1 for r in rows if r["clobbered"])
    print("═══ WRITE BASELINE ═══")
    print(f"  Durable-fact recall (stored what it should)  : "
          f"{(store_hit / store_tot * 100) if store_tot else 100:.0f}%  ({store_hit}/{store_tot})")
    print(f"  Junk stored (kept what it shouldn't)         : {junk_tot} item(s)")
    print(f"  Distinct facts clobbered by dedup            : {clob}/{n} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
