#!/usr/bin/env python3
"""Memory recall benchmark — the North Star for disambiguation.

Plants known facts for two (or more) close-in-time events, fires a deliberately
VAGUE query ("who was with me at the barbecue again?"), and measures whether the
memory system surfaces the RIGHT event's facts without dragging in the wrong
event's. Two levels, both scored by plain keyword matching against ground truth
we constructed — NO LLM judges anything, so there's no LLM-grading-LLM loop:

  1. Retrieval   — does semantic recall put the target event's facts in context,
                   and keep the distractor event's facts out?
  2. Answer      — does the model's actual reply mention the target's details
                   and NOT the wrong event's (the real collision failure)?

Runs against a throwaway store with the real embedder (needs Ollama +
nomic-embed-text). Scenarios live in benchmark_scenarios.json — real datasets
(LongMemEval, LoCoMo) can be converted into that same shape later.

Usage:  .venv/bin/python tests/memory_benchmark.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import voicechat as vc  # noqa: E402

TOP_K = 8
LLM_CFG = {"backend": "ollama", "ollama_model": os.environ.get("BENCH_MODEL", "qwen3.5:4b"),
           "temperature": 0.3}


def _hit(keys, texts) -> int:
    """How many keys appear as a substring in any of the texts (case-insensitive)."""
    blob = " ".join(texts).lower()
    return sum(1 for k in keys if k.lower() in blob)


def run_scenario(sc, filler, embedder) -> dict:
    with tempfile.TemporaryDirectory() as d:
        from strata.gateway.api import Strata
        st = Strata.open(db_path=str(Path(d) / "b.db"), embedder=embedder)
        # size the store past the recall threshold so semantic selection actually
        # runs (below it the app injects everything and there's nothing to test)
        for f in filler:
            st.write_memory(f)
        # plant each event's facts, source-linked to the utterance L0 event
        for ev in sc["events"]:
            eid = vc.record_event(st, ev["utterance"])
            vc.add_facts(st, ev["facts"], eid)
        vc.warm_index(st)

        q = sc["query"]
        tgt, dis = sc["target_keys"], sc["distractor_keys"]

        # 1) retrieval level
        recalled = vc.recall_memories(st, q, top_k=TOP_K)
        r_recall = _hit(tgt, recalled) / len(tgt) if tgt else 1.0
        r_leak = _hit(dis, recalled)            # distractor facts that slipped into context

        # 2) answer level (the collision that actually reaches the user)
        mem_text = vc.select_memories(st, q, semantic=True)
        msgs = vc.build_messages([{"role": "user", "content": q}], mem_text)
        try:
            reply = vc.llm_complete(msgs, LLM_CFG)
        except Exception as e:
            reply = f"(llm error: {e})"
        a_recall = _hit(tgt, [reply]) / len(tgt) if tgt else 1.0
        a_collision = _hit(dis, [reply])        # wrong event's details in the actual reply
        st.close()

    return {"id": sc["id"], "kind": sc["kind"], "query": q, "reply": reply,
            "recalled": recalled,
            "r_recall": r_recall, "r_leak": r_leak, "n_dis": len(dis),
            "a_recall": a_recall, "a_collision": a_collision}


def main():
    data = json.loads((Path(__file__).parent / "benchmark_scenarios.json").read_text())
    embedder = vc.make_embedder()
    if embedder is None:
        print("✗ No embedder (need Ollama + nomic-embed-text). Semantic recall can't be "
              "measured without it — start Ollama and pull the model, then re-run.")
        return 1

    print(f"Memory recall benchmark · {len(data['scenarios'])} scenarios · model={LLM_CFG['ollama_model']}\n")
    rows = [run_scenario(sc, data["filler"], embedder) for sc in data["scenarios"]]

    for r in rows:
        print(f"── {r['id']}  ({r['kind']})")
        print(f"   vague query : {r['query']!r}")
        print(f"   reply       : {r['reply'][:150]!r}")
        leak = f"{r['r_leak']}/{r['n_dis']}" if r['n_dis'] else "n/a"
        print(f"   retrieval   : recall {r['r_recall']*100:3.0f}%   distractor leak {leak}")
        col = "CLEAN" if r['a_collision'] == 0 else f"COLLIDED ({r['a_collision']} wrong detail(s))"
        print(f"   answer      : recall {r['a_recall']*100:3.0f}%   {col}\n")

    n = len(rows)
    ar = sum(r["a_recall"] for r in rows) / n
    collided = sum(1 for r in rows if r["a_collision"] > 0)
    rr = sum(r["r_recall"] for r in rows) / n
    leaked = sum(1 for r in rows if r["r_leak"] > 0)
    print("═══ BASELINE ═══")
    print(f"  Answer recall (target details remembered) : {ar*100:.0f}%")
    print(f"  Answer collisions (wrong event bled in)   : {collided}/{n} scenarios")
    print(f"  Retrieval recall (target facts in context): {rr*100:.0f}%")
    print(f"  Retrieval leakage (distractor in context) : {leaked}/{n} scenarios")
    print("\n  (This is today's number. Two-pass retrieval + weighting should move "
          "collisions toward 0 without dropping recall.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
