#!/usr/bin/env python3
"""Temporal recall benchmark — can it answer time-scoped questions?

Embeddings encode topic, not time, so "what did I do yesterday?" can't be ranked
by semantic similarity. This plants topically-similar events at KNOWN times, asks
time-scoped queries, and checks whether the RIGHT-window memory surfaces first
(and whether the reply stays on it). Deterministic scoring — no LLM judges.

A/B with the fix:  VOICE_TEMPORAL=0 (baseline) vs VOICE_TEMPORAL=1 (time-aware).
Needs Ollama. Usage:  .venv/bin/python tests/temporal_benchmark.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import voicechat as vc  # noqa: E402

LLM_CFG = {"backend": "ollama", "ollama_model": os.environ.get("BENCH_MODEL", "qwen3.5:4b"),
           "temperature": 0.0}
D = 86_400_000


def _rank(keys, texts):
    """Index of the first text containing any key, or None."""
    for i, t in enumerate(texts):
        if any(k.lower() in t.lower() for k in keys):
            return i
    return None


def run_scenario(sc, filler, embedder) -> dict:
    now_ms = int(datetime.now().timestamp() * 1000)
    with tempfile.TemporaryDirectory() as d:
        from strata.gateway.api import Strata
        st = Strata.open(db_path=str(Path(d) / "t.db"), embedder=embedder)
        for i, f in enumerate(filler):
            st.write_memory(f, created_at=now_ms - (i + 1) * 30 * D)  # spread 1-12 months back,
            #                                                           # so filler never pollutes recent windows
        for ev in sc["events"]:
            ts = now_ms - ev["days_ago"] * D
            eid = vc.record_event(st, ev["utterance"], ts_ms=ts)
            for fact in ev["facts"]:
                res = st.write_memory(fact, created_at=ts)        # backdated fact
                fid = res.get("id") if isinstance(res, dict) else None
                if fid and eid:
                    st.link_source(fid, eid)
        vc.warm_index(st)

        q, tgt, dis = sc["query"], sc["target_keys"], sc["distractor_keys"]
        ctx = vc.select_memories(st, q, semantic=True)
        tr, dr = _rank(tgt, ctx), _rank(dis, ctx)
        # right window ranks ahead of the wrong one (or the wrong one is absent)
        order_ok = tr is not None and (dr is None or tr < dr)

        msgs = vc.build_messages([{"role": "user", "content": q}], ctx)
        try:
            reply = vc.llm_complete(msgs, LLM_CFG)
        except Exception as e:
            reply = f"(llm error: {e})"
        a_ok = any(k.lower() in reply.lower() for k in tgt) and \
            not any(k.lower() in reply.lower() for k in dis)
        st.close()
    return {"id": sc["id"], "kind": sc["kind"], "query": q, "reply": reply,
            "t_rank": tr, "d_rank": dr, "order_ok": order_ok, "answer_ok": a_ok}


def main():
    data = json.loads((Path(__file__).parent / "temporal_scenarios.json").read_text())
    embedder = vc.make_embedder()
    if embedder is None:
        print("✗ needs Ollama + nomic-embed-text"); return 1
    mode = "TIME-AWARE" if vc.TEMPORAL else "BASELINE (semantic only)"
    print(f"Temporal recall benchmark · {mode} · model={LLM_CFG['ollama_model']}\n")
    rows = [run_scenario(sc, data["filler"], embedder) for sc in data["scenarios"]]
    for r in rows:
        print(f"── {r['id']}  ({r['kind']})")
        print(f"   query : {r['query']!r}")
        print(f"   reply : {r['reply'][:130]!r}")
        print(f"   right-window rank {r['t_rank']}  vs wrong-window rank {r['d_rank']}  "
              f"→ order {'OK' if r['order_ok'] else 'WRONG'} · answer "
              f"{'OK' if r['answer_ok'] else 'WRONG'}\n")
    n = len(rows)
    print("═══ RESULT ═══")
    print(f"  Right window ranked first : {sum(r['order_ok'] for r in rows)}/{n}")
    print(f"  Answer on the right event : {sum(r['answer_ok'] for r in rows)}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
