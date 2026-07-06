#!/usr/bin/env python3
"""Consolidation benchmark — does the smoothing pass merge fragments of the same
fact WITHOUT merging genuinely distinct ones?

polish_memory_store is a pure function (memory list -> proposed changes), so this
needs no store or embedder — just the LLM. We plant a memory list with a known
fragment group, a distinct-but-similar pair (must NOT merge), and clean facts,
apply the proposed changes deterministically, and check the result by keyword.

Needs Ollama. Usage:  .venv/bin/python tests/consolidation_benchmark.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import voicechat as vc  # noqa: E402

CFG = {"backend": "ollama", "ollama_model": os.environ.get("BENCH_MODEL", "qwen3.5:4b"),
       "temperature": 0.0}

SCENARIOS = [
    {
        "id": "merge-interview-fragments",
        "memories": ["Has an interview Tuesday", "Interviewing at Acme",
                     "Lives in Arvada, Colorado", "Has a dog named Molly"],
        # the two fragments should collapse into ONE memory holding both anchors
        "want_together": [["tuesday", "acme"]],
        "want_separate": [],
        "keep": ["arvada", "molly"],
    },
    {
        "id": "keep-distinct-interviews",
        "memories": ["Has an interview Tuesday with Acme for a backend role",
                     "Has an interview Thursday with Globex for a data role",
                     "Works as a software engineer"],
        # two DIFFERENT interviews must stay two memories
        "want_together": [],
        "want_separate": [["acme", "globex"]],
        "keep": ["engineer"],
    },
]


def apply_changes(memories, changes):
    """Simulate the approve-all result: delete drops, rewrite replaces."""
    items = [{"id": i, "text": t} for i, t in enumerate(memories)]
    idmap = {i: it for i, it in enumerate(items)}
    # polish returns ids matching the input order (memories[idx]["id"]); here id==index
    drop = set()
    for ch in changes:
        cid = ch.get("id")
        if ch.get("action") == "delete":
            drop.add(cid)
        elif ch.get("action") == "rewrite" and idmap.get(cid) is not None:
            idmap[cid]["text"] = ch.get("text", idmap[cid]["text"])
    return [it["text"] for it in items if it["id"] not in drop]


def run(sc):
    mems = [{"id": i, "text": t} for i, t in enumerate(sc["memories"])]
    changes = vc.polish_memory_store(mems, CFG)
    final = apply_changes(sc["memories"], changes)
    blob = " || ".join(final).lower()

    ok = True
    notes = []
    # fragments now live in ONE memory (both anchors in the same line)
    for grp in sc["want_together"]:
        hit = any(all(k in line.lower() for k in grp) for line in final)
        ok &= hit
        notes.append(("merged " + "+".join(grp)) if hit else ("✗ NOT merged " + "+".join(grp)))
    # distinct anchors stay in SEPARATE memories
    for grp in sc["want_separate"]:
        one_each = all(sum(k in line.lower() for line in final) >= 1 for k in grp) and \
            not any(all(k in line.lower() for k in grp) for line in final)
        ok &= one_each
        notes.append(("kept separate " + "+".join(grp)) if one_each
                     else ("✗ WRONGLY merged " + "+".join(grp)))
    for k in sc["keep"]:
        present = k in blob
        ok &= present
        if not present:
            notes.append(f"✗ lost {k!r}")
    return sc["id"], ok, final, notes


def main():
    print(f"Consolidation benchmark · {len(SCENARIOS)} scenarios · model={CFG['ollama_model']}\n")
    passed = 0
    for sc in SCENARIOS:
        sid, ok, final, notes = run(sc)
        passed += ok
        print(f"── {sid}  {'✓' if ok else '✗'}")
        for t in final:
            print(f"     -> {t!r}")
        print(f"     {'; '.join(notes)}\n")
    print(f"═══ {passed}/{len(SCENARIOS)} scenarios correct ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
