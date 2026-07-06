#!/usr/bin/env python3
"""Proactive-nudge benchmark — does it surface a genuine upcoming commitment at
the start of a conversation, and stay quiet otherwise?

upcoming_nudge is a near-pure function (memory list -> a short heads-up or None),
so this needs no store — just the LLM. We build memory lists with known
mention-dates and check: fires on a real upcoming commitment (naming the right
thing), skips a past event, a stable fact, and an old/stale commitment. Scored by
keyword — no LLM judges.

Needs Ollama. Usage:  .venv/bin/python tests/proactive_benchmark.py
"""
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import voicechat as vc  # noqa: E402

CFG = {"backend": "ollama", "ollama_model": os.environ.get("BENCH_MODEL", "qwen3.5:4b"),
       "temperature": 0.0}
NOW = int(datetime.now().timestamp() * 1000)
D = 86_400_000


def mem(text, days_ago):
    return {"id": abs(hash(text)) % (10**9), "text": text, "t": NOW - days_ago * D}


SCENARIOS = [
    {"id": "upcoming-interview", "should_fire": True, "want": ["globex"], "avoid": ["dentist"],
     "memories": [mem("Has a job interview tomorrow with Globex", 0),
                  mem("Had a dentist appointment yesterday", 0),
                  mem("Enjoys hiking on weekends", 0),
                  mem("Lives in Arvada, Colorado", 0)]},
    {"id": "all-stable", "should_fire": False, "want": [], "avoid": [],
     "memories": [mem("Likes coffee", 1), mem("Works as a nurse", 2),
                  mem("Has a dog named Molly", 3)]},
    {"id": "old-commitment", "should_fire": False, "want": [], "avoid": [],
     # a commitment word, but mentioned 40 days ago → filtered out by recency
     "memories": [mem("Had an interview with Acme", 40),
                  mem("Lives in Denver", 30)]},
    {"id": "only-past", "should_fire": False, "want": [], "avoid": [],
     "memories": [mem("Went to the dentist yesterday for a cleaning", 0),
                  mem("Watched a movie last night", 0)]},
]


def main():
    print(f"Proactive-nudge benchmark · {len(SCENARIOS)} scenarios · model={CFG['ollama_model']}\n")
    ok = 0
    for sc in SCENARIOS:
        nudge = vc.upcoming_nudge(sc["memories"], CFG)
        fired = nudge is not None
        good = fired == sc["should_fire"]
        if fired and sc["should_fire"]:
            n = nudge.lower()
            good = all(w in n for w in sc["want"]) and not any(a in n for a in sc["avoid"])
        ok += good
        print(f"── {sc['id']}  {'✓' if good else '✗'}")
        print(f"     nudge: {nudge!r}  (expected {'fire' if sc['should_fire'] else 'quiet'})\n")
    print(f"═══ {ok}/{len(SCENARIOS)} correct ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
