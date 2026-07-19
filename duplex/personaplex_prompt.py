"""Experiment 1 bridge: Strata memory -> PersonaPlex role prompt.

PersonaPlex (full-duplex speech-to-speech) has no per-turn prompt assembly;
its only text conditioning is a role prompt prefilled at session start. This
script builds that role prompt from the user's REAL local data — profile,
standing rules (L4), and memories — and optionally launches the MLX port's
web mode with it, so we can measure how well session-start injection carries
the memory experience before investing in the re-prefill design
(docs/DUPLEX-PROTOTYPE.md).

Read-only by contract: lists records from the Strata DB and never writes,
supersedes, or deletes anything, and never mutates settings or profile.
(Strata.open will create an empty DB file if none exists yet — same as the
server's own startup.) Safe against a real store; VOICE_DB works here
exactly as it does for the server if you want scratch data instead.

Run (on the Mac, from the repo root):
    python duplex/personaplex_prompt.py                # print the prompt
    python duplex/personaplex_prompt.py --launch       # print, then start
        personaplex-mlx web mode (http://localhost:8998) with the prompt

Requires `pip install personaplex-mlx` (github.com/mu-hashmi/personaplex-mlx)
and an HF_TOKEN with access to nvidia/personaplex-7b-v1 for --launch; the
prompt itself builds with no extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
import voicechat as vc  # noqa: E402  (list_memories / list_rules / DB_PATH)

STORE = Path(vc.DB_PATH).parent / "voicechat"
PROFILE_FILE = STORE / "profile.json"
SETTINGS_FILE = STORE / "settings.json"

# PersonaPlex prompts are short second-person role descriptions ("You are...").
# The Moshi-style context the prompt prefills into is small, and long prompts
# eat into conversation headroom — so the whole prompt gets a character budget.
# Rules and profile are kept whole; memories are trimmed newest-first-kept.
DEFAULT_BUDGET = int(os.environ.get("DUPLEX_PROMPT_CHARS", "2000"))


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _profile_lines(profile: dict) -> list[str]:
    labels = {"name": "Their full name is",
              "preferred_name": "They prefer to be called",
              "location": "They live in",
              "gender": "Their gender is"}
    return [f"{labels[k]} {profile[k]}." for k in labels if profile.get(k)]


def build_role_prompt(strata, budget: int = DEFAULT_BUDGET) -> str:
    settings = _read_json(SETTINGS_FILE, {})
    profile = _read_json(PROFILE_FILE, {})
    name = settings.get("assistant_name") or vc.ASSISTANT_NAME_DEFAULT
    user = profile.get("preferred_name") or profile.get("name") or "the user"

    head = (f"You are {name}, {user}'s personal voice assistant. You are warm, "
            "natural, and brief — one or two spoken sentences at a time, never "
            "lists read aloud. You have talked with them before and remember "
            "what they've told you; bring facts up only when relevant.")

    parts = [head]
    prof = _profile_lines(profile)
    if prof:
        parts.append(" ".join(prof))

    rules = [r["text"] for r in vc.list_rules(strata)]
    if rules:
        parts.append("Rules they have set, which you always follow: "
                     + " ".join(f"{t.rstrip('.')}." for t in rules))

    mems = [m["text"] for m in vc.list_memories(strata)]
    if mems:
        # Fit newest memories into whatever budget remains after the fixed parts.
        fixed = "\n\n".join(parts) + "\n\nThings you know about them:\n"
        room = budget - len(fixed)
        kept: list[str] = []
        for text in reversed(mems):                    # newest first
            line = f"- {text}\n"
            if room - len(line) < 0:
                break
            room -= len(line)
            kept.append(line)
        if kept:
            parts.append("Things you know about them:\n"
                         + "".join(reversed(kept)).rstrip())
        dropped = len(mems) - len(kept)
        if dropped:
            print(f"[bridge] budget {budget} chars: kept {len(kept)}/{len(mems)} "
                  f"memories (dropped the {dropped} oldest)", file=sys.stderr)

    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--launch", action="store_true",
                    help="after printing, exec personaplex-mlx realtime web mode")
    ap.add_argument("--voice", default="NATF2",
                    help="PersonaPlex voice id (NATF0-3, NATM0-3, VARF0-4, VARM0-4)")
    ap.add_argument("-q", "--quant", default="4",
                    help="quantization bits for the MLX port (default 4)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help=f"max prompt characters (default {DEFAULT_BUDGET})")
    ap.add_argument("--bare", action="store_true",
                    help="skip memory injection (control condition for the A/B)")
    args = ap.parse_args()

    from strata.gateway.api import Strata
    strata = Strata.open(db_path=vc.DB_PATH)   # default embedder; reads only

    if args.bare:
        prompt = "You enjoy having a good conversation."
    else:
        prompt = build_role_prompt(strata, budget=args.budget)

    print("=" * 72, file=sys.stderr)
    print(prompt)
    print("=" * 72, file=sys.stderr)
    print(f"[bridge] {len(prompt)} chars", file=sys.stderr)

    if args.launch:
        cmd = [sys.executable, "-m", "personaplex_mlx.local_web",
               "-q", str(args.quant), "--voice", args.voice,
               "--text-prompt", prompt]
        print(f"[bridge] launching: personaplex_mlx.local_web "
              f"(voice={args.voice}, q={args.quant}) — http://localhost:8998",
              file=sys.stderr)
        os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
