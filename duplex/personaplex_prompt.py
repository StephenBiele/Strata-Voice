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
from duplex import role_prompt  # noqa: E402

STORE = Path(vc.DB_PATH).parent / "voicechat"
PROFILE_FILE = STORE / "profile.json"
SETTINGS_FILE = STORE / "settings.json"

# PersonaPlex prompts are short second-person role descriptions ("You are...").
# The Moshi-style context the prompt prefills into is small, and long prompts
# eat into conversation headroom — so the whole prompt gets a character budget.
# Rules and profile are kept whole; memories are trimmed newest-first-kept.
DEFAULT_BUDGET = int(os.environ.get("DUPLEX_PROMPT_CHARS", str(role_prompt.DEFAULT_BUDGET)))


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def build_role_prompt(strata, budget: int = DEFAULT_BUDGET) -> str:
    """Session-start prompt from the WHOLE store — the experiment-1 condition.
    (The cortex builds from a recalled subset instead; both share role_prompt.)"""
    settings = _read_json(SETTINGS_FILE, {})
    profile = _read_json(PROFILE_FILE, {})
    prompt, dropped = role_prompt.build(
        assistant_name=settings.get("assistant_name") or vc.ASSISTANT_NAME_DEFAULT,
        user_name=profile.get("preferred_name") or profile.get("name") or "the user",
        profile=role_prompt.profile_lines(profile),
        rules=[r["text"] for r in vc.list_rules(strata)],
        memories=[m["text"] for m in vc.list_memories(strata)],
        budget=budget)
    if dropped:
        print(f"[bridge] budget {budget} chars: dropped the {dropped} oldest "
              "memories to fit", file=sys.stderr)
    return prompt


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
