"""Pure assembly of a PersonaPlex role prompt from Strata data.

No I/O and no heavy imports — just string assembly — so it's shared by both
the experiment-1 CLI (duplex/personaplex_prompt.py) and the background cortex
(duplex/cortex.py), and is trivially unit-testable off a Mac.

PersonaPlex's only text conditioning is a short second-person role prompt
prefilled at session start. Long prompts eat conversation headroom, so the
whole thing gets a character budget; the fixed head/profile/rules are kept
whole and MEMORIES are trimmed newest-first-kept to fit.
"""

from __future__ import annotations

DEFAULT_BUDGET = 2000

PROFILE_LABELS = {
    "name": "Their full name is",
    "preferred_name": "They prefer to be called",
    "location": "They live in",
    "gender": "Their gender is",
}


def profile_lines(profile: dict) -> list[str]:
    return [f"{PROFILE_LABELS[k]} {profile[k]}." for k in PROFILE_LABELS
            if profile.get(k)]


def build(*, assistant_name: str, user_name: str,
          profile: list[str] | None = None,
          rules: list[str] | None = None,
          memories: list[str] | None = None,
          budget: int = DEFAULT_BUDGET) -> tuple[str, int]:
    """Return (prompt, dropped) where ``dropped`` is how many memories didn't
    fit the budget (oldest-dropped). ``profile`` is pre-formatted lines (see
    profile_lines); ``memories`` is newest-last (same order as list_memories),
    and the newest are kept first when trimming."""
    head = (f"You are {assistant_name}, {user_name}'s personal voice assistant. "
            "You are warm, natural, and brief — one or two spoken sentences at a "
            "time, never lists read aloud. You have talked with them before and "
            "remember what they've told you; bring facts up only when relevant.")
    parts = [head]
    if profile:
        parts.append(" ".join(profile))
    if rules:
        parts.append("Rules they have set, which you always follow: "
                     + " ".join(f"{t.rstrip('.')}." for t in rules))

    dropped = 0
    mems = memories or []
    if mems:
        fixed = "\n\n".join(parts) + "\n\nThings you know about them:\n"
        room = budget - len(fixed)
        kept: list[str] = []
        for text in reversed(mems):                    # newest first
            line = f"- {text}\n"
            if room - len(line) < 0:
                break
            room -= len(line)
            kept.append(line)
        dropped = len(mems) - len(kept)
        if kept:
            parts.append("Things you know about them:\n"
                         + "".join(reversed(kept)).rstrip())

    return "\n\n".join(parts), dropped
