# Product

## Register

product

## Users

One person and, soon, their friends: privacy-minded Mac owners who want a voice
companion that lives entirely on their machine. They're mid-conversation when they
use it — cooking, debugging, walking past the desk — often not looking at the screen
at all. The UI's job is to be legible at a glance and invisible the rest of the time.

## Product Purpose

A local-first voice-to-voice companion that actually remembers you. Conversations
accumulate into owned memory (facts, timeline, recaps) via the Strata Memory engine;
speech never leaves the device. Success: talking to it feels like talking, not
operating software — and what it remembers feels like what a person would remember.

## Brand Personality

Calm, trustworthy, alive. A quiet companion — warmth comes from motion, voice, and
plain language, never from decoration. States (listening, thinking, speaking, muted)
are communicated by the orb's behavior, in the way a person's face communicates
attention.

## Anti-references

- Corporate SaaS chrome: dashboards, dense toolbars, enterprise sterility.
- Gadgety voice-assistant clichés: waveforms everywhere, sci-fi glow, robot/Jarvis
  styling, cyan-on-black "AI" aesthetics.
- Verbatim-transcript energy in copy: filler, jargon, em-dash-heavy prose. Speak
  plainly; every label should survive being read aloud.

## Design Principles

1. **The conversation is the interface.** Controls exist only for what voice can't
   do (mute, end, type). Everything else is the orb telling you what's happening.
2. **One mode at a time.** Live means "just talk." Muted means "it cannot hear me."
   Manual turns means "it waits for my release." Never show controls that imply two
   modes at once.
3. **Trust is visible.** Mic state is never ambiguous: the OS indicator, the mute
   button, and the orb always agree.
4. **Same vocabulary everywhere.** One button shape per role, one icon style, one
   sizing scale. A control that looks different from its siblings is a bug.
5. **Memory like a person's.** Gist over transcript, spoken-friendly wording,
   nothing stored that a friend wouldn't remember.

## Accessibility & Inclusion

- Every interactive element keyboard-reachable with visible focus (`:focus-visible`
  ring exists — keep it).
- ARIA roles/labels on all icon-only controls; states via `aria-pressed`/
  `aria-checked`; live regions for status.
- Touch targets ≥44px on coarse pointers.
- `prefers-reduced-motion` honored on all animation, including the orb.
- Contrast ≥4.5:1 for text (subtext was recently darkened for exactly this).
