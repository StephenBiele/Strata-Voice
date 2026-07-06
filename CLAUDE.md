# Strata Voice — working rules

Local-first voice assistant with owned memory. macOS/Apple Silicon (MLX).
Stack: `server.py` (stdlib HTTP, single-threaded on purpose — MLX's GPU stream
lives in the loading thread), `voicechat.py` (models/pipeline/memory),
`index.html` (entire frontend, one file), Strata Memory (SQLite) underneath.

## Data safety — read before touching anything that stores

**The product's one promise is that the user's memory is safe and theirs.**
The full contract lives in [docs/DATA-SAFETY.md](docs/DATA-SAFETY.md) — follow
it for any change that reads, writes, deletes, or migrates data under `~/.vui`.
The short version:

- Nothing deletes user data except an explicit user action. LLM output may
  propose deletions; only a user click applies them (preview → approve, with a
  stale guard).
- All JSON writes go through `_write_json` (atomic temp+rename). Never write a
  user data file directly.
- Updater = `git pull --ff-only` only. Installer/updater/reinstall never touch
  `~/.vui`. Uninstall prompts before every destructive step.
- Incognito writes nothing; web results never persist.
- **Before committing any data-touching change, run `./tests/run.sh`** (spins a
  throwaway server on a throwaway DB) and keep it green. Add a test when you
  add a data path.
- **Memory quality** has three deterministic benchmarks (all need Ollama; run the
  relevant one before/after any change to extraction, recall, ranking, or memory
  prompts). Each has an external `*_scenarios.json`:
  - `tests/memory_benchmark.py` — disambiguation (close-in-time events, vague
    queries): recall + collision.
  - `tests/write_benchmark.py` — storage decisions: durable-fact recall, junk
    kept out, distinct facts not clobbered by dedup.
  - `tests/temporal_benchmark.py` — time-scoped queries ("yesterday", "lately"):
    right-window ranking + answer. A/B the fix with `VOICE_TEMPORAL=0/1`.

## Conventions

- Verify changes against a real running instance (throwaway server on a spare
  port with `VOICE_DB` pointed at scratch space), not just by reading code.
- **Any visible UI change must be checked in BOTH light and dark mode** before
  it's considered done — the theme uses OS `prefers-color-scheme` plus a manual
  override (`applyTheme('light')` / `applyTheme('dark')`). A change that looks
  right in one mode can have wrong contrast, invisible borders, or washed-out
  surfaces in the other.
- Frontend and server are cache-sensitive: the page is served no-store; a
  server restart is needed for `server.py`/`voicechat.py` changes, refresh for
  `index.html`.
- Commit locally as work lands; **never `git push` unless the user says to.**
- Settings copy is plain language — write for a person, not a spec sheet.
