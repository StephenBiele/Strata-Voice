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
- **Memory quality** has its own benchmark: `.venv/bin/python tests/memory_benchmark.py`
  (needs Ollama) plants close-in-time events, fires vague queries, and scores
  recall + collision deterministically. Run it before/after any change to
  extraction, recall, ranking, or the memory prompts — it's the North Star for
  disambiguation. Scenarios live in `tests/benchmark_scenarios.json`.

## Conventions

- Verify changes against a real running instance (throwaway server on a spare
  port with `VOICE_DB` pointed at scratch space), not just by reading code.
- Frontend and server are cache-sensitive: the page is served no-store; a
  server restart is needed for `server.py`/`voicechat.py` changes, refresh for
  `index.html`.
- Commit locally as work lands; **never `git push` unless the user says to.**
- Settings copy is plain language — write for a person, not a spec sheet.
