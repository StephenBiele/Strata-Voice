# Data safety contract

The whole point of Strata Voice is that **the user's memory is theirs**. Losing
it — even a slice of it, even once — breaks the product's one promise. Every
change to this codebase follows the rules below, and `tests/data_safety_check.py`
enforces the ones a machine can check.

## What counts as user data

Everything under `~/.vui/`, plus the API key in the macOS Keychain:

| Data | Where | Notes |
| :--- | :--- | :--- |
| Memories, timeline, recaps | `~/.vui/strata_memory.db` (+ `-wal`/`-shm`) | the crown jewels |
| Conversations | `~/.vui/voicechat/sessions.json` | rewritten every turn |
| Profile | `~/.vui/voicechat/profile.json` | |
| Settings | `~/.vui/voicechat/settings.json` | |
| Reference files | `~/.vui/voicechat/docs.json` + uploads | |
| Custom cloned voices | `~/.vui/voicechat/voices/` | user recordings |
| API key | macOS Keychain | never on disk |

Model caches (`~/.cache/huggingface`, Ollama models) are re-downloadable and
are NOT user data — but deleting them still requires a prompt (uninstall).

## The rules

1. **Nothing deletes user data except an explicit user action.** No background
   job, migration, update, reinstall, or error handler may delete or overwrite
   memories, sessions, profile, files, or voices. LLM output may *propose*
   deletions; only a user click applies them.
2. **Bulk changes are propose-then-approve.** Any operation that could touch
   many records at once (memory smoothing) previews its changes and applies
   only what the user approved — never a blind rewrite. Approved changes
   re-verify the record is unchanged since preview (stale guard) and skip
   rather than guess.
3. **All JSON writes are atomic** — `_write_json` only (serialize → temp file →
   `os.replace`). Never `open(path, "w")` / `write_text` directly on a user
   data file: a crash mid-write must leave the old file intact, not garbage.
4. **The updater never touches `~/.vui`** and applies only fast-forward pulls
   (`--ff-only`): it can never rewrite history, drop local edits, or run
   `git clean`/`reset --hard`.
5. **Install and reinstall never touch `~/.vui`.** Running `install.sh` over an
   existing setup preserves every byte of user data.
6. **Uninstall prompts before every destructive step**, and asks about `~/.vui`
   separately (twice). No `rm` of user data without a typed yes.
7. **Incognito writes nothing.** A private conversation leaves no session, no
   events, no memories.
8. **Ephemeral means ephemeral.** Web results live in RAM with a TTL and are
   never written to disk, transcripts, or memory.
9. **Forgetting is deterministic** — deletion never depends on an LLM's
   judgment, only on rule-based matching of what the user asked to forget.
10. **Schema/format migrations must be backwards-safe**: read the old format,
    write the new one atomically, and never drop fields you don't understand.

## Review checklist for every change

Before committing anything, ask:

- Does this code path write to or delete anything under `~/.vui`? If yes:
  is the write atomic, and is any deletion behind an explicit user action?
- Does a new endpoint mutate stored data? Does it need a preview/approve split?
- Could this run concurrently with another writer (background threads)? Is the
  write still whole-file-consistent?
- Does the updater/installer/uninstaller behavior change? Re-read rules 4–6.
- Run the data-safety tests: `./tests/run.sh` — all green before commit.

## Testing

`tests/data_safety_check.py` runs against a **throwaway server on a throwaway
database** (never the real `~/.vui`) and verifies, among others: atomic-write
crash behavior, settings-merge key preservation, the memory-smoothing stale
guard, that deletes require exact ids, that the updater is `--ff-only`, and
that the uninstaller prompts before touching user data. `./tests/run.sh` wraps
it. Add a test whenever you add a data-touching path.
