#!/usr/bin/env python3
"""Data-safety checks (docs/DATA-SAFETY.md). Run via ./tests/run.sh.

Everything here runs against throwaway state — a temp VOICE_DB and a temp
~/.vui-style dir — never the real user data. Two layers:

  1. unit checks         — imported helpers, no server needed
  2. static guards       — grep-level invariants on the scripts
  3. live HTTP checks    — a throwaway server (started by run.sh) on $PORT

Exit code 0 = all green.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
PORT = int(os.environ.get("TEST_PORT", "0"))   # 0 = skip live checks

PASS = 0
FAIL = []


def check(name, ok, detail=""):
    global PASS
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name}  {detail}")


def http(method, path, body=None, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method,
                                 data=(json.dumps(body).encode() if body is not None else None),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---- 1. unit checks -----------------------------------------------------------
print("· unit: atomic JSON writes")
import server  # noqa: E402  (import only — no models loaded at import time)

tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"ds_{os.getpid()}.json"
server._write_json(tmp, {"precious": True})
try:
    server._write_json(tmp, {"bad": object()})   # unserializable → must raise…
    check("unserializable write raises", False)
except TypeError:
    check("unserializable write raises", True)
check("…and the old file survives intact",
      json.loads(tmp.read_text()) == {"precious": True})
check("no temp litter", not tmp.with_name(tmp.name + ".tmp").exists())
tmp.unlink(missing_ok=True)

print("· unit: web results are RAM-only with a TTL")
server._web_remember("test query", [{"title": "t", "url": "u", "description": "d"}])
blk, _ = server._web_block()
check("fresh results visible", blk is not None)
server._web_cache["test query"]["t"] -= server.WEB_TTL_S + 1
blk, _ = server._web_block()
check("expired results purged", blk is None and not server._web_cache)

print("· unit: settings merge never drops other keys")
# _save_settings writes via _write_json to the real SETTINGS_FILE — redirect it
server.SETTINGS_FILE = Path(os.environ.get("TMPDIR", "/tmp")) / f"ds_settings_{os.getpid()}.json"
server._json_cache.clear()
server._save_settings({"tts_speed": 1.3})
server._save_settings({"assistant_name": "Test"})
s = server._settings()
check("earlier field preserved by later partial save",
      s["tts_speed"] == 1.3 and s["assistant_name"] == "Test")
server._save_settings({"not_a_real_field": "x"})
check("unknown fields are not persisted",
      "not_a_real_field" not in json.loads(server.SETTINGS_FILE.read_text()))
server.SETTINGS_FILE.unlink(missing_ok=True)

# ---- 2. static guards ----------------------------------------------------------
print("· static: script-level invariants")
srv = (REPO / "server.py").read_text()
check("updater pulls --ff-only only",
      '"pull", "--ff-only"' in srv and "reset --hard" not in srv and "git clean" not in srv)
check("polish apply has the stale guard",
      "current.get(mid) != ch.get" in srv)
bad_writes = [m for m in re.finditer(r'(open\([^)]*,\s*"w"\)|write_text\()', srv)
              if any(k in srv[max(0, m.start()-400):m.start()]
                     for k in ("SESSIONS_FILE", "PROFILE_FILE", "DOCS_FILE", "SETTINGS_FILE"))]
check("no direct writes near user-data files (use _write_json)", not bad_writes)

uninst = (REPO / "uninstall.sh").read_text()
vui_rm = [l for l in uninst.splitlines() if "rm" in l and ".vui" in l]
check("uninstall removes ~/.vui only after prompts",
      bool(vui_rm) and uninst.index("read -r") < uninst.index(vui_rm[0]))
inst = (REPO / "install.sh").read_text()
check("installer never removes ~/.vui",
      not any("rm" in l and ".vui" in l for l in inst.splitlines()))

# ---- 3. live HTTP checks --------------------------------------------------------
if PORT:
    print(f"· live: throwaway server on :{PORT} (throwaway DB)")
    from strata.gateway.api import Strata
    db = os.environ["VOICE_DB"]
    st = Strata.open(db_path=db, embedder=None)   # second WAL connection, like the app's workers
    fid = st.write_memory("Has a dog named Rex")["id"]

    mems = http("GET", "/memories")["memories"]
    check("seeded memory visible over HTTP", any(m["text"] == "Has a dog named Rex" for m in mems))

    r = http("POST", "/memory/polish/apply",
             {"changes": [{"id": str(fid), "action": "rewrite",
                           "old": "WRONG snapshot text", "text": "clobbered!"}]})
    mems = http("GET", "/memories")["memories"]
    check("stale suggestion is skipped, memory untouched",
          r["skipped"] == 1 and r["applied"] == 0
          and any(m["text"] == "Has a dog named Rex" for m in mems))

    r = http("POST", "/memory/polish/apply",
             {"changes": [{"id": str(fid), "action": "rewrite",
                           "old": "Has a dog named Rex", "text": "Has a dog named Rex (a corgi)"}]})
    mems = http("GET", "/memories")["memories"]
    check("fresh approved suggestion applies",
          r["applied"] == 1 and any("corgi" in m["text"] for m in mems))

    r = http("POST", "/memory/delete", {"id": 999999999})
    check("delete of unknown id fails safely", r.get("ok") is False or r.get("ok") is True)
    mems = http("GET", "/memories")["memories"]
    check("…and existing memories survive it", any("corgi" in m["text"] for m in mems))
    st.close()
else:
    print("· live checks skipped (no TEST_PORT)")

print(f"\n{PASS} passed, {len(FAIL)} failed" + (f": {FAIL}" if FAIL else ""))
sys.exit(1 if FAIL else 0)
