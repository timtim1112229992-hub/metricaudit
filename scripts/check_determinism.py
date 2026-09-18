#!/usr/bin/env python
"""G7: re-execute from the fixed snapshot and confirm every value returns.

Run in a fresh process so that nothing cached in the first run can supply an
answer to the second. The comparison is over a digest of everything the audit
reports, which catches a changed figure anywhere rather than only in the places
someone thought to check.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHILD = """
import json, sys
from pathlib import Path
sys.path.insert(0, r"{src}")
from metricaudit.ingest import load
from metricaudit.pipeline import run, _result_digest
results = run(load(), output_dir=Path(r"{out}"), publish_provenance=False)
print(json.dumps({{"digest": _result_digest(results),
                   "source": results["corpus"]["source"]}}))
"""


def once(tmp: Path) -> dict:
    code = CHILD.format(src=ROOT / "src", out=tmp)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=ROOT)
    if done.returncode != 0:
        raise RuntimeError(f"re-execution failed:\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        first = once(Path(a))
        second = once(Path(b))

    print(f"source:        {first['source']}")
    print(f"first digest:  {first['digest']}")
    print(f"second digest: {second['digest']}")

    if first["digest"] != second["digest"]:
        print("\nG7 FAILS: two clean executions did not agree. Something in the "
              "pipeline depends on state that is not the fixed snapshot.")
        return 1

    held = ROOT / "outputs" / "results.json"
    if held.exists():
        recorded = json.loads(held.read_text(encoding="utf-8"))
        stored = recorded.get("gates", {}).get(
            "G7_clean_re_execution_reproduces", {}).get("result_digest")
        if stored and stored != first["digest"]:
            print(f"\nG7 FAILS: the stored results carry digest {stored}, which "
                  f"a clean re-execution does not reproduce. The outputs are "
                  f"stale, or the code changed after they were written.")
            return 1
        if stored:
            print("\nthe stored results carry the same digest")

    print("\nG7 passes: a clean re-execution reproduces every reported value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
