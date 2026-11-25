"""The synthetic end-to-end cases of tools/run_synth.py as unit tests (`lls test`).

LLS_CASES picks them: empty for the quick set, `all`, or a comma list (pal,tempo,...).
Output goes to test_out/ at the repo root.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_synth  # noqa: E402

QUICK = ["pal", "tempo", "drop_pal"]
_sel = os.environ.get("LLS_CASES", "").strip()
CASES = list(run_synth.CASES) if _sel == "all" else [c for c in _sel.split(",") if c] or QUICK


class Synth(unittest.TestCase):
    pass


def _case(name):
    def test(self):
        self.assertIn(name, run_synth.CASES, "unknown case")
        self.assertTrue(run_synth.run_case(os.path.join(ROOT, "test_out"), name), f"{name} failed")
    return test


for _c in CASES:
    setattr(Synth, f"test_{_c}", _case(_c))


if __name__ == "__main__":
    unittest.main()
