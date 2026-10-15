"""Run from training/: python -m unittest. No server needed."""
import json
import tempfile
import unittest
from pathlib import Path

from dqn import ValueNet
from viewlog import RunLog


class TestRunLog(unittest.TestCase):
    def test_writes_metrics_and_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLog(tmp, "r")
            log.snapshot(0, ValueNet())
            log.episode(10, 100.0, 1, 30, 0.9)
            log.loss(100, 0.5)
            log.validation(100, 5.0, 1.0, 9.0)
            log.snapshot(5000, ValueNet())
            log.done(5000)

            kinds = [json.loads(line)["t"] for line in (Path(tmp) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(kinds, ["episode", "loss", "validation", "done"])
            names = sorted(p.name for p in (Path(tmp) / "snapshots").iterdir())
            self.assertEqual(names, ["step_000000000.json", "step_000005000.json"])
            doc = json.loads((Path(tmp) / "snapshots" / "step_000005000.json").read_text())
            self.assertEqual(doc["name"], "r@5000")
            self.assertEqual(len(doc["layers"]), 3)

    def test_new_run_clears_old_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = RunLog(tmp, "r")
            first.snapshot(9000, ValueNet())
            first.done(9000)
            RunLog(tmp, "r").done(0)
            self.assertEqual(list((Path(tmp) / "snapshots").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
