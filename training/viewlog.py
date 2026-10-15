"""Writes a run's metrics and agent snapshots in the layout the Go viewer reads.

  <dir>/metrics.jsonl            one JSON object per line: episode, loss, validation, done
  <dir>/snapshots/step_<N>.json  the network at step N, same format as agents/*.json
"""
import json
from pathlib import Path

from dqn import agent_doc


class RunLog:
    def __init__(self, directory, name: str):
        self.name = name
        self.dir = Path(directory)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        for old in (self.dir / "snapshots").glob("step_*.json"):
            old.unlink()
        self._file = open(self.dir / "metrics.jsonl", "w", buffering=1)  # line-buffered so the viewer sees lines promptly

    def _write(self, **record):
        self._file.write(json.dumps(record) + "\n")

    def episode(self, step, score, lines, pieces, epsilon):
        self._write(t="episode", step=step, score=score, lines=lines, pieces=pieces, epsilon=epsilon)

    def loss(self, step, value):
        self._write(t="loss", step=step, value=value)

    def validation(self, step, score, lines, pieces):
        self._write(t="validation", step=step, score=score, lines=lines, pieces=pieces)

    def snapshot(self, step, net):
        path = self.dir / "snapshots" / f"step_{step:09d}.json"
        tmp = path.with_suffix(".tmp")  # write then rename, so the viewer never reads a half-written file
        tmp.write_text(json.dumps(agent_doc(net, f"{self.name}@{step}")))
        tmp.replace(path)

    def done(self, step):
        self._write(t="done", step=step)
        self._file.close()
