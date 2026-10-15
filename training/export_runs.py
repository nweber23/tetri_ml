"""Converts finished TensorBoard runs into the viewer's layout (see viewlog.py).

  python export_runs.py raw dqn_s1 ddqn_s0      # reads runs/<name> and checkpoints/<name>.pt, writes ../viewdata/<name>

Older runs only kept their final network, so they get a single snapshot.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dqn import ValueNet
from viewlog import RunLog

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+")
p.add_argument("--viewdata", default="../viewdata")
args = p.parse_args()

for name in args.names:
    events = EventAccumulator(f"runs/{name}", size_guidance={"scalars": 0})  # 0 = keep every point
    events.Reload()
    scalars = {tag: {e.step: e.value for e in events.Scalars(tag)} for tag in events.Tags()["scalars"]}

    log = RunLog(Path(args.viewdata) / name, name)
    records = []
    for step, score in scalars["episode/score"].items():
        records.append((step, {"t": "episode", "step": step, "score": score, "lines": scalars["episode/lines"][step],
                               "pieces": scalars["episode/pieces"][step], "epsilon": scalars["train/epsilon"][step]}))
    for step, value in scalars["train/loss"].items():
        records.append((step, {"t": "loss", "step": step, "value": value}))
    for step, score in scalars["validation/score"].items():
        records.append((step, {"t": "validation", "step": step, "score": score,
                               "lines": scalars["validation/lines"][step], "pieces": scalars["validation/pieces"][step]}))
    last = max(step for step, _ in records)
    for _, record in sorted(records, key=lambda r: r[0]):
        log._file.write(json.dumps(record) + "\n")

    net = ValueNet()
    net.load_state_dict(torch.load(f"checkpoints/{name}.pt"))
    log.snapshot(last, net)
    log.done(last)
    print(f"{name}: {len(records)} records, snapshot at step {last}")
