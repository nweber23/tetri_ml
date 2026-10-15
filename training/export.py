"""Exports trained agents for the Go viewer: weights, validation curve and a few test vectors.

  python export.py raw ddqn_s0        # reads checkpoints/<name>_best.pt and runs/<name>, writes ../agents/<name>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dqn import ValueNet, agent_doc

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+")
p.add_argument("--out", default="../agents")
args = p.parse_args()

out = Path(args.out)
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
for name in args.names:
    net = ValueNet()
    net.load_state_dict(torch.load(f"checkpoints/{name}_best.pt"))
    doc = agent_doc(net, name)

    events = EventAccumulator(f"runs/{name}")
    events.Reload()
    curve = [{"step": e.step, "score": e.value} for e in events.Scalars("validation/score")]

    cases = (rng.random((8, 4)) * [60, 8, 20, 4]).astype(np.float32)
    with torch.no_grad():
        values = net(torch.from_numpy(cases)).tolist()
    check = [{"features": c.tolist(), "value": v} for c, v in zip(cases, values)]

    doc.update(curve=curve, check=check)
    (out / f"{name}.json").write_text(json.dumps(doc))
    print(f"wrote {out / name}.json ({len(curve)} curve points)")
