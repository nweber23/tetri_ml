"""Exports trained move agents for the Go viewer and baseline: weights, validation curve and test vectors.

  python export_moves.py moves_s0     # reads checkpoints/<name>_best.pt and runs/<name>, writes ../agents/<name>.json
  python export_moves.py moves_nb_s0 --no-board     # for runs trained with --no-board
  python export_moves.py moves_nb_s0 --final        # the last network instead of the best-validation one (written as <name>_final.json)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dqn_moves import MOVE_FEATURES, MoveNet, agent_doc

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+")
p.add_argument("--out", default="../agents")
p.add_argument("--no-board", action="store_true", help="the run hid the board from the network (its board weights were never trained)")
p.add_argument("--board-scale", type=float, default=1.0, help="the run's --board-scale")
p.add_argument("--final", action="store_true")
args = p.parse_args()

out = Path(args.out)
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
for name in args.names:
    net = MoveNet(warm_start=False, board_scale=0.0 if args.no_board else args.board_scale)
    net.load_state_dict(torch.load(f"checkpoints/{name}.pt" if args.final else f"checkpoints/{name}_best.pt"))
    label = f"{name}_final" if args.final else name
    doc = agent_doc(net, label)

    events = EventAccumulator(f"runs/{name}")
    events.Reload()
    curve = [{"step": e.step, "score": e.value} for e in events.Scalars("validation/score")]

    cases = (rng.random((8, MOVE_FEATURES)) < 0.3).astype(np.float32)  # random boards and one-hots
    cases[:, :4] = rng.random((8, 4)) * [60, 8, 20, 4]
    with torch.no_grad():
        values = net(torch.from_numpy(cases)).tolist()
    doc.update(curve=curve, check=[{"features": c.tolist(), "value": v} for c, v in zip(cases, values)])
    (out / f"{label}.json").write_text(json.dumps(doc))
    print(f"wrote {out / label}.json ({len(curve)} curve points)")
