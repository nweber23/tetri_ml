"""Greedy evaluation of a checkpoint on the baseline seeds; compare with docs/BASELINE.md.

  python evaluate.py checkpoints/raw_best.pt --games 100 --max-pieces 1000
"""
import argparse

import gymnasium as gym
import torch

import tetris_env  # noqa: F401
from dqn import ValueNet, evaluate

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--addr", default="127.0.0.1:50051")
p.add_argument("--games", type=int, default=100)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--max-pieces", type=int, default=1000)
args = p.parse_args()

net = ValueNet()
net.load_state_dict(torch.load(args.checkpoint))
env = gym.make("Tetris-v0", addr=args.addr)
r = evaluate(net, env, range(args.seed, args.seed + args.games), args.max_pieces)
print(f"games={args.games} seeds={args.seed}..{args.seed + args.games - 1} max-pieces={args.max_pieces}")
print(f"avg score={r['score']:.1f} lines={r['lines']:.1f} pieces={r['pieces']:.1f}")
