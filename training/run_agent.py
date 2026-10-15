"""Plays seeded games through the Gymnasium env with a random or cycle agent.

Start the server first: go run ./cmd/server

  python run_agent.py --games 100                 random agent, one env
  python run_agent.py --agent cycle -v            same output format as `go run ./cmd/baseline -agent cycle -v`
  python run_agent.py --envs 8 --steps 20000      random agent on 8 parallel envs, reports steps/s
"""
import argparse
import time

import gymnasium as gym
import numpy as np

import tetris_env  # noqa: F401  registers Tetris-v0


def pick(agent, mask, info, rng):
    valid = np.flatnonzero(mask)
    if len(valid) == 0:  # finished game: the vector env resets on the next step and ignores the action
        return 0
    if agent == "cycle":
        return valid[7 * info["pieces"] % len(valid)]
    return rng.choice(valid)


def run_games(args):
    env = gym.make("Tetris-v0", addr=args.addr)
    rng = np.random.default_rng(args.seed)
    totals = np.zeros(3)
    for i in range(args.games):
        seed = args.seed + i
        _, info = env.reset(seed=seed)
        done = False
        while not done and info["pieces"] < args.max_pieces:
            _, _, done, _, info = env.step(pick(args.agent, info["action_mask"], info, rng))
        if args.verbose:
            print(f"seed={seed} score={info['score']} lines={info['lines']} pieces={info['pieces']}")
        totals += [info["score"], info["lines"], info["pieces"]]
    env.close()
    avg = totals / args.games
    print(f"games={args.games} avg score={avg[0]:.1f} lines={avg[1]:.1f} pieces={avg[2]:.1f}")


def run_parallel(args):
    envs = gym.make_vec("Tetris-v0", num_envs=args.envs, vectorization_mode="async", addr=args.addr)
    rng = np.random.default_rng(args.seed)
    _, infos = envs.reset(seed=args.seed)
    start = time.perf_counter()
    for _ in range(args.steps // args.envs):
        actions = [pick(args.agent, m, {"pieces": 0}, rng) for m in infos["action_mask"]]
        _, _, _, _, infos = envs.step(np.array(actions))
    elapsed = time.perf_counter() - start
    envs.close()
    print(f"envs={args.envs} steps={args.steps} {args.steps / elapsed:.0f} steps/s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--addr", default="127.0.0.1:50051")
    p.add_argument("--agent", choices=["random", "cycle"], default="random")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-pieces", type=int, default=1000)
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    run_parallel(args) if args.envs > 1 else run_games(args)
