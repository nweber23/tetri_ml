# Go <-> Python bridge

`proto/tetris/v1/tetris.proto` defines one bidirectional stream, `Play`. Each stream is one game session: the client sends `Reset(seed)` and then `Placement`s, and every request gets a `Response` with the state, the reward and the valid placements. The server listens on loopback only and has no authentication.

```
go run ./cmd/server                       # 127.0.0.1:50051
cd training
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python run_agent.py --games 100           # random agent
python run_agent.py --envs 8 --steps 40000  # parallel envs
python -m unittest                        # builds and starts its own server
../proto/generate.sh                      # regenerate the Go and Python stubs
```

`Tetris-v0` (`training/tetris_env`) is placement-based: action `rot * 12 + (x + 2)`, `info["action_mask"]` marks valid actions, an invalid action raises `ValueError`.

## Full-game stream: `PlayMoves` and `Tetris-v1`

`PlayMoves` is a second stream of the same shape for the agent that uses the whole game (slides, tucks, spins, T-spins, hold; see [MOVES.md](MOVES.md)). The client sends `Reset(seed)` and then the index of one of the listed moves. Each response lists every reachable way to lock the piece: whether it uses hold, whether it is a T-spin, lines cleared, reward and the 218-byte network input for the state after the move (`agent.MoveFeatures`, the one definition both languages use). The list is cut at `server.MaxMoves = 128` without a warning; in play it has about 43 entries and at most 70 were seen.

`Tetris-v1` (`MoveEnv`) takes an index into that list. `info` carries `action_mask`, `features` (128 x 218, uint8), `rewards`, `bait` (zero-line T-spins, which pay 400 without clearing anything), `score`, `lines`, `pieces`, `tspins` and `holds`.

A hand-built linear network with the heuristic weights, run through this path in Python, survives the 500-piece cap just as `heuristic-moves` does in Go. That checks the features, the move list and the move choice end to end, but unlike the placement agent it is not an exact game-by-game comparison between the two sides.

## Verification

Random agents never clear lines in ~20 pieces, so score and lines cannot tell Go and Python apart. The check is therefore a deterministic `cycle` agent (placement index `7 * pieces mod count`) run on both sides:

```
go run ./cmd/baseline -agent cycle -games 200 -v
python run_agent.py --agent cycle --games 200 -v
```

All 200 per-game lines (seed, score, lines, pieces) were identical. Reward values themselves are covered by the Go engine tests.

## Throughput

Random agent, async vector env, one server on the same machine (Windows, 32-bit Go build): 4 envs ~5200 steps/s, 8 envs ~7100 steps/s. A step is one piece placement.
