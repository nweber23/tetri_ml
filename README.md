# Tetris ML

A Tetris engine written in Go, a reinforcement learning agent trained with PyTorch, and a viewer that shows the agent playing and learning. I built it to learn RL by doing every layer myself.

![Late-training agent](docs/media/late.gif)

| Agent (seeds 1-100, 3000-piece cap) | Avg score | Avg lines |
|---|---|---|
| Random placement (1000-piece cap) | 2 | 0.0 |
| Hand-weighted heuristic | 119,880 | 1,098.8 |
| DQN (mean of 3 training seeds) | 172,673 | 1,179.8 |
| Double DQN (mean of 3 training seeds) | 173,305 | 1,179.4 |

### An agent that uses the whole game

Hold, tucks, spins and T-spins are in the engine too. A move agent picks among every reachable way to lock a piece, and it learns only when its network cannot see the board (seeds 1-100, 3000-piece cap, move-agent scores include T-spin points):

| Agent | Avg score | Reached the cap | Score / piece | T-spins / game |
|---|---|---|---|---|
| Scripted, all moves, same heuristic weights | 164,331 | 100/100 | 54.8 | 70 |
| Learned move agent, 2 training seeds | 241,493 and 236,973 | 66/100 and 54/100 | 97.4 and 102.7 | 233 and 254 |

![Move agent after 100,000 steps](docs/media/moves_late.gif)

It scores more but survives less often, and the T-spin points are partly a product of the reward. What failed first, and why, is in [docs/MOVES.md](docs/MOVES.md).

The write-up, including what did not work, is in [docs/BLOG.md](docs/BLOG.md). A follow-up measurement (clear sizes, score per piece, 10,000-piece cap) is in [docs/ANALYSIS.md](docs/ANALYSIS.md).

## How it fits together

```mermaid
flowchart LR
    E[engine<br/>Go, deterministic] --> S[server<br/>gRPC, one game per stream]
    S --> G[Gymnasium env<br/>Python]
    G --> T[DQN training<br/>PyTorch]
    T -->|metrics + snapshots| D[(viewdata/)]
    T -->|export.py| A[(agents/*.json)]
    E --> V[viewer<br/>Ebitengine]
    A --> V
    D --> V
```

- **Engine** (`engine/`): board, 7 tetrominoes, 7-bag, SRS rotation and kicks, gravity, lock, line clears. Seedable, so every game is reproducible. It has no rendering code.
- **Heuristic** (`heuristic/`): placement features (aggregate height, holes, bumpiness, lines) and a hand-weighted picker, used as the baseline.
- **Server** (`server/`, `proto/`): exposes the engine over gRPC. Each `Play` stream is one game and returns the valid placements together with the features of the board after each one.
- **Training** (`training/`): a Gymnasium environment on top of the server, a DQN that scores the board after each placement, and the logging the viewer reads.
- **Viewer** (`cmd/viewer`): a separate Ebitengine client for human play, agent playback, replays and watching training. It runs the exported networks itself and does not need the server.

Training never depends on rendering.

## Setup

Needs Go 1.25+ and Python 3.11. Tested on Windows 11 and Linux.

```bash
go test ./...

cd training
python -m venv .venv
# use .venv/bin/pip on Linux and macOS; install the CPU build of PyTorch first (the network is tiny)
.venv/Scripts/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.venv/Scripts/pip install -r requirements.txt
python -m unittest
```

The generated gRPC code is committed. To regenerate it after editing `proto/tetris/v1/tetris.proto`, run `proto/generate.sh` (it needs `protoc-gen-go` and `protoc-gen-go-grpc`, see the script).

## Run it

Baselines, no Python needed:

```bash
go run ./cmd/baseline -games 100 -max-pieces 3000   # heuristic
go run ./cmd/baseline -agent agents/raw.json -games 100 -max-pieces 3000   # an exported network
go run ./cmd/headless -seed 1                       # one random game
```

Train an agent (the server must be running):

```bash
go run ./cmd/server                                 # terminal 1, listens on 127.0.0.1:50051

cd training                                         # terminal 2
python train.py --name mine --double                # writes ../viewdata/mine while it trains
python evaluate.py checkpoints/mine_best.pt --max-pieces 3000
python train_moves.py --name moves --no-board       # the full-game agent (about an hour)
python export_moves.py moves --no-board --final     # then: go run ./cmd/baseline -agent agents/moves_final.json
tensorboard --logdir runs
```

Viewer:

```bash
go run ./cmd/viewer                                 # trained agents in agents/
go run ./cmd/viewer -mode human                     # play yourself
go run ./cmd/viewer -mode training -run mine        # watch a run, live or finished
```

Keys are listed in [docs/VIEWER.md](docs/VIEWER.md). Trained agents are included in `agents/`: two placement agents (`raw`, `ddqn_s0`) and three move agents (`moves_nb_s0`, `moves_nb_s0_final`, `moves_nb_s1`). To show your own, run `python export.py mine` (placement) or `export_moves.py` (moves) in `training/`.

## Docs

- [BLOG.md](docs/BLOG.md): the write-up
- [PLAN.md](docs/PLAN.md): the original plan
- [BASELINE.md](docs/BASELINE.md): heuristic baseline numbers
- [BRIDGE.md](docs/BRIDGE.md): the Go-Python bridge, how it was verified, throughput
- [DQN.md](docs/DQN.md) and [EXPERIMENTS.md](docs/EXPERIMENTS.md): training results
- [VIEWER.md](docs/VIEWER.md): viewer controls and limits
- [MOVES.md](docs/MOVES.md): the full-game agent (hold, tucks, spins, T-spins): what works and what failed

## Limits

- The trained agents in this README choose placements (a rotation and a column per piece), so they never tuck, spin or hold. An agent that uses the whole game is the open problem of [docs/MOVES.md](docs/MOVES.md): the search, the bridge and a scripted baseline work, a learned one does not yet.
- The training environments have no lock delay, and `Tetris-v0` applies one gravity row per step. The viewer's human mode adds hold, a lock delay, a ghost piece and a five-piece preview.
- Results come from 3 training seeds per variant at most, and evaluation is capped at 3000 pieces, which hides differences between the strongest agents.
