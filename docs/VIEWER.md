# Viewer

Ebitengine client. It links the engine directly and does not need the gRPC server.

```
go run ./cmd/viewer                          # plays the exported agents, agent mode
go run ./cmd/viewer -mode human              # play yourself
go run ./cmd/viewer -replay replays/x.json   # replay a saved game
go run ./cmd/viewer -screenshot out.png -screenshot-frame 240   # save a frame and exit
go run ./cmd/viewer -mode training -run show -snapshot 20000 -speed 2 -record out.gif -record-frames 600   # record a GIF and exit
```

`-snapshot N` picks the latest snapshot at or before step N, and `-record` captures every 6th frame (`-record-every`) with a fixed 216-colour palette, so GIFs are a bit coarser than the window.

| Key | Action |
|---|---|
| Tab | next mode: human, agent, replay (modes without agents or a replay are skipped) |
| N | new game, next seed |
| `[` `]` | previous or next agent checkpoint, same seed, so checkpoints can be compared on one game |
| P, `+`, `-` | pause, double or halve the agent and replay speed |
| S | save the current agent game to `replays/<agent>_<seed>.json` |
| Arrows, Up or X, Z, C, Space | human mode: move and soft drop, rotate, rotate back, hold, hard drop |

The heatmap shows the agent's value for every candidate placement of the current piece (row = rotation, column = x, green is high, the white outline is the placement it plays, also outlined on the board). The learning-curve panel shows the validation score recorded during that agent's training.

## Training mode

Tab to `training` shows how a run learned: the agent plays the same seed at each snapshot of the run, with charts on the right for the episode score (moving average), validation score of all runs with the shown snapshot marked, loss and epsilon.

| Key | Action |
|---|---|
| `[` `]` | step back and forward through the run's snapshots (early, mid, late training) |
| R | next run |
| F | follow live: jump to each new snapshot as training writes it |

`train.py` writes `viewdata/<run>/metrics.jsonl` and a snapshot every 5000 steps (`--snapshot-every`), including the untrained network at step 0. The viewer re-reads those files once a second, so the same view works for a run in progress ("LIVE") and a finished one. Earlier TensorBoard-only runs can be converted with `export_runs.py`, which gives them one snapshot (the final network).

```
cd training
python train.py --name show --double          # writes ../viewdata/show while it trains
python export_runs.py raw ddqn_s0 ddqn_s1     # older runs, from runs/ and checkpoints/
cd .. && go run ./cmd/viewer -mode training -run show
```

`viewdata/` is not committed. The `-runs` flag points the viewer at another directory.

## Agents

The viewer runs the network itself from JSON, no Python needed. Export trained checkpoints with:

```
cd training
python export.py raw ddqn_s0     # reads checkpoints/<name>_best.pt and runs/<name>, writes ../agents/<name>.json
```

`agents/raw.json` and `agents/ddqn_s0.json` are committed. Each file carries test vectors computed by PyTorch, and `go test ./agent` checks that the Go network reproduces them.

## Limits

- Replays store the seed and the placements, so only agent games can be saved. Human games use slides and rotations that a straight-drop placement cannot reproduce.
- Human mode has gravity on a 500 ms timer and a 500 ms lock delay that restarts on every move or rotation, at most 15 times per piece. The delay lives in the viewer session; `engine.Step` still locks at once, so training is unchanged.
- Human mode also has a hold piece (once per piece, `engine.Hold`), a ghost outline, and T-spin scoring (`engine.State.TSpin`): a T that ends on a rotation with 3 of its 4 box corners blocked scores 400/800/1200/1600 for 0/1/2/3 lines. There is no distinction between T-spins and mini T-spins, and no kick-based special case. The placement API the agents use never rotates in place, so agents never score T-spins and their numbers are unaffected.
- Human vs. agent (listed as optional in the plan) is not built.
- The window itself was checked through screenshots of the three modes, not by playing it by hand.
