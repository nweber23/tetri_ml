# Viewer

Ebitengine client. It links the engine directly and does not need the gRPC server.

```
go run ./cmd/viewer                          # plays the exported agents, agent mode
go run ./cmd/viewer -mode human              # play yourself
go run ./cmd/viewer -replay replays/x.json   # replay a saved game
go run ./cmd/viewer -screenshot out.png -screenshot-frame 240   # save a frame and exit
```

| Key | Action |
|---|---|
| Tab | next mode: human, agent, replay (modes without agents or a replay are skipped) |
| N | new game, next seed |
| `[` `]` | previous or next agent checkpoint, same seed, so checkpoints can be compared on one game |
| P, `+`, `-` | pause, double or halve the agent and replay speed |
| S | save the current agent game to `replays/<agent>_<seed>.json` |
| Arrows, Up or X, Z, Space | human mode: move, soft drop, rotate, rotate back, hard drop |

The heatmap shows the agent's value for every candidate placement of the current piece (row = rotation, column = x, green is high, the white outline is the placement it plays, also outlined on the board). The learning-curve panel shows the validation score recorded during that agent's training.

## Agents

The viewer runs the network itself from JSON, no Python needed. Export trained checkpoints with:

```
cd training
python export.py raw ddqn_s0     # reads checkpoints/<name>_best.pt and runs/<name>, writes ../agents/<name>.json
```

`agents/raw.json` and `agents/ddqn_s0.json` are committed. Each file carries test vectors computed by PyTorch, and `go test ./agent` checks that the Go network reproduces them.

## Limits

- Replays store the seed and the placements, so only agent games can be saved. Human games use slides and rotations that a straight-drop placement cannot reproduce.
- Human mode has gravity on a 500 ms timer and no lock delay, as in the engine.
- Human vs. agent (listed as optional in the plan) is not built.
- The window itself was checked through screenshots of the three modes, not by playing it by hand.
