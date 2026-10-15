# Phase 4: first DQN agent

**Agent:** `training/dqn.py`, trained by `training/train.py`. A 4-64-64-1 MLP scores the features of the board after a placement (aggregate height, holes, bumpiness, lines cleared; computed in Go, sent by the server as `afterstates`). The action is the placement with the highest value; the TD target is `r + gamma * max V_target(next afterstates)`. Replay buffer 50k, hard target update every 1000 steps, epsilon 1.0 -> 0.01 over 50k steps, Adam 1e-3, batch 128, gamma 0.99, 100k steps, one seed (0). Training episodes are cut at 1000 pieces.

Rewards are the line-clear score divided by 100. The only difference between the two runs is the shaping:

- `raw`: score reward only.
- `shaped`: adds the potential-based term `gamma * phi(s') - phi(s)` with `phi = -(0.5 * holes + 0.05 * aggregate height)` (`phi = 0` when the game ends).

```
go run ./cmd/server
cd training
python train.py --name raw --reward raw
python train.py --name shaped --reward shaped
python evaluate.py checkpoints/raw_best.pt
tensorboard --logdir runs
```

Seeds: training uses 1,000,000 and up, validation (every 10k steps, 10 greedy games, 500-piece cap, used to pick the `_best` checkpoint) uses 5001-5010, and the final numbers use the baseline seeds 1-100 that the agent never trained on.

## Results

Seeds 1-100, 1000-piece cap, greedy policy (same setting as `docs/BASELINE.md`):

| Agent | Avg score | Avg lines | Avg pieces |
|---|---|---|---|
| Random placement | 2.0 | 0.0 | 22.0 |
| Heuristic | 42591.0 | 390.9 | 988.6 |
| DQN raw (best checkpoint) | 58539.0 | 394.5 | 1000.0 |
| DQN shaped (best checkpoint) | 57692.0 | 392.0 | 993.5 |

Final checkpoints give the same picture (raw 58841.0, shaped identical to its best).

## What the numbers do and do not say

- Both agents clearly beat the random agent and match or beat the heuristic. The heuristic's lines and survival are already near the cap (1000 pieces fit about 400 lines), so the difference shows up in score. Likely cause (not checked): the DQN clears lines in bigger groups, which the score rewards and the heuristic's weights do not.
- The cap hides survival differences. Compare agents on score; use a higher cap (`evaluate.py --max-pieces`) before claiming one survives longer.
- One seed per run and 100 games. The raw-vs-shaped gap (58.5k vs 57.7k) is within what I would expect from noise; do not read it as shaping hurting.
- The learning curves differ though. Raw reached the 500-piece validation cap by 10k steps and kept improving its score. Shaped started well, then collapsed at 20k-60k steps (validation score 0, about 20 pieces per game) and recovered only after epsilon reached its floor at 50k. Possible cause: the hole penalty rewards stacking low-risk placements early while the value estimates are still wrong, but I have not tested that.
