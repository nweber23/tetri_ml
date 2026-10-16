# Phase 5b: testing two guesses from the write-up

Two claims in `docs/DQN.md` and `docs/EXPERIMENTS.md` were marked "not tested": that the DQN scores higher because it clears lines in bigger groups, and that the 3000-piece cap hides survival differences. `cmd/baseline` now also takes `-agent agents/<name>.json` (it runs the exported network in Go, same code the viewer uses) and prints score per piece and the share of cleared lines by clear size. No new training; only the two exported agents are in the repo (`raw` = DQN seed 0, `ddqn_s0` = Double DQN seed 0).

```
go run ./cmd/baseline -agent agents/raw.json -games 100 -max-pieces 10000
```

Seeds 1-100. The heuristic and DQN seed 0 rows at the 3000 cap reproduce the PyTorch numbers in `docs/EXPERIMENTS.md` exactly (119880.0 and 173983.0), which also checks that the Go network matches the PyTorch one.

| Agent | Cap | Avg score | Score / piece | Hit cap | Singles | Doubles | Triples | Tetrises |
|---|---|---|---|---|---|---|---|---|
| Heuristic | 3000 | 119880 | 43.4 | 83/100 | 83.0% | 14.2% | 2.3% | 0.5% |
| DQN (seed 0) | 3000 | 173983 | 59.0 | 96/100 | 15.9% | 61.4% | 15.4% | 7.4% |
| Double DQN (seed 0) | 3000 | 173757 | 58.5 | 99/100 | 18.6% | 59.4% | 14.9% | 7.0% |
| Heuristic | 10000 | 318458 | 43.5 | 51/100 | 82.8% | 14.4% | 2.3% | 0.5% |
| DQN (seed 0) | 10000 | 562349 | 59.2 | 90/100 | 16.1% | 61.4% | 15.3% | 7.2% |
| Double DQN (seed 0) | 10000 | 554858 | 58.6 | 85/100 | 18.5% | 59.2% | 15.1% | 7.1% |

Clear-size columns are the share of all cleared lines that came from clears of that size.

## Reading

- **Bigger groups: confirmed.** The heuristic takes 83% of its lines as singles, the DQN 16%, with most of its lines in doubles (61%). Since a single is worth 100 per line and a double 150, a triple 167 and a tetris 200, this accounts for the score gap at equal lines.
- **Score per piece is stable across caps** (43.4 vs 43.5, 59.0 vs 59.2), so it is a cap-independent way to compare agents on score. It does not separate DQN from Double DQN either (59.2 vs 58.6, one seed each).
- **Survival: the cap was hiding a gap.** At 10000 pieces the heuristic hits the cap in 51 of 100 games, the DQN in 90 and the Double DQN in 85. The learned agents are clearly more robust than the heuristic. The 90 vs 85 is not a result: it is one seed per variant.
- **The worst games differ.** The heuristic's shortest game is 478 pieces, the DQN's 1359, the Double DQN's 268. One early death in a single game is enough to explain Double DQN's lower average at the high cap; that is an outlier, not a trend.
- **Not covered:** the other four training seeds are not exported, so this says nothing about the seed spread. Re-running those needs their checkpoints (`training/export.py`).
