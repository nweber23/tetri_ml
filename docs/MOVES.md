# Phase 8: an agent that uses the whole game (hold, tucks, spins, T-spins)

The placement agents only choose a column and rotation for a straight drop. This phase gives the agent every way a player can lock a piece. Status: **a learned agent works, with three training seeds. It learns reliably when the network cannot see the board, and the board can only be added back if its inputs are scaled down 100 times.** The runs that failed are kept below so nobody repeats them.

![Move agent after 100,000 steps](media/moves_late.gif)

## What was built

- `engine.Moves()` (`engine/moves.go`): a BFS over keys (left, right, rotate both ways, soft drop) from the spawn position. Every grounded position it reaches is a candidate, deduplicated by the cells it fills and by whether it is a T-spin. With hold available it also searches from the piece a hold would bring in. Each `Move` carries its key sequence, the board after locking, lines cleared and reward. `Game.Do(move)` plays it.
- No gravity and no lock delay inside the search: the piece can be steered at will, as in high-level play. About 43 candidate moves per state on average, 70 at most in play; the server caps the list at 128 (`server.MaxMoves`).
- T-spin: the three-corner rule, only when the last key was a rotation. Scores 400/800/1200/1600 for 0/1/2/3 lines. No mini T-spins. A zero-line T-spin is worth 400, which is a lot (a double is 300).
- `PlayMoves` RPC and the `Tetris-v1` env (`training/tetris_env/env.py`, `MoveEnv`): the action is an index into the listed moves (see [BRIDGE.md](BRIDGE.md)).
- Network input (`agent.MoveFeatures`, 218 bytes, one source for Go and Python): aggregate height, holes, bumpiness, lines; the 20 visible rows as 0/1; one-hot of the held piece after the move; one-hot of the next piece (zero if unknown after a hold from an empty slot).
- `training/dqn_moves.py`, `train_moves.py`, `export_moves.py`: U(afterstate) excludes the move's own reward; a move is worth `reward/100 + 0.99 * U`; the TD target is the best such value over the next state's moves, or `-death_penalty` for a finished game.
- Viewer: move agents are steered key by key, so slides, spins and holds are visible. `go run ./cmd/baseline -agent <agent>` reports T-spins and holds per game for any move agent, exported or scripted.

## Results

Seeds 1-100, 3000-piece cap, greedy play. These seeds were never trained on. Move-agent scores include T-spin points, so they are not directly comparable with the placement scores; compare lines per piece and score per piece with that in mind.

| Agent | Avg score | Avg lines | Avg pieces | Hit cap | Score / piece | T-spins / game | Holds / game |
|---|---|---|---|---|---|---|---|
| Placement heuristic | 119,880 | 1,098.8 | 2,764 | 83/100 | 43.4 | | |
| Placement DQN (`raw`) | 173,983 | 1,173.3 | 2,948 | 96/100 | 59.0 | | |
| Scripted `heuristic-moves` | 164,331 | 1,198.5 | 3,000 | 100/100 | 54.8 | 70.2 | 874.3 |
| Learned move agent, seed 0, final (100k) | 241,493 | 984.9 | 2,481 | 66/100 | 97.4 | 233.4 | 730.4 |
| Learned move agent, seed 0, best validation (90k) | 206,816 | 812.6 | 2,057 | 42/100 | 100.5 | 222.8 | 795.9 |
| Learned move agent, seed 1 (best = final, 100k) | 236,973 | 912.9 | 2,308 | 54/100 | 102.7 | 253.9 | 931.6 |
| Learned move agent, seed 2, best validation (80k) | 285,432 | 1,157.0 | 2,905 | 94/100 | 98.3 | 312.7 | 1,204.7 |
| Learned move agent, seed 2, final (100k) | 104,800 | 406.1 | 1,046 | 8/100 | 100.2 | 105.7 | 415.3 |
| Board visible at scale 0.01, best validation (90k) | 69,820 | 333.4 | 875 | 1/100 | 79.8 | 80.8 | 363.9 |
| Board visible at scale 0.01, final (100k) | 121,679 | 573.9 | 1,474 | 21/100 | 82.5 | 140.1 | 578.6 |

What this shows and what it does not:

- **The board-hidden agents score more than the scripted agent that already uses everything, except when a late checkpoint collapses** (seed 0 final 241k, seed 1 237k, seed 2 best 285k, against 164k), with about 100 points per piece against 55. They do it with T-spins: 233, 254 and 313 per game, against 70. At 400 points or more each, that is at least 93,000 points of the 241,000 for seed 0. I did not split the score further.
- **They are less robust, and how much depends on the checkpoint.** The scripted agent reached the cap in every game; the learned ones in 66, 54 and 94 of 100 (and 8 for the collapsed seed 2 final). Their line rate is the same as everyone else's when they survive (0.40 lines per piece), so they are not clearing more lines, they are scoring more per line cleared and dying more often.
- **Clear sizes shifted too** (`go run ./cmd/baseline -agent agents/moves_nb_s0_final.json`): doubles are 48% of cleared lines for seed 0 (single 34%, triple 13%, tetris 4.5%), against 90% singles for the scripted agent.
- **Three training seeds, all learned.** In validation, seed 0 was at 496 of 500 pieces by 30k steps, seed 2 at 500 by 50k and seed 1 at 498.5 by 70k, so the speed varies.
- **Late training is unstable and the "best" checkpoint, picked on 10 validation games, is a noisy choice in both directions.** Seed 0: the pick (90k) scored 206,816 on seeds 1-100, the final network 241,493. Seed 2: the pick (80k) scored 285,432, the final network 104,800 and reached the cap in only 8 of 100 games. All three board-hidden runs held about 500 pieces in validation for a while, and seeds 0 and 2 dipped at 100k. Over the three seeds the means are 243,074 for the best-validation picks and 194,422 for the final networks; the spread between checkpoints of one run is as large as the gap to the scripted agent. I would use more validation games and a decaying learning rate before trusting a single network.
- **The board-hidden agents cannot see the board**, only height, holes, bumpiness, lines, the held piece and the next piece after each candidate move. It finds T-spins because the move list already contains them with their 400+ reward; the network never has to spot a T-slot, it only has to keep the stack in a shape where such moves appear. This is my reading, I did not test it.

## Training runs

Validation is 10 greedy games on seeds 5001-5010, 500-piece cap, every 10k steps. A game that ends after about 22 pieces is no better than random play (random play lasts about 25 pieces). For comparison, the placement DQN was already at the 500-piece cap at 10k steps. Columns: score / lines / pieces / T-spins / holds per game.

| Run | Setup | 10k | 20k | 30k |
|---|---|---|---|---|
| `moves_s0` | random init, no death penalty, lr 1e-3, board visible; stopped at ~35k | 440 / 0 / 19.9 / 1.1 / 4.0 | 160 / 0.4 / 23.8 / 0.3 / 8.3 | 630 / 0.3 / 23.3 / 1.5 / 8.8 |
| `moves_warm_s0` (first) | heuristic warm start, no death penalty; stopped at 20k | 160 / 0 / 13.4 / 0.4 / 5.1 | 120 / 0 / 16.7 / 0.3 / 5.1 | |
| `moves_warm_s0` (second) | heuristic warm start + death penalty 5; stopped at 20k | 920 / 0 / 22.5 / 2.3 / 6.1 | 490 / 0.1 / 21.5 / 1.2 / 7.1 | |
| `moves_cold_s0` | random init + death penalty 5; stopped at 20k | 1050 / 0.1 / 22.8 / 2.6 / 8.0 | 520 / 0 / 23.5 / 1.3 / 6.9 | |
| `moves_lowlr_s0` | warm + penalty 5, lr 1e-4, board visible; stopped at 10k | 1700 / 0.6 / 29.8 / 4.1 / 9.3 | | |
| `moves_bait_s0` | warm + penalty 5, zero-line T-spins paid nothing in training, board visible; stopped at 20k | 80 / 0 / 15.2 / 0.2 / 4.8 | 0 / 0 / 13.8 / 0 / 4.9 | |
| **`moves_nb_s0`** | warm + penalty 5, **board hidden** (`--no-board`) | 7,120 / 24.0 / 95.7 / 10.9 / 29.1 | 27,610 / 120.2 / 344.1 / 35.5 / 147.7 | 43,680 / 189.4 / 496.0 / 54.3 / 217.1 |
| **`moves_nb_s1`** | same, seed 1 | 2,050 / 3.3 / 34.2 / 4.3 / 10.0 | 6,570 / 19.8 / 88.4 / 10.7 / 32.3 | 13,960 / 60.5 / 192.9 / 18.2 / 70.6 |
| **`moves_nb_s2`** | same, seed 2 | 4,090 / 9.9 / 58.1 / 7.5 / 19.5 | 4,640 / 11.8 / 65.3 / 8.4 / 23.4 | 31,440 / 126.2 / 350.4 / 40.3 / 133.8 |
| `moves_bs01_s0` | warm + penalty 5, board visible at **scale 0.1**; stopped at 40k | 1,370 / 0.5 / 31.2 / 3.3 / 6.5 | 1,010 / 0.5 / 28.8 / 2.4 / 7.3 | 1,470 / 0.7 / 29.2 / 3.5 / 7.9 |
| **`moves_bs001_s0`** | warm + penalty 5, board visible at **scale 0.01** | 1,800 / 0.8 / 33.0 / 4.2 / 9.4 | 2,620 / 4.9 / 49.1 / 5.3 / 18.0 | 3,230 / 11.6 / 58.7 / 4.9 / 20.5 |

`moves_nb_s0` went on to 47,700 at 60k, 49,020 at 90k and 44,500 at 100k; `moves_nb_s1` reached 52,370 at 100k with all 500 pieces survived. `moves_nb_s2` held 500 pieces from 50k to 90k (45,070 to 48,280) and fell to 39,540 (409 pieces) at 100k. `moves_bs001_s0` stayed under 140 pieces until 70k, reached 189 at 80k, 486 at 90k (38,550) and ended at 470 pieces (38,190).

Controls that rule out a pipeline bug:

- A hand-built linear net with the heuristic weights, run through the Python env and `best_move`: 5 games, 500-piece cap, 33,400 score, 197.4 lines, all 500 pieces survived, 28.4 T-spins.
- The warm-started network before any training, same path: 300-piece cap, 20,080 score, 118.8 lines, 300 pieces, 17.2 T-spins, 87.6 holds.
- The exported agents reproduce PyTorch's outputs in Go (`go test ./agent`).

## What the experiments say about why training failed

Training degraded a network that started out playing well, in every run where the 200 board cells were an input. The hypotheses, with what was found:

1. **Board input: confirmed as the cause, mechanism still open.** Hiding the board made training work on three seeds (`moves_nb_*`). Scaling the board inputs down also works, but only a lot: at 0.1 the run stayed flat (31 pieces at 40k, stopped), at 0.01 it learned, late (the jump came between 80k and 90k steps) and to a weaker agent than the board-hidden ones (best checkpoint 70k points on seeds 1-100, 1 of 100 games at the cap; the final one 122k, 21 of 100; clears are 91-94% singles). With Adam, scaling an input by s makes its weights change s times as much per step, so this says the board weights move too fast at normal scale. It does not say whether the cause is noise, capacity or the interaction with the warm start. Looking at the finished scale-0.01 network explains how it plays (mean change of the value per unit of each input, over the states it visited in seeds 1-3, compared with the seed 2 board-hidden network): holes -0.02 against -1.14, aggregate height -0.65 against +0.13, bumpiness -0.21 against -0.05, and each of the 200 board cells about 0.41 (up to 1.2), so one filled cell moves the value about twenty times more than one hole. It nearly ignores holes and attaches value to individual cells, which fits the tall one-column towers with holes under them in its games. My reading is that it memorises cell positions instead of learning a general rule, and that it had too little time at that level (it was under 140 pieces until 70k steps); neither is tested. A lower learning rate on every weight (`moves_lowlr_s0`, 1e-4) was stopped after one check (29.8 pieces at 10k) and is not conclusive.
2. **Zero-line T-spin bait: rejected as the cause.** Paying nothing for them in training (`moves_bait_s0`, board visible) still failed, and worse (13.8 pieces at 20k, no T-spins at all). The board-hidden agents learn with the 400 points on offer.
3. **Value scale and the game-over target.** A finished game used to bootstrap to exactly 0, which beats the negative values of a warm-started net; the death penalty of 5 is in every run since the second warm start and did not fix the board-visible runs. `train_moves.py` now logs the mean, min and max of the learned values to TensorBoard (`train/value_*`); I did not look at them.
4. **Update dynamics** (128-candidate max over a bootstrapped target, target sync every 1000 steps): not investigated. The placement DQN used the same scheme and worked on 4 features.

Do not redo: cold vs warm, with/without the death penalty at lr 1e-3 with the board visible (all failed the same way), nor the bait test above.

## How the training code works

- `MoveNet` (`dqn_moves.py`): MLP 218 -> 128 -> 64 -> 1, input divided by `SCALE` (first four features scaled, the rest 0/1). `warm_start=True` makes units 0-3 of both hidden layers pass the four scaled heuristic features through and sets the output weights to the heuristic weights times 0.1 (all other output weights 0), so it starts as the heuristic. `use_board=False` multiplies the 200 board inputs by 0 (the `visible` buffer); `agent_doc` zeroes those first-layer weights on export so the Go side stays a plain MLP. **Export runs trained with `--no-board` with `export_moves.py --no-board`**, otherwise the board weights, which never trained, are exported as they were initialised.
- U excludes the move's own reward. Move value = `reward/100 + 0.99 * U(afterstate)`, in Go (`agent.BestMove`) and in Python. Truncation at `--max-pieces` is not a terminal.
- `Replay` stores the 200 board cells bit-packed (43 bytes per candidate) so 50k transitions x 128 candidates fit in about 300 MB.
- Speed: about 25-40 environment steps/s with one run alone (100k steps take 40-60 minutes), slower when runs share the machine.
- Seeds: training games 1,000,000+; validation 5001-5010; final comparison seeds 1-100. Same as the placement agents.
- Flags: `--cold`, `--no-board`, `--board-scale`, `--zero-tspin-bait`, `--death-penalty`, `--lr`, `--update-every`, `--steps`, `--name`, `--seed`. Checkpoints `training/checkpoints/<name>_best.pt` (best validation) and `<name>.pt`; TensorBoard in `training/runs/<name>`; viewer data in `viewdata/<name>/`. All git-ignored.

```
go run ./cmd/server
cd training
python train_moves.py --name moves_nb_s0 --no-board
python export_moves.py moves_nb_s0 --no-board --final     # writes ../agents/moves_nb_s0_final.json (add --board-scale 0.01 instead of --no-board for that run type)
cd .. && go run ./cmd/baseline -agent agents/moves_nb_s0_final.json -games 100 -max-pieces 3000
go run ./cmd/viewer -mode training -run moves_nb_s0
```

## Limits and what is still open

- The search assumes the piece can be steered at will: no gravity, no lock delay. Real play with a 500 ms gravity could not do every listed move.
- T-spin = three-corner rule only; no mini, no wall-kick special case. A zero-line T-spin scores 400. No combos, no back-to-back.
- `server.MaxMoves = 128` silently truncates (never reached in play; 70 was the most seen).
- `engine.Step(SoftDrop)` moves one row and does not apply gravity on top; `Step` never locks on a soft drop.
- After a hold from an empty slot the "next piece" feature is zero (unknown); this is intentional.
- Viewer replays store placements, so move-agent games and human games cannot be saved.
- The agent sees one next piece even though the engine and viewer now queue five.
- Open: why the final networks dip (more validation games, a learning-rate decay, averaging checkpoints); an agent that sees the board and plays as well as the board-hidden ones (compact slot features such as column heights and wells instead of 200 raw cells, or board scales between 0.1 and 0.01); whether a death penalty larger than 5 improves robustness; a survival-focused measure.
