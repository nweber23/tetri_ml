# Phase 8: an agent that uses the whole game (hold, tucks, spins, T-spins)

The placement agents only choose a column and rotation for a straight drop. This phase gives the agent every way a player can lock a piece. Status: **the search, bridge and baseline work; no learned agent has learned yet.** What was tried and what it showed is below, so the next attempt does not repeat it.

## What was built

- `engine.Moves()` (`engine/moves.go`): a BFS over keys (left, right, rotate both ways, soft drop) from the spawn position. Every grounded position it reaches is a candidate, deduplicated by the cells it fills and by whether it is a T-spin. With hold available it also searches from the piece a hold would bring in. Each `Move` carries its key sequence, the board after locking, lines cleared and reward. `Game.Do(move)` plays it.
- No gravity and no lock delay inside the search: the piece can be steered at will, as in high-level play. About 43 candidate moves per state on average, 70 at most in play; the server caps the list at 128 (`server.MaxMoves`).
- T-spin: the three-corner rule, only when the last key was a rotation. Scores 400/800/1200/1600 for 0/1/2/3 lines. No mini T-spins. A zero-line T-spin is worth 400, which is a lot (a double is 300).
- `PlayMoves` RPC and the `Tetris-v1` env (`training/tetris_env/env.py`, `MoveEnv`): the action is an index into the listed moves.
- Network input (`agent.MoveFeatures`, 218 bytes, one source for Go and Python): aggregate height, holes, bumpiness, lines; the 20 visible rows as 0/1; one-hot of the held piece after the move; one-hot of the next piece (zero if unknown after a hold from an empty slot).
- `training/dqn_moves.py`, `train_moves.py`, `export_moves.py`: U(afterstate) excludes the move's own reward; a move is worth `reward/100 + 0.99 * U`; the TD target is the best such value over the next state's moves, or `-death_penalty` for a finished game.
- Viewer: move agents are steered key by key, so slides, spins and holds are visible. `go run ./cmd/baseline -agent heuristic-moves` (or any exported move agent) reports T-spins and holds per game.

## Baseline that already uses everything

The same four hand-set weights, choosing among all moves (`heuristic-moves`), seeds 1-20, 1000-piece cap: 54,920 average score, 398.4 lines, all 20 games reached the cap (the placement heuristic reached it in 96 of 100). 23.9 T-spins and 291.9 holds per game; 29% of moves used hold. The T-spins are accidental, the weights know nothing about them.

## Training attempts (all failed to learn so far)

Validation is 10 greedy games on seeds 5001-5010, 500-piece cap, every 10k steps. A game that ends after about 22 pieces is no better than random play (random play lasts about 25 pieces). For comparison, the placement DQN was already at the 500-piece cap at 10k steps.

| Run | Setup | 10k: score / lines / pieces / T-spins / holds | 20k | 30k |
|---|---|---|---|---|
| `moves_s0` | random init, no death penalty, lr 1e-3, board visible; stopped at ~35k | 440 / 0 / 19.9 / 1.1 / 4.0 | 160 / 0.4 / 23.8 / 0.3 / 8.3 | 630 / 0.3 / 23.3 / 1.5 / 8.8 |
| `moves_warm_s0` (first) | heuristic warm start, no death penalty; stopped at 20k | 160 / 0 / 13.4 / 0.4 / 5.1 | 120 / 0 / 16.7 / 0.3 / 5.1 | |
| `moves_warm_s0` (second) | heuristic warm start + death penalty 5; stopped at 20k | 920 / 0 / 22.5 / 2.3 / 6.1 | 490 / 0.1 / 21.5 / 1.2 / 7.1 | |
| `moves_cold_s0` | random init + death penalty 5; stopped at 20k | 1050 / 0.1 / 22.8 / 2.6 / 8.0 | 520 / 0 / 23.5 / 1.3 / 6.9 | |

Controls that rule out a pipeline bug:

- A hand-built linear net with the heuristic weights, run through the Python env and `best_move`: 5 games, 500-piece cap, 33,400 score, 197.4 lines, all 500 pieces survived, 28.4 T-spins.
- The warm-started network before any training, same path: 300-piece cap, 20,080 score, 118.8 lines, 300 pieces, 17.2 T-spins, 87.6 holds.

So the environment, features and move selection are right, and the untrained warm network plays well. Training itself makes it worse within 10k steps.

## What is known, what is guessed

Known: the points above.

Guesses, none verified:

1. The 200 raw board inputs add noise to the values of near-identical candidate states and that noise swamps the small real differences. Tested by two runs started last (`moves_nb_s0`: board hidden from the network; `moves_lowlr_s0`: board visible, lr 1e-4), results not yet read.
2. Zero-line T-spins (400 points, 4.0 in value units) bait the greedy policy. Validation scores of 400-1000 with about 0 lines are almost all T-spins.
3. A game-over target of exactly 0 made dying look better than living when values are negative (first warm run). The penalty did not fix the learning, so this is at most part of it.
