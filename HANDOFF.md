# HANDOFF: finish the full-game ("uses hold, tucks, spins, T-spins") agent

**If you continue this work, delete this file (`git rm HANDOFF.md` if it is committed, otherwise `rm HANDOFF.md`) as soon as you have read it and moved anything still relevant into `docs/`.** It is a one-time note, not documentation. The durable record is `docs/MOVES.md`; read that too.

Written 2026-10-16 by the previous agent. The user (Niklas) wants: "a machine that can use all the features of tetris, not just drop". Everything below is verified unless marked **GUESS**.

## 0. Rules you must follow (from `CLAUDE.md` and `docs/`)

- Commits: Conventional Commits, imperative subject, a body that says *why*. **Never add a `Co-Authored-By` line or any attribution** (user's global rule). Commit in reasonable steps. The user asked for commits, so committing finished work is expected; do not push.
- Think before building, state assumptions, prefer the simplest thing, touch only what the task needs, mention dead code instead of deleting it. Ask before adding any new third-party dependency. Security notes in `docs/RULES.md`.
- Report results honestly. The blog post is built from your numbers: say plainly when something did not work.
- Do not claim a training result you did not read from a log or `metrics.jsonl`.

## 1. State of the repo

`git log` (newest first): `3f34312` docs (MOVES.md, blog follow-up), `84402ac` training options (warm start, death penalty, board-hidden), `f322975` baseline/analysis docs, `e952371` viewer, `08a5174` PlayMoves RPC + Tetris-v1 + move DQN, `260983f` engine moves/hold/T-spin. Working tree was clean except this file.

All tests pass: `go vet ./... && go test ./...` and, from `training/`, `.venv/bin/python -m unittest` (16 tests, builds its own server).

What exists and works:

| Piece | Where | Notes |
|---|---|---|
| Hold, T-spin (3-corner rule, rotation last), `SoftDrop` action, `Lock`, `Grounded` | `engine/game.go`, `engine/board.go` | T-spin 400/800/1200/1600 for 0..3 lines, no minis. Placement API cannot rotate in place, so old results are untouched |
| Reachable-move search | `engine/moves.go` (`Moves`, `Do`) | BFS, dedup by filled cells + T-spin flag, ~43 moves avg, 70 max, ~0.5-1.3 ms/call. Hold candidates are marked `Move.Hold` |
| Network features | `agent/features.go` (`MoveFeatures`, 218 bytes) | layout documented in the file; Python depends on it (`training/dqn_moves.py`: `BOARD = slice(4, 204)`, `pack/unpack`) |
| Move agent in Go | `agent/agent.go` (`PlaysMoves`, `EvaluateMoves`, `BestMove`) | `BestMove` ranks `reward/100 + U` **without gamma**; Python ranks `reward/100 + 0.99*U`. **Make the Go side use 0.99 before exporting an agent** (one-line fix, add a test); currently they could disagree on ties |
| gRPC | `proto/tetris/v1/tetris.proto` (`PlayMoves`), `server/server.go` | `MaxMoves = 128` truncation (silent) |
| Gym env | `training/tetris_env/env.py` (`MoveEnv`, id `Tetris-v1`) | `info`: `action_mask`, `features` (128x218 uint8), `rewards`, `score`, `lines`, `pieces`, `tspins`, `holds` |
| Training | `training/dqn_moves.py`, `train_moves.py`, `export_moves.py` | see section 3 |
| Viewer | `viewer/session.go`, `cmd/viewer/main.go` | human mode: hold (C), 500 ms lock delay (15 resets), ghost, T-spin readout. Move agents steered key by key (`keyInterval` 70 ms). Not run by hand, only screenshot + unit tests |
| Baseline | `cmd/baseline` | `-agent heuristic`, `heuristic-moves`, `agents/*.json` (placement or move agents); prints clear-size shares, T-spins, holds |

**Not done / not documented yet** (do these once there is something to show):

- `docs/BRIDGE.md` and `README.md` do not mention `PlayMoves` / `Tetris-v1`. `docs/VIEWER.md` does not mention move agents. Replays cannot be saved for move agents (`Played` stays empty); the error message says "only agent games with at least one placement".
- `docs/BLOG_FOLLOWUP.md` section 6 has a placeholder line "Fill in the outcome of those two runs" — resolve it.
- No move agent is exported in `agents/` (only `raw.json`, `ddqn_s0.json`).

## 2. The problem: the learned move agent does not learn

Validation = 10 greedy games, seeds 5001-5010, 500-piece cap, every 10k steps (the `train_moves.py` log line and `viewdata/<run>/metrics.jsonl`). A game of ~22 pieces equals random play (random ≈ 25). For reference the old placement DQN was at the 500-piece cap by 10k steps (`docs/DQN.md`).

Columns: score / lines / pieces / T-spins / holds (per game).

| Run (name on disk) | Setup | 10k | 20k | 30k |
|---|---|---|---|---|
| `moves_s0` | random init, no death penalty, lr 1e-3, board visible (killed ~35k) | 440 / 0 / 19.9 / 1.1 / 4.0 | 160 / 0.4 / 23.8 / 0.3 / 8.3 | 630 / 0.3 / 23.3 / 1.5 / 8.8 |
| first warm run (overwritten on disk) | heuristic warm start, no penalty (killed 20k) | 160 / 0 / 13.4 / 0.4 / 5.1 | 120 / 0 / 16.7 / 0.3 / 5.1 | |
| `moves_warm_s0` | warm start + death penalty 5 (killed 20k) | 920 / 0 / 22.5 / 2.3 / 6.1 | 490 / 0.1 / 21.5 / 1.2 / 7.1 | |
| `moves_cold_s0` | random init + penalty 5 (killed 20k) | 1050 / 0.1 / 22.8 / 2.6 / 8.0 | 520 / 0 / 23.5 / 1.3 / 6.9 | |
| `moves_nb_s0` | warm + penalty 5, `--no-board` (board hidden) | **running when written** | | |
| `moves_lowlr_s0` | warm + penalty 5, `--lr 1e-4`, board visible | **running when written** | | |

Controls (these rule out a bug in env/features/selection):

- Hand-built linear net with the heuristic weights through `Tetris-v1` + `best_move`: 5 games, cap 500 → score 33,400, lines 197.4, 500 pieces (all survived), T-spins 28.4, holds 140.2.
- Untrained warm-started `MoveNet` (shrink 0.1) through the same path: cap 300 → 20,080, lines 118.8, 300 pieces, T-spins 17.2, holds 87.6. (shrink 1.0: 19,580 / 119.0 / 300 / 15.6 / 81.0.)
- Scripted `heuristic-moves` (`go run ./cmd/baseline -agent heuristic-moves -games 20 -max-pieces 1000`): score 54,920, lines 398.4, 20/20 reached the cap, 23.9 T-spins and 291.9 holds per game, 29.2% of moves used hold, clear shares single 89.5 / double 9.5 / triple 0.9 / tetris 0.1 %.

Conclusion that is verified: **training degrades a network that starts out playing well**, within 10k steps. Cause is not yet known. The validation scores of 400-1000 with ~0 lines are almost entirely zero-line T-spins (400 points each).

### Hypotheses (all **GUESS**, ordered by how cheap they are to test)

1. **Board-input noise.** 200 raw cells make the value of near-identical candidates noisy; the noise exceeds the small real differences. The two running experiments test it: if `moves_nb_s0` (board hidden) learns and `moves_lowlr_s0` does not, this is it. If both fail, it is not (only).
2. **Zero-line T-spin bait.** A 0-line T-spin pays 400 = 4.0 value units, more than any normal clear (single 1.0, double 3.0). The greedy policy may chase it. Test: train with that reward zeroed. Needs per-move `t_spin`/`lines` in `info` (the proto `Move` already carries `t_spin` and `lines`; `MoveEnv._call` currently drops them) and the same zeroing in both action selection and the TD target, or put it server-side behind a flag. Report real (unmodified) scores in validation either way.
3. **Value scale / terminal handling.** A finished game used to bootstrap to exactly 0, which beats the negative values of a warm-started net. Penalty 5 did not fix learning, so this is at most one contributor. Also check `U` statistics: log mean/min/max of `net(features)` and of targets during training.
4. **Update dynamics.** 128-candidate max over a bootstrapped target is optimistic; `target-every 1000`, lr 1e-3, batch 128. Not investigated. The placement DQN used the same values and worked on 4 features.

Do **not** redo: cold vs warm, with/without death penalty at lr 1e-3 with the board visible (all four failed the same way).

## 3. How the training code works (so you can change it safely)

- `MoveNet` (`dqn_moves.py`): MLP 218 → 128 → 64 → 1, input divided by `SCALE` (first four features scaled, the rest 0/1). `warm_start=True` makes units 0-3 of both hidden layers pass the four scaled heuristic features through and sets the output weights to the heuristic weights times 0.1 (all other output weights 0), so it starts as the heuristic. `use_board=False` multiplies the 200 board inputs by 0 (`visible` buffer) and `agent_doc` zeroes those first-layer weights on export so the Go side stays a plain MLP.
- U excludes the move's own reward. Move value = `reward/100 + 0.99 * U(afterstate)`. TD target for the played move's U = max over the next state's moves of that value, or `-death_penalty` (units of 100 points, default 5) if the game ended. Truncation at `--max-pieces` is not a terminal.
- `Replay` stores the 200 board cells bit-packed (43 bytes/candidate) so 50k transitions x 128 candidates fit in ~300 MB (the machine has 7 GB, ~2 GB free). `sample` trims the candidate dimension to the largest count in the batch.
- Speed: about 20-30 env steps/s with one run (≈40 min for 100k steps), slower with two at once (set `OMP_NUM_THREADS=3`). 6 cores, no GPU.
- Seeds: training games 1,000,000+; validation 5001-5010; final comparison seeds 1-100 (never trained on). Same as the placement agents.
- Flags: `--cold`, `--no-board`, `--death-penalty`, `--lr`, `--update-every`, `--steps`, `--name`. Checkpoints `training/checkpoints/<name>_best.pt` (best validation) and `<name>.pt`; TensorBoard in `training/runs/<name>` (same-name runs append to one dir, so it is mixed for the reused names); viewer data in `viewdata/<name>/` (metrics + a snapshot every 5k steps). `runs/`, `checkpoints/`, `viewdata/` are git-ignored.

## 4. What to do first

1. **Read the two running experiments.** They were started in the background with `nohup` and may still be running (or finished, or killed if the machine restarted):
   ```
   ps -eo pid,args | grep train_moves
   grep '"t": "validation"' viewdata/moves_nb_s0/metrics.jsonl viewdata/moves_lowlr_s0/metrics.jsonl
   ```
   Also the `go run ./cmd/server` process (needed by any training/eval) was left running; if it is gone start it: `go run ./cmd/server` (listens on 127.0.0.1:50051). Each run needs ~40+ min for 100k steps. Kill by PID from the `ps` output. **Never use `pkill -f` / `grep`-based kills with the process name in the command line: the pattern matches your own shell and kills it (happened twice).**
2. Decide from the results:
   - Board-hidden run learns (validation pieces clearly rising above ~25, lines > 0): hypothesis 1 is confirmed. Let it finish, then try the board visible with small lr only on the board weights, or add a few compact slot features instead of raw cells (e.g. column heights, wells, and "best T-spin reward available to a T piece on this board", computed by `engine.search` with a spawned T; cost ≈ one extra BFS per candidate). Report that the board-hidden agent cannot see T-slots, so its T-spins stay incidental.
   - Neither learns: test hypothesis 2 (zero-line T-spin reward zeroed in training), then log value statistics (hypothesis 3).
   - Fallback that still delivers the user's goal ("a machine that uses all the features") with little risk: tune the weights of `heuristic-moves` (including a T-spin / hold bonus term) with random search or CMA-style evolution against `cmd/baseline`. It needs no gradients and already reaches the cap. Say clearly in the blog that this is a scripted agent, not a learned one.
3. When any agent works: train to 100k steps, `cd training && python export_moves.py <name>` (writes `agents/<name>.json` with PyTorch test vectors; `go test ./agent` then checks the Go network reproduces them — it globs `agents/*.json`), fix the `BestMove` gamma noted above, then evaluate and compare on seeds 1-100:
   ```
   go run ./cmd/baseline -agent agents/<name>.json -games 100 -max-pieces 3000
   go run ./cmd/baseline -agent heuristic-moves   -games 100 -max-pieces 3000
   go run ./cmd/baseline -agent agents/raw.json   -games 100 -max-pieces 3000   # placement DQN, 173,983
   ```
   (~1-3 min each for placement agents; move agents are slower. Add `-max-pieces 10000` for survival.) Report score, lines, pieces, score/piece, T-spins/game, holds/game, and the cap hit rate. Note that move-agent scores include T-spin points and are not directly comparable to placement scores; lines/piece is the fairer comparison.
4. Update the docs: `docs/MOVES.md` (results table), `docs/BRIDGE.md` (`PlayMoves`, `Tetris-v1`), `docs/VIEWER.md` (move agents, no replay saving), `README.md` (table row, architecture, the run commands), `docs/BLOG_FOLLOWUP.md` section 6 (replace the placeholder with the real outcome; keep the failures, the blog's value is the honest account). Watch the viewer once by hand if a display is available (`go run ./cmd/viewer -mode training -run <name>`): the UI was only checked via screenshots.
5. Commit in small steps; then delete this file.

## 5. Known limitations to keep in mind (all deliberate, document if relevant)

- The search assumes the piece can be steered at will: no gravity, no lock delay. Real play with a 500 ms gravity could not do every listed move.
- T-spin = 3-corner rule only; no mini, no wall-kick special case. Zero-line T-spin scores 400. No combos, no back-to-back.
- `server.MaxMoves = 128` silently truncates (never reached in play; max seen 70 over 5 games of the heuristic).
- `engine.Step(SoftDrop)` moves one row and does not apply gravity on top; `Step` never locks on a soft drop.
- After a hold from an empty slot the "next piece" feature is zero (unknown); this is intentional.
- `heuristic.Extract` counts uint8 features; aggregate height can reach 240, still a byte.
- Viewer replays store placements, so move-agent games and human games cannot be saved.

## 6. Environment notes

- Linux, Go 1.27 (`go.mod` says 1.25.4), Python 3.12 venv at `training/.venv` (use `.venv/bin/python`), CPU-only PyTorch. Training scripts are run from `training/`.
- Regenerating gRPC stubs: `bash proto/generate.sh` (the script is not executable). `protoc-gen-go` v1.36.12 and `protoc-gen-go-grpc` v1.6.2 are installed in `~/go/bin` already; they match the committed generated files.
- Go test of the viewer needs no display. A screenshot run needs one: `go run ./cmd/viewer -mode human -screenshot out.png -screenshot-frame 30`.
- Scratch outputs of this session lived in `/tmp/claude-1000/` (logs of the failed runs, `cap10k_*.txt` for the 10,000-piece evaluation). They may be gone; everything that matters is in `docs/MOVES.md`, `docs/ANALYSIS.md` and this file.
- Two earlier deliverables already committed and unrelated to the open problem: `docs/ANALYSIS.md` (clear-size shares and the 10,000-piece survival numbers for the placement agents) and the blog follow-up sections 1-5.
