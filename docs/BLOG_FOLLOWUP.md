# Blog update: two guesses, tested

*Paste-ready additions for the portfolio post. Numbers are in `docs/ANALYSIS.md`.*

## 1. Replace the "my guess, which I did not test" sentence under "What worked"

> The agent beats the heuristic it started from by about 44% on score while playing about as many lines, because it clears them in bigger groups. I checked this afterwards: 83% of the heuristic's cleared lines come from single-line clears, against 16% for the DQN, which gets 61% of its lines from doubles and 7% from tetrises. A single is worth 100 points per line and a tetris 200, so the same number of lines is worth much more.

## 2. Add under "The measurement was the real problem"

> Two fixes helped. **Score per piece** barely moves with the cap (heuristic 43.4 at 3,000 pieces and 43.5 at 10,000; DQN 59.0 and 59.2), so it compares agents without depending on where I stop the game. And **raising the cap to 10,000 pieces** finally showed a survival gap that 3,000 hid: the heuristic reached the cap in 51 of 100 games, the DQN in 90 and the Double DQN in 85. The learned agents are more robust, not just higher-scoring. The 90 vs 85 is one training seed each and I do not read anything into it. One of the Double DQN's games ended after only 268 pieces, which alone is enough to pull its average down.

## 3. Add one line to "Things I did not try" / "What I would do next"

> I have not re-run these measurements on the other training seeds, because only one network per variant is exported. That is the first thing to do before claiming anything about the spread between seeds.

## 4. Tooling note (optional, one sentence in "Watching it learn" or the setup section)

> The Go baseline tool can now run any exported network (`go run ./cmd/baseline -agent agents/raw.json`), using the same code as the viewer. At the 3,000 cap it reproduces the PyTorch evaluation exactly (173,983 for the DQN seed 0, 119,880 for the heuristic), which doubles as a check that the Go and PyTorch networks agree.

## 5. New section: "Making it playable" (the show-off part)

> The agent only ever picks a final column and rotation, so it never needed the parts of Tetris that make it feel like Tetris. I added them to the engine and the viewer's human mode, as a way to play against what I had built:
>
> - **Hold piece.** Swap the current piece with a stored one, once per piece.
> - **Lock delay.** A piece resting on the stack waits half a second before it locks, and every move or rotation restarts the wait, at most 15 times per piece, so you cannot stall forever. This is the mechanic that makes sliding and tucking a piece in at the last moment possible.
> - **T-spins.** A T piece that ends on a rotation with three of its four corners blocked scores 400, 800, 1200 or 1600 for 0 to 3 lines. This is the simple three-corner rule, with no mini T-spins.
> - **Ghost piece** showing where a piece would land.
>
> The playable game does not touch training: the agent still plays through the placement API, which never rotates in place, so every number above is unchanged. The engine's `Step` still locks immediately; the lock delay lives in the viewer.

## 6. New section: "Giving the agent the whole game" (what worked and what did not)

> A placement agent cannot tuck, slide or spin, so I tried to give it everything. Teaching it key presses directly would make a T-spin a five-step sequence with no reward until the end, the sparse-reward problem again and worse. Instead I wrote a search: from the spawn position, a breadth-first search over left, right, rotate and soft drop finds every spot where a piece can lock, and with hold, every spot for the piece a hold would bring in. That is about 43 candidate moves per position, and the agent picks one. A move remembers the keys that get there, so the viewer can play it out and you can watch the slide or the spin.
>
> **A scripted baseline already benefits.** The same four hand-set heuristic weights, now choosing among all moves, reached the 1,000-piece cap in all 20 games (the placement version did in 96 of 100) and made about 24 T-spins and 292 holds per game, without knowing what a T-spin is. The weights say nothing about them; the search simply finds them.
>
> **The learned agent has not learned yet.** I gave the network the board, the held piece and the next piece as input, because four summary features cannot describe a T-slot. In four runs (random initialisation, heuristic warm start, with and without a game-over penalty) it stayed at about 22 pieces per game, which is what random play does, while the placement agent reaches the cap within 10,000 steps. The pipeline is not the cause: a hand-built network with the heuristic weights survives all 500 pieces through the same code, and so does the untrained warm-started network. Training itself degrades it. My leading suspects, not yet tested: the 200 raw board inputs add noise that swamps small differences between candidate moves, and a zero-line T-spin pays 400 points, which baits a greedy policy. Two follow-up runs (board hidden; lower learning rate) were still going when I wrote this.

(Fill in the outcome of those two runs before publishing. If a learned move agent works, replace the last paragraph with its numbers: score and lines at the 3,000-piece cap, T-spins and holds per game, against the heuristic-moves baseline above. Full record in `docs/MOVES.md`.)

Also update "Limits": hold, lock delay and T-spins exist in the human mode, and the search lets a scripted agent use them; a learned agent does not yet (see section 6).

## Also change

- Results table: add a "Score / piece" column (43.4, 59.0, 58.5).
- Limits: "Tested on Windows 11 only" is out of date; the README now says Windows and Linux.
