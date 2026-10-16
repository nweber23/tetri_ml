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
> - **Five-piece preview.** The engine now queues the next five pieces and the viewer shows them in a column. The piece order is unchanged (same 7-bag, one draw per piece), so every seeded result above still holds; the agents still only see the next piece.
>
> The playable game does not touch training: the agent still plays through the placement API, which never rotates in place, so every number above is unchanged. The engine's `Step` still locks immediately; the lock delay lives in the viewer.

## 6. New section: "Giving the agent the whole game" (what worked and what did not)

> A placement agent cannot tuck, slide or spin, so I tried to give it everything. Teaching it key presses directly would make a T-spin a five-step sequence with no reward until the end, the sparse-reward problem again and worse. Instead I wrote a search: from the spawn position, a breadth-first search over left, right, rotate and soft drop finds every spot where a piece can lock, and with hold, every spot for the piece a hold would bring in. That is about 43 candidate moves per position, and the agent picks one. A move remembers the keys that get there, so the viewer can play it out and you can watch the slide or the spin.
>
> **A scripted baseline already benefits.** The same four hand-set heuristic weights, now choosing among all moves, reached the 1,000-piece cap in all 20 games (the placement version did in 96 of 100) and made about 24 T-spins and 292 holds per game, without knowing what a T-spin is. The weights say nothing about them; the search simply finds them.
>
> **The learned agent failed four times, then worked when I hid the board.** I gave the network the board, the held piece and the next piece as input, because four summary features cannot describe a T-slot. In four runs (random initialisation, heuristic warm start, with and without a game-over penalty) it stayed at about 22 pieces per game, which is what random play does, while the placement agent reaches the cap within 10,000 steps. The pipeline was not the cause: a hand-built network with the heuristic weights survives all 500 pieces through the same code, and so does the untrained warm-started network. Training itself degraded it.
>
> I had two suspects. First, the 200 raw board inputs add noise that swamps small differences between candidate moves. Second, a zero-line T-spin pays 400 points, more than a double, which could bait a greedy policy. I tested them one at a time. Paying nothing for zero-line T-spins did not help (the run was worse, 14 pieces per game at 20,000 steps). **Hiding the board from the network did**: at 30,000 steps it survived 496 of 500 pieces in validation, with 189 lines, 54 T-spins and 217 holds per game, and two more training seeds learned too, both a bit later than the first. So the raw board input is what breaks learning. Why, I can only guess: noise, too much capacity, or the way the input interacts with the warm start would look the same from where I stand.
>
> I then tried to give the board back without breaking it, by multiplying the board inputs by a small number. With Adam that makes the board weights change that many times less per step. At 0.1 the run stayed flat (about 30 pieces per game, so I stopped it at 40,000 steps). At 0.01 it learned, but slowly: it was still under 140 pieces at 70,000 steps and only reached 486 at 90,000. The agent it produced is clearly worse than the board-hidden ones (below), and its games show why: it builds tall one-column towers with holes underneath. I looked at what its network responds to. Adding a hole changes its value by 0.02 on average, against 1.14 for the board-hidden agent, while a single filled board cell changes it by about 0.4, twenty times more than a hole. It has all but stopped caring about holes and attached its value to individual cells, which do not carry over to new boards. That is my reading of the numbers; I did not test it. So a network can use the board, but only if the board is made 100 times quieter, and what it gets out of it is not better play.
>
> **The result, on the 100 seeds the agent never trained on, with a 3,000-piece cap:**
>
> | Agent | Score | Lines | Pieces | Reached the cap | Score / piece | T-spins / game |
> |---|---|---|---|---|---|---|
> | Placement DQN | 173,983 | 1,173 | 2,948 | 96/100 | 59.0 | - |
> | Scripted agent choosing among all moves | 164,331 | 1,199 | 3,000 | 100/100 | 54.8 | 70 |
> | Learned move agent, seed 0, final network | 241,493 | 985 | 2,481 | 66/100 | 97.4 | 233 |
> | Learned move agent, seed 1 | 236,973 | 913 | 2,308 | 54/100 | 102.7 | 254 |
> | Learned move agent, seed 2, best-validation network | 285,432 | 1,157 | 2,905 | 94/100 | 98.3 | 313 |
> | Learned move agent, seed 2, final network | 104,800 | 406 | 1,046 | 8/100 | 100.2 | 106 |
> | Board visible at scale 0.01, final network | 121,679 | 574 | 1,474 | 21/100 | 82.5 | 140 |
>
> The board-hidden agents score about 44% to 74% more than the scripted one when they hold together, and they get there with T-spins, which pay 400 to 1,600 points: 233 to 313 per game against 70, so at least 93,000 of seed 0's 241,000 points. They do not clear more lines (0.40 per piece, the same as every other agent). They are **less robust**: they die before 3,000 pieces in 6% to 46% of the games for the good checkpoints, where the scripted agent never does. A higher score and a worse survival rate is a trade, not a clear win, and the score is partly my reward design: a zero-line T-spin pays 400, so an agent can be paid well for a risky stack.
>
> **The part I would not skip: one training run is not one agent.** The seed 2 network that scored 285,432 at 80,000 steps scored 104,800 at 100,000, with 8 of 100 games reaching the cap. For seed 0 it went the other way: the checkpoint I would have picked from 10 validation games scored 206,816 and the last one 241,493. Averaged over the three seeds, the best-validation picks score 243,074 and the final networks 194,422. The spread inside one run is as large as the gap to the scripted agent, so "the learned agent is better" is something I can say about what it can reach, not about what a given training run gives you. What I would try next: more validation games, a decaying learning rate, and averaging the last checkpoints.
>
> One more detail worth keeping. The board-hidden agent does not find T-spin slots by looking; I think it finds them because the move list already contains them with their reward, but I did not test that.

Also update "Limits": hold, lock delay and T-spins exist in the human mode, and the search lets both a scripted agent and a learned one use them (see section 6). The learned move agent has three training seeds, its checkpoints vary a lot within one run, it ends early in up to a half of 3,000-piece games, and it only learns if the board is hidden or scaled down.

## Also change

- Results table: add a "Score / piece" column (43.4, 59.0, 58.5).
- Add the GIFs of the move agent (`docs/media/moves_early.gif`, `moves_mid.gif`, `moves_late.gif`: untrained warm start, 20,000 steps, 100,000 steps, same seed) next to the placement ones; the placement GIFs were re-recorded with the five-piece preview.
- Limits: "Tested on Windows 11 only" is out of date; the README now says Windows and Linux.

## 7. New section: "Keeping it patched" (short)

> Before calling the project done I ran `govulncheck` on it. It found about two dozen vulnerabilities in the Go standard library, all because the toolchain on my machine was Go 1.25.4, plus some in `golang.org/x/net`, which gRPC pulls in. The scan says the code reaches some of them through gRPC and Ebitengine, though the server only listens on the loopback address, and they were easy to remove. The fix was to move `go.mod` to the newer toolchain (Go 1.27.2, which the `go` command downloads by itself) and upgrade `x/net` and its neighbours. The newer `x/net` needs Go 1.26, so the minimum Go version in the README went up from 1.25 to 1.26. `govulncheck` now reports nothing and the tests pass. `pip-audit` on the Python side was clean the whole time.
