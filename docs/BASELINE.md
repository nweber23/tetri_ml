# Heuristic baseline

Agent: `heuristic.Best` with `heuristic.Default` weights (aggregate height -0.510066, holes -0.35663, bumpiness -0.184483, lines +0.760666). It picks the placement whose afterstate scores highest. Placements are straight drops, no tucks or spins.

Score is the line-clear score (100/300/500/800). Games stop at game over or at the piece cap.

```
go run ./cmd/baseline -games 100 -seed 1 -max-pieces <cap>
```

| Cap (pieces) | Games | Avg score | Avg lines | Avg pieces | Min pieces | Hit cap |
|---|---|---|---|---|---|---|
| 1000 | 100 (seeds 1-100) | 42591.0 | 390.9 | 988.6 | 478 | 96/100 |
| 10000 | 100 (seeds 1-100) | 318458.0 | 2915.5 | 7315.8 | 478 | 51/100 |

Capped games make avg pieces, lines and score lower bounds. Compare agents against the 1000-piece row: it is fast to run, and a learning agent will not survive that long early on.
