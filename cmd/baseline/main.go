// Command baseline plays the heuristic agent on a range of seeds and prints average results.
package main

import (
	"flag"
	"fmt"

	"github.com/nweber23/tetri_ml/engine"
	"github.com/nweber23/tetri_ml/heuristic"
)

func main() {
	games := flag.Int("games", 100, "number of games, seeds are first-seed .. first-seed+games-1")
	first := flag.Uint64("seed", 1, "first seed")
	maxPieces := flag.Int("max-pieces", 1000, "stop a game after this many pieces")
	flag.Parse()

	var score, lines, pieces, capped int
	minPieces := -1
	for i := 0; i < *games; i++ {
		g := engine.New(*first + uint64(i))
		for s := g.State(); !s.Done && s.Pieces < *maxPieces; s = g.State() {
			g.Place(heuristic.Best(g, heuristic.Default))
		}
		s := g.State()
		score += s.Score
		lines += s.Lines
		pieces += s.Pieces
		if !s.Done {
			capped++
		}
		if minPieces < 0 || s.Pieces < minPieces {
			minPieces = s.Pieces
		}
	}
	n := float64(*games)
	fmt.Printf("games=%d seeds=%d..%d max-pieces=%d\n", *games, *first, *first+uint64(*games)-1, *maxPieces)
	fmt.Printf("avg score=%.1f lines=%.1f pieces=%.1f | min pieces=%d | hit cap=%d/%d\n",
		float64(score)/n, float64(lines)/n, float64(pieces)/n, minPieces, capped, *games)
}
