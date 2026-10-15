// Command headless plays one seeded game with random actions and prints the result.
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"

	"github.com/nweber23/tetri_ml/engine"
)

func main() {
	seed := flag.Uint64("seed", 1, "game seed")
	flag.Parse()

	g := engine.New(*seed)
	r := rand.New(rand.NewPCG(*seed, 1))
	steps := 0
	for done := false; !done; steps++ {
		_, done = g.Step(engine.Action(r.IntN(int(engine.NumActions))))
	}
	s := g.State()
	fmt.Printf("seed=%d steps=%d pieces=%d lines=%d score=%d\n", *seed, steps, s.Pieces, s.Lines, s.Score)
}
