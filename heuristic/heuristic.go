// Package heuristic scores board states with a hand-weighted linear function
// and picks the best placement for the current piece.
package heuristic

import "github.com/nweber23/tetri_ml/engine"

// Features describes an afterstate.
type Features struct {
	AggregateHeight int // sum of column heights
	Holes           int // empty cells with a filled cell somewhere above them
	Bumpiness       int // sum of height differences between neighbouring columns
	Lines           int // lines cleared by the placement
}

// Extract computes the features of b, the board after a placement that cleared lines.
func Extract(b engine.Board, lines int) Features {
	f := Features{Lines: lines}
	var heights [engine.Width]int
	for x := 0; x < engine.Width; x++ {
		for y := engine.Height - 1; y >= 0; y-- {
			if b[y][x] == 0 {
				continue
			}
			heights[x] = y + 1
			for yy := 0; yy < y; yy++ {
				if b[yy][x] == 0 {
					f.Holes++
				}
			}
			break
		}
		f.AggregateHeight += heights[x]
		if x > 0 {
			f.Bumpiness += abs(heights[x] - heights[x-1])
		}
	}
	return f
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// Weights multiply the features to give a score; higher is better.
type Weights struct {
	AggregateHeight, Holes, Bumpiness, Lines float64
}

// Default are the weights popularised by Yiyuan Lee's Tetris AI write-up.
var Default = Weights{
	AggregateHeight: -0.510066,
	Holes:           -0.35663,
	Bumpiness:       -0.184483,
	Lines:           0.760666,
}

func (w Weights) Score(f Features) float64 {
	return w.AggregateHeight*float64(f.AggregateHeight) +
		w.Holes*float64(f.Holes) +
		w.Bumpiness*float64(f.Bumpiness) +
		w.Lines*float64(f.Lines)
}

// Best returns the placement of g's current piece with the highest score. It
// must not be called on a finished game. Ties go to the first placement found.
func Best(g *engine.Game, w Weights) engine.Placement {
	var best engine.Placement
	bestScore := 0.0
	for i, pl := range g.Placements() {
		b, lines := g.Afterstate(pl)
		if s := w.Score(Extract(b, lines)); i == 0 || s > bestScore {
			best, bestScore = pl, s
		}
	}
	return best
}
