// Command baseline plays a scripted agent on a range of seeds and prints average results.
package main

import (
	"flag"
	"fmt"
	"log"

	"math"
	"strings"

	net "github.com/nweber23/tetri_ml/agent"
	"github.com/nweber23/tetri_ml/engine"
	"github.com/nweber23/tetri_ml/heuristic"
)

func main() {
	games := flag.Int("games", 100, "number of games, seeds are first-seed .. first-seed+games-1")
	first := flag.Uint64("seed", 1, "first seed")
	maxPieces := flag.Int("max-pieces", 1000, "stop a game after this many pieces")
	agent := flag.String("agent", "heuristic", "heuristic, heuristic-moves (the same weights choosing among slides, spins and hold), a path to an exported agents/*.json, or cycle (deterministic placement index 7*pieces mod count, used to cross-check the Python env)")
	verbose := flag.Bool("v", false, "print one line per game")
	flag.Parse()

	var pick func(g *engine.Game) engine.Placement
	var playMove func(g *engine.Game) engine.Move // set for agents that choose full moves; pick is used otherwise
	switch *agent {
	case "heuristic-moves":
		playMove = func(g *engine.Game) engine.Move {
			best, bestScore := engine.Move{}, math.Inf(-1)
			for _, m := range g.Moves() {
				if sc := heuristic.Default.Score(heuristic.Extract(m.Board, m.Lines)); sc > bestScore {
					best, bestScore = m, sc
				}
			}
			return best
		}
	case "heuristic":
		pick = func(g *engine.Game) engine.Placement { return heuristic.Best(g, heuristic.Default) }
	case "cycle":
		pick = func(g *engine.Game) engine.Placement {
			pls := g.Placements()
			return pls[7*g.State().Pieces%len(pls)]
		}
	default:
		if !strings.HasSuffix(*agent, ".json") {
			log.Fatalf("unknown agent %q", *agent)
		}
		a, err := net.Load(*agent)
		if err != nil {
			log.Fatal(err)
		}
		if a.PlaysMoves() {
			playMove = func(g *engine.Game) engine.Move {
				sc := a.EvaluateMoves(g)
				return sc[net.BestMove(sc)].Move
			}
		} else {
			pick = func(g *engine.Game) engine.Placement {
				sc := a.Evaluate(g)
				return sc[net.Best(sc)].Placement
			}
		}
	}

	var score, lines, pieces, capped int
	var clears [5]int // clears[n] = placements that cleared n lines
	var tspins, holds int
	minPieces := -1
	for i := 0; i < *games; i++ {
		seed := *first + uint64(i)
		g := engine.New(seed)
		for s := g.State(); !s.Done && s.Pieces < *maxPieces; s = g.State() {
			before := s.Lines
			if playMove != nil {
				m := playMove(g)
				g.Do(m)
				if m.TSpin {
					tspins++
				}
				if m.Hold {
					holds++
				}
			} else {
				g.Place(pick(g))
			}
			clears[g.State().Lines-before]++
		}
		s := g.State()
		if *verbose {
			fmt.Printf("seed=%d score=%d lines=%d pieces=%d\n", seed, s.Score, s.Lines, s.Pieces)
		}
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
	fmt.Printf("score/piece=%.2f | share of cleared lines by clear size: single=%.1f%% double=%.1f%% triple=%.1f%% tetris=%.1f%%\n",
		float64(score)/float64(pieces), share(clears, 1), share(clears, 2), share(clears, 3), share(clears, 4))
	if playMove != nil {
		fmt.Printf("per game: T-spins=%.1f holds=%.1f (%.1f%% of moves used hold)\n", float64(tspins)/n, float64(holds)/n, 100*float64(holds)/float64(pieces))
	}
}

// share is the percentage of all cleared lines that came from clears of n lines.
func share(clears [5]int, n int) float64 {
	total := 0
	for k, c := range clears {
		total += k * c
	}
	return 100 * float64(n*clears[n]) / float64(total)
}
