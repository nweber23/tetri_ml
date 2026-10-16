// Package agent runs trained afterstate-value networks exported by training/export.py.
package agent

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/nweber23/tetri_ml/engine"
	"github.com/nweber23/tetri_ml/heuristic"
)

type Layer struct {
	W [][]float64 `json:"w"` // [out][in]
	B []float64   `json:"b"`
}

type CurvePoint struct {
	Step  int     `json:"step"`
	Score float64 `json:"score"`
}

type checkCase struct {
	Features []float64 `json:"features"`
	Value    float64   `json:"value"`
}

// Agent is a ReLU MLP over the features of a placement's afterstate.
type Agent struct {
	Name   string       `json:"name"`
	Scale  []float64    `json:"scale"` // input scaling; 4 inputs for a placement agent, NumMoveFeatures for a move agent
	Layers []Layer      `json:"layers"`
	Curve  []CurvePoint `json:"curve"` // validation score during training
	Check  []checkCase  `json:"check"` // inputs with the values PyTorch produced
}

// Scored is one candidate placement with its afterstate features and value.
type Scored struct {
	Placement engine.Placement
	Features  heuristic.Features
	Value     float64
}

func Load(path string) (*Agent, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var a Agent
	if err := json.Unmarshal(data, &a); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	in := len(a.Scale)
	if in != 4 && in != NumMoveFeatures {
		return nil, fmt.Errorf("%s: %d inputs, want 4 (placement agent) or %d (move agent)", path, in, NumMoveFeatures)
	}
	for i, l := range a.Layers {
		if len(l.W) != len(l.B) {
			return nil, fmt.Errorf("%s: layer %d has %d weight rows but %d biases", path, i, len(l.W), len(l.B))
		}
		for _, row := range l.W {
			if len(row) != in {
				return nil, fmt.Errorf("%s: layer %d expects %d inputs, got %d", path, i, in, len(row))
			}
		}
		in = len(l.B)
	}
	if in != 1 {
		return nil, fmt.Errorf("%s: network must end in one output, got %d", path, in)
	}
	return &a, nil
}

// Value is the network's value of an afterstate with the given features.
func (a *Agent) Value(f []float64) float64 {
	x := make([]float64, len(f))
	for i := range f {
		x[i] = f[i] / a.Scale[i]
	}
	for i, l := range a.Layers {
		y := make([]float64, len(l.B))
		for o := range y {
			sum := l.B[o]
			for j, v := range x {
				sum += l.W[o][j] * v
			}
			if i < len(a.Layers)-1 && sum < 0 {
				sum = 0
			}
			y[o] = sum
		}
		x = y
	}
	return x[0]
}

// Evaluate scores every placement of g's current piece, in Placements order.
func (a *Agent) Evaluate(g *engine.Game) []Scored {
	var out []Scored
	for _, pl := range g.Placements() {
		b, lines := g.Afterstate(pl)
		f := heuristic.Extract(b, lines)
		v := a.Value([]float64{float64(f.AggregateHeight), float64(f.Holes), float64(f.Bumpiness), float64(f.Lines)})
		out = append(out, Scored{pl, f, v})
	}
	return out
}

// Best returns the index of the highest-valued entry, or -1 if there is none.
func Best(scored []Scored) int {
	best := -1
	for i, s := range scored {
		if best < 0 || s.Value > scored[best].Value {
			best = i
		}
	}
	return best
}

// PlaysMoves reports whether the agent chooses among engine.Moves (slides, spins, hold) rather than straight-drop placements.
func (a *Agent) PlaysMoves() bool { return len(a.Scale) == NumMoveFeatures }

// ScoredMove is one candidate move with the network's value of the state after it.
type ScoredMove struct {
	Move  engine.Move
	Value float64
}

// EvaluateMoves scores every move of g, in Moves order. Reward is not added:
// the network values the afterstate, and the caller adds the move's own reward.
func (a *Agent) EvaluateMoves(g *engine.Game) []ScoredMove {
	moves := g.Moves()
	out := make([]ScoredMove, len(moves))
	x := make([]float64, NumMoveFeatures)
	for i, m := range moves {
		for j, v := range MoveFeatures(g, m) {
			x[j] = float64(v)
		}
		out[i] = ScoredMove{m, a.Value(x)}
	}
	return out
}

// moveGamma discounts the afterstate value against the move's own reward; it must equal GAMMA in training/dqn_moves.py.
const moveGamma = 0.99

// BestMove returns the index of the move with the highest reward/100 + gamma * afterstate value, or -1 if there is none.
func BestMove(scored []ScoredMove) int {
	best, bestValue := -1, 0.0
	for i, s := range scored {
		if v := float64(s.Move.Reward)/100 + moveGamma*s.Value; best < 0 || v > bestValue {
			best, bestValue = i, v
		}
	}
	return best
}
