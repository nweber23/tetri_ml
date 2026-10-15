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
	Features [4]float64 `json:"features"`
	Value    float64    `json:"value"`
}

// Agent is a ReLU MLP over the features of a placement's afterstate.
type Agent struct {
	Name   string       `json:"name"`
	Scale  [4]float64   `json:"scale"`
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
func (a *Agent) Value(f [4]float64) float64 {
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
		v := a.Value([4]float64{float64(f.AggregateHeight), float64(f.Holes), float64(f.Bumpiness), float64(f.Lines)})
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
