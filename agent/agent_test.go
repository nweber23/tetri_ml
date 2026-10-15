package agent

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/nweber23/tetri_ml/engine"
)

func TestValueMatchesPyTorch(t *testing.T) {
	files, _ := filepath.Glob("../agents/*.json")
	if len(files) == 0 {
		t.Skip("no exported agents")
	}
	for _, f := range files {
		a, err := Load(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(a.Check) == 0 {
			t.Fatalf("%s has no check cases", f)
		}
		for _, c := range a.Check {
			if got := a.Value(c.Features); math.Abs(got-c.Value) > 1e-3 {
				t.Errorf("%s: Value(%v) = %v, PyTorch gave %v", f, c.Features, got, c.Value)
			}
		}
	}
}

func TestLoadRejectsBadShapes(t *testing.T) {
	if _, err := Load("nonexistent.json"); err == nil {
		t.Error("expected an error for a missing file")
	}
}

func TestEvaluateAndBest(t *testing.T) {
	// One linear layer that prefers fewer holes.
	a := &Agent{Scale: [4]float64{1, 1, 1, 1}, Layers: []Layer{{W: [][]float64{{0, -1, 0, 0}}, B: []float64{0}}}}
	g := engine.New(1)
	scored := a.Evaluate(g)
	if len(scored) != len(g.Placements()) {
		t.Fatalf("%d scores for %d placements", len(scored), len(g.Placements()))
	}
	best := Best(scored)
	for _, s := range scored {
		if s.Value > scored[best].Value {
			t.Errorf("Best picked %v but %v is higher", scored[best].Value, s.Value)
		}
	}
	if Best(nil) != -1 {
		t.Error("Best(nil) should be -1")
	}
}
