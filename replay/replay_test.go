package replay

import (
	"path/filepath"
	"testing"

	"github.com/nweber23/tetri_ml/engine"
)

func TestRoundTripAndResimulation(t *testing.T) {
	g := engine.New(9)
	r := Replay{Seed: 9, Agent: "test"}
	for i := 0; i < 15; i++ {
		pl := g.Placements()[i%len(g.Placements())]
		r.Placements = append(r.Placements, pl)
		g.Place(pl)
	}
	path := filepath.Join(t.TempDir(), "r.json")
	if err := Save(path, r); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	h := engine.New(loaded.Seed)
	for _, pl := range loaded.Placements {
		h.Place(pl)
	}
	if h.State() != g.State() {
		t.Error("replayed game differs from the recorded one")
	}
}
