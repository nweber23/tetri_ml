package engine

import "testing"

func TestPlacementCounts(t *testing.T) {
	want := map[Kind]int{I: 7 + 10, O: 9, T: 8 + 9 + 8 + 9, S: 8 + 9, Z: 8 + 9, J: 8 + 9 + 8 + 9, L: 8 + 9 + 8 + 9}
	for k, n := range want {
		g := New(1)
		g.cur = spawnPiece(k)
		if got := len(g.Placements()); got != n {
			t.Errorf("kind %d: %d placements on an empty board, want %d", k, got, n)
		}
	}
}

func TestPlaceMatchesAfterstate(t *testing.T) {
	g := New(1)
	for x := 0; x < 6; x++ {
		g.board[0][x] = 1
	}
	g.cur = spawnPiece(I)
	pl := Placement{Rot: 0, X: 6}
	before := g.State()
	b, lines := g.Afterstate(pl)
	if g.State() != before {
		t.Fatal("Afterstate changed the game")
	}
	if lines != 1 || b[0] != [Width]uint8{} {
		t.Errorf("afterstate lines=%d row0=%v", lines, b[0])
	}
	if reward, _ := g.Place(pl); reward != 100 {
		t.Errorf("reward = %d, want 100", reward)
	}
	if g.State().Board != b {
		t.Error("Place and Afterstate disagree on the board")
	}
}

func TestPlaceIgnoresInvalid(t *testing.T) {
	g := New(1)
	before := g.State()
	g.Place(Placement{Rot: 0, X: 50})
	if g.State() != before {
		t.Error("invalid placement changed the game")
	}
}

func TestPlacementsOnlyGameEnds(t *testing.T) {
	g := New(1)
	for !g.State().Done {
		g.Place(g.Placements()[0])
	}
	if len(g.Placements()) != 0 {
		t.Error("finished game should have no placements")
	}
}
