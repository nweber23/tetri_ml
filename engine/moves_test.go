package engine

import "testing"

// fill sets the given cells of b, one string per row from the bottom up; '#' is filled.
func fill(b *Board, rows ...string) {
	for y, row := range rows {
		for x, c := range row {
			if c == '#' {
				b[y][x] = 1
			}
		}
	}
}

func TestMovesIncludePlacements(t *testing.T) {
	g := New(7)
	have := map[[6]Point]bool{}
	for _, m := range g.Moves() {
		have[cellsID(m.Piece, false)] = true
	}
	for _, pl := range g.Placements() {
		if p := g.Landing(pl); !have[cellsID(p, false)] {
			t.Errorf("placement %+v landing %+v is not among the moves", pl, p)
		}
	}
}

func TestMovesSlideUnderOverhang(t *testing.T) {
	g := New(1)
	fill(&g.board, "", "", "######")
	g.cur = spawnPiece(I)
	g.holdUsed = true
	var found bool
	for _, m := range g.Moves() {
		if m.Piece.Kind != I {
			continue
		}
		for _, c := range m.Piece.Cells() {
			if c.X <= 3 && c.Y == 0 {
				found = true
			}
		}
	}
	if !found {
		t.Error("no move slides the I piece under the shelf; a straight drop cannot reach it")
	}
	for _, pl := range g.Placements() {
		for _, c := range g.Landing(pl).Cells() {
			if c.X <= 3 && c.Y == 0 {
				t.Fatal("test board is wrong: a straight drop reaches under the shelf")
			}
		}
	}
}

func TestMovesFindTSpinDouble(t *testing.T) {
	g := New(1)
	fill(&g.board, "####.#####", "###...####", "   #")
	g.cur = spawnPiece(T)
	g.holdUsed = true
	for _, m := range g.Moves() {
		if m.TSpin && m.Lines == 2 {
			before := g.State().Score
			if r, _ := g.Do(m); r != 1200 || g.State().Score != before+1200 || !g.State().TSpin {
				t.Errorf("playing the T-spin double gave reward %d", r)
			}
			return
		}
	}
	t.Error("no T-spin double among the moves")
}

func TestDoMatchesPrediction(t *testing.T) {
	g := New(3)
	for i := 0; i < 300 && !g.State().Done; i++ {
		ms := g.Moves()
		m := ms[(7*i+3)%len(ms)]
		want := g.State().Score + m.Reward
		wantBoard := m.Board
		if r, _ := g.Do(m); r != m.Reward || g.State().Score != want {
			t.Fatalf("move %d: reward %d, predicted %d", i, r, m.Reward)
		}
		if st := g.State(); !st.Done && st.Board != wantBoard {
			t.Fatalf("move %d: board differs from the predicted afterstate", i)
		}
	}
}
