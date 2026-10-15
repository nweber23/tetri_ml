package engine

import (
	"math/rand/v2"
	"testing"
)

func TestShapes(t *testing.T) {
	for k := Kind(0); k < NumKinds; k++ {
		n := boxSize(k)
		for rot := 0; rot < 4; rot++ {
			seen := map[Point]bool{}
			for _, c := range shapes[k][rot] {
				if c.X < 0 || c.X >= n || c.Y < 0 || c.Y >= n {
					t.Errorf("kind %d rot %d: cell %v outside %dx%d box", k, rot, c, n, n)
				}
				seen[c] = true
			}
			if len(seen) != 4 {
				t.Errorf("kind %d rot %d: want 4 distinct cells, got %d", k, rot, len(seen))
			}
		}
	}
}

func TestSRSShapes(t *testing.T) {
	// T pointing right (state R): .#. / .## / .#.
	want := map[Point]bool{{1, 2}: true, {1, 1}: true, {2, 1}: true, {1, 0}: true}
	for _, c := range shapes[T][1] {
		if !want[c] {
			t.Errorf("T rot 1: unexpected cell %v", c)
		}
	}
	// Vertical I in state R occupies box column 2.
	for _, c := range shapes[I][1] {
		if c.X != 2 {
			t.Errorf("I rot 1: unexpected cell %v", c)
		}
	}
}

func TestWallKick(t *testing.T) {
	g := New(1)
	// T in state R hugging the left wall cannot rotate to state 0 in place;
	// the second kick test (+1, 0) must move it right by one.
	g.cur = Piece{T, 1, -1, 10}
	g.tryRotate(3)
	if want := (Piece{T, 0, 0, 10}); g.cur != want {
		t.Errorf("got %+v, want %+v", g.cur, want)
	}
}

func TestRotateBlocked(t *testing.T) {
	g := New(1)
	g.cur = Piece{T, 0, 3, 10}
	for _, c := range [][2]int{{3, 11}, {4, 11}, {5, 11}, {3, 12}, {4, 12}, {5, 12}, {3, 9}, {5, 9}, {4, 9}, {3, 8}, {4, 8}, {5, 8}, {2, 10}, {6, 10}, {2, 11}, {6, 11}, {2, 12}, {6, 12}, {2, 9}, {6, 9}, {2, 8}, {6, 8}} {
		g.board[c[1]][c[0]] = 1
	}
	before := g.cur
	g.tryRotate(1)
	if g.cur != before {
		t.Errorf("rotation should fail when boxed in, got %+v", g.cur)
	}
}

func TestClearLines(t *testing.T) {
	var b Board
	for x := 0; x < Width; x++ {
		b[0][x] = 1
		b[2][x] = 1
	}
	b[1][4] = 3
	b[3][7] = 5
	if n := b.clearLines(); n != 2 {
		t.Fatalf("cleared %d lines, want 2", n)
	}
	if b[0][4] != 3 || b[1][7] != 5 {
		t.Errorf("rows did not drop correctly: row0=%v row1=%v", b[0], b[1])
	}
	for y := 2; y < Height; y++ {
		if b[y] != [Width]uint8{} {
			t.Errorf("row %d should be empty", y)
		}
	}
}

func TestLineClearReward(t *testing.T) {
	g := New(1)
	for x := 0; x < 6; x++ {
		g.board[0][x] = 1
	}
	g.cur = Piece{I, 0, 6, 18}
	reward, done := g.Step(HardDrop)
	if reward != 100 || done {
		t.Fatalf("reward=%d done=%v, want 100 false", reward, done)
	}
	s := g.State()
	if s.Lines != 1 || s.Score != 100 || s.Pieces != 1 {
		t.Errorf("state = %+v", s)
	}
	if s.Board[0] != [Width]uint8{} {
		t.Errorf("bottom row should be cleared: %v", s.Board[0])
	}
}

func TestBag(t *testing.T) {
	b := newBag(42)
	for round := 0; round < 10; round++ {
		seen := map[Kind]bool{}
		for i := 0; i < int(NumKinds); i++ {
			seen[b.pop()] = true
		}
		if len(seen) != int(NumKinds) {
			t.Fatalf("round %d: bag had %d distinct kinds, want %d", round, len(seen), NumKinds)
		}
	}
}

func TestDeterminism(t *testing.T) {
	run := func(seed uint64) State {
		g := New(seed)
		r := rand.New(rand.NewPCG(7, 7))
		for i := 0; i < 500; i++ {
			g.Step(Action(r.IntN(int(NumActions))))
		}
		return g.State()
	}
	if run(5) != run(5) {
		t.Error("same seed and actions produced different states")
	}
	a, b := New(1), New(2)
	same := true
	for i := 0; i < 20 && same; i++ {
		a.Step(HardDrop)
		b.Step(HardDrop)
		same = a.State().Piece.Kind == b.State().Piece.Kind
	}
	if same {
		t.Error("different seeds produced the same piece sequence")
	}
}

func TestResetRestarts(t *testing.T) {
	g := New(3)
	first := g.State()
	for i := 0; i < 30; i++ {
		g.Step(HardDrop)
	}
	if g.Reset(3) != first {
		t.Error("Reset with the same seed did not restore the initial state")
	}
}

func TestFullGameHeadless(t *testing.T) {
	g := New(1)
	r := rand.New(rand.NewPCG(1, 1))
	steps := 0
	for ; steps < 1_000_000; steps++ {
		if _, done := g.Step(Action(r.IntN(int(NumActions)))); done {
			break
		}
	}
	s := g.State()
	if !s.Done || s.Pieces == 0 {
		t.Fatalf("game did not finish: %+v after %d steps", s, steps)
	}
	if reward, done := g.Step(Left); reward != 0 || !done {
		t.Errorf("stepping a finished game returned %d, %v", reward, done)
	}
}

func TestStackingEndsGame(t *testing.T) {
	g := New(1)
	for i := 0; i < 100; i++ {
		if _, done := g.Step(HardDrop); done {
			return
		}
	}
	t.Error("dropping every piece in the same column should end the game")
}
