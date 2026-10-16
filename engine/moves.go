package engine

// Move is one way to lock a piece: the final resting position and the keys
// that get there. Unlike a Placement it can use slides, tucks and spins, and
// it can swap in the held piece first.
type Move struct {
	Hold   bool     // swap with the held (or next) piece before the keys
	Piece  Piece    // where it locks
	TSpin  bool     // the lock scores as a T-spin
	Keys   []Action // Left, Right, RotateCW, RotateCCW and SoftDrop, in order
	Board  Board    // the board after locking and clearing lines
	Lines  int      // lines cleared
	Reward int      // score gained
}

type searchState struct {
	p    Piece
	spun bool // the last key was a rotation
}

// Moves lists every distinct way to lock the current piece, plus, if hold is
// available, every way to lock the piece a hold would bring in. A piece may be
// steered freely (no gravity, no lock delay) as long as each key is legal, so
// the list contains everything a player with enough time could do. It is empty
// once the game is over.
func (g *Game) Moves() []Move {
	if g.done {
		return nil
	}
	moves := g.search(g.cur, false)
	if !g.holdUsed {
		k := g.queue[0]
		if g.hasHold {
			k = g.hold
		}
		if p := spawnPiece(k); g.board.fits(p) {
			moves = append(moves, g.search(p, true)...)
		}
	}
	return moves
}

// search finds the resting positions reachable from start with a breadth-first search over keys.
func (g *Game) search(start Piece, hold bool) []Move {
	type node struct {
		parent int
		key    Action
		searchState
	}
	var seen [4][Width + 8][Height + 8][2]bool // rot, x+4, y+4, spun (boxes poke out below and left of the board)
	mark := func(n searchState) bool {
		spun := 0
		if n.spun {
			spun = 1
		}
		old := &seen[n.p.Rot][n.p.X+4][n.p.Y+4][spun]
		was := *old
		*old = true
		return !was
	}
	mark(searchState{p: start})
	nodes := []node{{parent: -1, searchState: searchState{p: start}}}
	dupes := map[[6]Point]bool{}
	var moves []Move
	for i := 0; i < len(nodes); i++ {
		st := nodes[i].searchState
		visit := func(key Action, p Piece, spun bool) {
			if n := (searchState{p, spun}); mark(n) {
				nodes = append(nodes, node{i, key, n})
			}
		}
		if p := (Piece{st.p.Kind, st.p.Rot, st.p.X - 1, st.p.Y}); g.board.fits(p) {
			visit(Left, p, false)
		}
		if p := (Piece{st.p.Kind, st.p.Rot, st.p.X + 1, st.p.Y}); g.board.fits(p) {
			visit(Right, p, false)
		}
		if p, ok := g.board.rotate(st.p, 1); ok {
			visit(RotateCW, p, true)
		}
		if p, ok := g.board.rotate(st.p, 3); ok {
			visit(RotateCCW, p, true)
		}
		down := st.p
		down.Y--
		if g.board.fits(down) {
			visit(SoftDrop, down, false)
			continue
		}
		// grounded: a place to lock
		tSpin := st.p.Kind == T && st.spun && g.board.tCorners(st.p) >= 3
		if id := cellsID(st.p, tSpin); !dupes[id] {
			dupes[id] = true
			m := Move{Hold: hold, Piece: st.p, TSpin: tSpin, Board: g.board}
			m.Board.place(st.p)
			m.Lines = m.Board.clearLines()
			m.Reward = lineScores[m.Lines]
			if tSpin {
				m.Reward = tSpinScores[m.Lines]
			}
			for n := i; nodes[n].parent >= 0; n = nodes[n].parent {
				m.Keys = append([]Action{nodes[n].key}, m.Keys...)
			}
			moves = append(moves, m)
		}
	}
	return moves
}

// cellsID identifies a lock by the cells it fills, so S, Z and I rotations that
// cover the same cells count once, and by whether it is a T-spin.
func cellsID(p Piece, tSpin bool) [6]Point {
	c := p.Cells()
	// insertion sort, four cells
	for i := 1; i < 4; i++ {
		for j := i; j > 0 && (c[j].Y < c[j-1].Y || c[j].Y == c[j-1].Y && c[j].X < c[j-1].X); j-- {
			c[j], c[j-1] = c[j-1], c[j]
		}
	}
	id := [6]Point{c[0], c[1], c[2], c[3]}
	if tSpin {
		id[4].X = 1
	}
	return id
}

// Do plays m: hold if it says so, then its keys, then lock. It returns the
// reward and whether the game is over. m must come from Moves for the current state.
func (g *Game) Do(m Move) (reward int, done bool) {
	if g.done {
		return 0, true
	}
	if m.Hold {
		g.Hold()
	}
	for _, k := range m.Keys {
		g.Apply(k)
	}
	return g.Lock(), g.done
}
