package engine

// Placement is a final resting position for the current piece: a rotation
// state and the x of its bounding box's left edge. The piece reaches it by
// moving sideways in the hidden rows and then dropping straight down.
type Placement struct{ Rot, X int }

// numRots is the number of rotation states with distinct shapes.
func numRots(k Kind) int {
	switch k {
	case O:
		return 1
	case I, S, Z:
		return 2
	}
	return 4
}

// Placements lists every (rotation, column) the current piece can be dropped
// at. It is empty once the game is over.
func (g *Game) Placements() []Placement {
	if g.done {
		return nil
	}
	var out []Placement
	for rot := 0; rot < numRots(g.cur.Kind); rot++ {
		for x := -2; x < Width; x++ {
			p := spawnPiece(g.cur.Kind)
			p.Rot, p.X = rot, x
			if g.board.fits(p) {
				out = append(out, Placement{rot, x})
			}
		}
	}
	return out
}

// Place drops the current piece at pl and locks it. pl must come from
// Placements; anything else is ignored. Reward and done are as in Step.
func (g *Game) Place(pl Placement) (reward int, done bool) {
	p := spawnPiece(g.cur.Kind)
	p.Rot, p.X = pl.Rot, pl.X
	if g.done || !g.board.fits(p) {
		return 0, g.done
	}
	g.cur = p
	return g.Step(HardDrop)
}

// Landing returns where the current piece comes to rest when dropped at pl.
func (g *Game) Landing(pl Placement) Piece {
	p := spawnPiece(g.cur.Kind)
	p.Rot, p.X = pl.Rot, pl.X
	for {
		q := p
		q.Y--
		if !g.board.fits(q) {
			return p
		}
		p = q
	}
}

// Afterstate returns the board after dropping the current piece at pl and
// clearing lines, plus the number of lines cleared. The game is not changed.
func (g *Game) Afterstate(pl Placement) (Board, int) {
	b := g.board
	b.place(g.Landing(pl))
	return b, b.clearLines()
}
