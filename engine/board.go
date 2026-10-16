package engine

const (
	Width         = 10
	VisibleHeight = 20
	// Height includes hidden rows above the visible field where pieces spawn.
	Height = VisibleHeight + 4
)

// Board holds locked cells, indexed [y][x] with y = 0 at the bottom.
// A cell is 0 when empty, otherwise the Kind that filled it plus one.
type Board [Height][Width]uint8

// fits reports whether every cell of p is inside the board and empty.
func (b *Board) fits(p Piece) bool {
	for _, c := range p.Cells() {
		if c.X < 0 || c.X >= Width || c.Y < 0 || c.Y >= Height || b[c.Y][c.X] != 0 {
			return false
		}
	}
	return true
}

func (b *Board) place(p Piece) {
	for _, c := range p.Cells() {
		b[c.Y][c.X] = uint8(p.Kind) + 1
	}
}

// clearLines removes full rows, drops everything above them, and returns how many were removed.
func (b *Board) clearLines() int {
	cleared := 0
	for y := 0; y < Height; y++ {
		full := true
		for x := 0; x < Width; x++ {
			if b[y][x] == 0 {
				full = false
				break
			}
		}
		if full {
			cleared++
			continue
		}
		b[y-cleared] = b[y]
	}
	for y := Height - cleared; y < Height; y++ {
		b[y] = [Width]uint8{}
	}
	return cleared
}

// tCorners counts the occupied corners (walls and floor count) of a T piece's 3x3 box.
func (b *Board) tCorners(p Piece) int {
	n := 0
	for _, c := range [4]Point{{0, 0}, {2, 0}, {0, 2}, {2, 2}} {
		x, y := p.X+c.X, p.Y+c.Y
		if x < 0 || x >= Width || y < 0 || y >= Height || b[y][x] != 0 {
			n++
		}
	}
	return n
}

// rotate turns p by turns quarter turns clockwise using the SRS kick tests.
// It reports false if the piece is an O or no kick fits.
func (b *Board) rotate(p Piece, turns int) (Piece, bool) {
	if p.Kind == O {
		return p, false
	}
	to := (p.Rot + turns) % 4
	kicks := kicksJLSTZ
	if p.Kind == I {
		kicks = kicksI
	}
	for _, k := range kicks[[2]int{p.Rot, to}] {
		if q := (Piece{p.Kind, to, p.X + k.X, p.Y + k.Y}); b.fits(q) {
			return q, true
		}
	}
	return p, false
}
