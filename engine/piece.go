package engine

// Kind is a tetromino type.
type Kind uint8

const (
	I Kind = iota
	O
	T
	S
	Z
	J
	L
	NumKinds
)

// Point is a cell offset or position. X grows right, Y grows up (row 0 is the bottom).
type Point struct{ X, Y int }

// Piece is a tetromino in play. X, Y is the bottom-left corner of its
// bounding box (3x3, or 4x4 for I and O) and Rot is the SRS rotation state:
// 0 = spawn, 1 = clockwise, 2 = 180°, 3 = counter-clockwise.
type Piece struct {
	Kind Kind
	Rot  int
	X, Y int
}

// Cells returns the absolute board coordinates of the piece's four cells.
func (p Piece) Cells() [4]Point {
	var out [4]Point
	for i, c := range shapes[p.Kind][p.Rot] {
		out[i] = Point{p.X + c.X, p.Y + c.Y}
	}
	return out
}

// boxSize returns the side length of the piece's SRS bounding box.
func boxSize(k Kind) int {
	if k == I || k == O {
		return 4
	}
	return 3
}

// spawnBases are the spawn-state shapes, top row first.
var spawnBases = [NumKinds][]string{
	I: {"....", "####", "....", "...."},
	O: {".##.", ".##.", "....", "...."},
	T: {".#.", "###", "..."},
	S: {".##", "##.", "..."},
	Z: {"##.", ".##", "..."},
	J: {"#..", "###", "..."},
	L: {"..#", "###", "..."},
}

// shapes[kind][rot] holds cell offsets relative to the bounding box's bottom-left.
var shapes = buildShapes()

func buildShapes() (out [NumKinds][4][4]Point) {
	for k := Kind(0); k < NumKinds; k++ {
		n := boxSize(k)
		grid := make([][]bool, n)
		for r, row := range spawnBases[k] {
			grid[r] = make([]bool, n)
			for c, ch := range row {
				grid[r][c] = ch == '#'
			}
		}
		for rot := 0; rot < 4; rot++ {
			i := 0
			for r := 0; r < n; r++ {
				for c := 0; c < n; c++ {
					if grid[r][c] {
						out[k][rot][i] = Point{c, n - 1 - r}
						i++
					}
				}
			}
			if k == O { // O never changes shape or position under rotation
				continue
			}
			rotated := make([][]bool, n)
			for r := range rotated {
				rotated[r] = make([]bool, n)
				for c := range rotated[r] {
					rotated[r][c] = grid[n-1-c][r] // clockwise
				}
			}
			grid = rotated
		}
	}
	return out
}

// SRS wall kick tests, keyed by [from, to] rotation state. The first test is
// always (0, 0). Offsets use Y up, matching the SRS reference tables.
var kicksJLSTZ = map[[2]int][5]Point{
	{0, 1}: {{0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2}},
	{1, 0}: {{0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2}},
	{1, 2}: {{0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2}},
	{2, 1}: {{0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2}},
	{2, 3}: {{0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2}},
	{3, 2}: {{0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2}},
	{3, 0}: {{0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2}},
	{0, 3}: {{0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2}},
}

var kicksI = map[[2]int][5]Point{
	{0, 1}: {{0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2}},
	{1, 0}: {{0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2}},
	{1, 2}: {{0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1}},
	{2, 1}: {{0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1}},
	{2, 3}: {{0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2}},
	{3, 2}: {{0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2}},
	{3, 0}: {{0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1}},
	{0, 3}: {{0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1}},
}
