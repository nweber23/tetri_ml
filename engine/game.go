package engine

// Action is one input per Step.
type Action uint8

const (
	NoOp Action = iota
	Left
	Right
	RotateCW
	RotateCCW
	HardDrop
	NumActions
)

// lineScores is the reward for clearing 0..4 lines at once.
var lineScores = [5]int{0, 100, 300, 500, 800}

// State is a snapshot of the game, safe to keep after further Steps.
type State struct {
	Board  Board
	Piece  Piece
	Next   Kind
	Score  int
	Lines  int
	Pieces int // pieces locked so far
	Done   bool
}

// Game is a single deterministic Tetris game. Create one with New or Reset it with a seed.
type Game struct {
	board  Board
	cur    Piece
	next   Kind
	bag    *bag
	score  int
	lines  int
	pieces int
	done   bool
}

// New returns a game started with the given seed.
func New(seed uint64) *Game {
	g := &Game{}
	g.Reset(seed)
	return g
}

// Reset starts a fresh game. The same seed always yields the same piece sequence.
func (g *Game) Reset(seed uint64) State {
	*g = Game{bag: newBag(seed)}
	g.next = g.bag.pop()
	g.spawn()
	return g.State()
}

// State returns a snapshot of the current game.
func (g *Game) State() State {
	return State{g.board, g.cur, g.next, g.score, g.lines, g.pieces, g.done}
}

// Step applies the action, then one gravity tick: the piece falls one row, or
// locks if it cannot (there is no lock delay). HardDrop drops and locks
// immediately. Reward is the score gained by this step. Stepping a finished
// game does nothing.
func (g *Game) Step(a Action) (reward int, done bool) {
	if g.done {
		return 0, true
	}
	switch a {
	case Left:
		g.tryMove(-1, 0)
	case Right:
		g.tryMove(1, 0)
	case RotateCW:
		g.tryRotate(1)
	case RotateCCW:
		g.tryRotate(3)
	case HardDrop:
		for g.tryMove(0, -1) {
		}
		return g.lock(), g.done
	}
	if !g.tryMove(0, -1) {
		reward = g.lock()
	}
	return reward, g.done
}

func (g *Game) tryMove(dx, dy int) bool {
	p := g.cur
	p.X += dx
	p.Y += dy
	if !g.board.fits(p) {
		return false
	}
	g.cur = p
	return true
}

// tryRotate rotates by turns quarter turns clockwise, using the SRS kick tests.
func (g *Game) tryRotate(turns int) {
	if g.cur.Kind == O {
		return
	}
	to := (g.cur.Rot + turns) % 4
	kicks := kicksJLSTZ
	if g.cur.Kind == I {
		kicks = kicksI
	}
	for _, k := range kicks[[2]int{g.cur.Rot, to}] {
		p := Piece{g.cur.Kind, to, g.cur.X + k.X, g.cur.Y + k.Y}
		if g.board.fits(p) {
			g.cur = p
			return
		}
	}
}

// lock fixes the current piece, clears lines, spawns the next piece and
// returns the reward. The game ends on lock out (piece entirely above the
// visible field) or block out (the new piece overlaps the stack).
func (g *Game) lock() int {
	g.board.place(g.cur)
	lockOut := true
	for _, c := range g.cur.Cells() {
		if c.Y < VisibleHeight {
			lockOut = false
		}
	}
	cleared := g.board.clearLines()
	reward := lineScores[cleared]
	g.score += reward
	g.lines += cleared
	g.pieces++
	if lockOut {
		g.done = true
		return reward
	}
	g.spawn()
	return reward
}

// spawn puts the next piece in the hidden rows above the field, horizontally centered.
func (g *Game) spawn() {
	k := g.next
	g.next = g.bag.pop()
	g.cur = spawnPiece(k)
	if !g.board.fits(g.cur) {
		g.done = true
	}
}

func spawnPiece(k Kind) Piece {
	return Piece{Kind: k, X: 3, Y: Height - boxSize(k)}
}
