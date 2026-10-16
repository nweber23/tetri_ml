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
	SoftDrop // moves the piece down one row; Step does not apply gravity on top of it
	NumActions
)

// lineScores is the reward for clearing 0..4 lines at once.
var lineScores = [5]int{0, 100, 300, 500, 800}

// tSpinScores is the reward for a T-spin that clears 0..3 lines.
var tSpinScores = [4]int{400, 800, 1200, 1600}

// State is a snapshot of the game, safe to keep after further Steps.
type State struct {
	Board  Board
	Piece  Piece
	Next   Kind
	Score  int
	Lines  int
	Pieces int // pieces locked so far
	Done   bool

	Hold      Kind // the held piece, valid if HasHold
	HasHold   bool
	LastClear int  // lines cleared by the most recent lock
	TSpin     bool // whether the most recent lock was a T-spin
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

	hold              Kind
	hasHold, holdUsed bool
	lastRot           bool // the last successful move of the current piece was a rotation
	lastClear         int
	tSpin             bool
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
	return State{g.board, g.cur, g.next, g.score, g.lines, g.pieces, g.done, g.hold, g.hasHold, g.lastClear, g.tSpin}
}

// Step applies the action, then one gravity tick: the piece falls one row, or
// locks if it cannot (there is no lock delay). HardDrop drops and locks
// immediately. Reward is the score gained by this step. Stepping a finished
// game does nothing.
func (g *Game) Step(a Action) (reward int, done bool) {
	if g.done {
		return 0, true
	}
	reward, locked := g.Apply(a)
	if !locked && a != SoftDrop {
		reward = g.Gravity()
	}
	return reward, g.done
}

// Apply performs only the action, without gravity. It reports whether the
// piece was locked, which only HardDrop does.
func (g *Game) Apply(a Action) (reward int, locked bool) {
	if g.done {
		return 0, false
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
	case SoftDrop:
		g.tryMove(0, -1)
	case HardDrop:
		for g.tryMove(0, -1) {
		}
		return g.lock(), true
	}
	return 0, false
}

// SoftDrop moves the piece down one row without locking it. It reports whether it moved.
func (g *Game) SoftDrop() bool { return !g.done && g.tryMove(0, -1) }

// Grounded reports whether the piece is resting on the stack or the floor.
func (g *Game) Grounded() bool {
	p := g.cur
	p.Y--
	return !g.done && !g.board.fits(p)
}

// Lock fixes the piece where it is and returns the reward. Together with
// Grounded and SoftDrop it lets a caller implement a lock delay.
func (g *Game) Lock() int {
	if g.done {
		return 0
	}
	return g.lock()
}

// Hold swaps the current piece with the held one, or with the next piece if
// nothing is held. It is allowed once per piece and reports whether it happened.
func (g *Game) Hold() bool {
	if g.done || g.holdUsed {
		return false
	}
	k := g.cur.Kind
	if g.hasHold {
		g.cur = spawnPiece(g.hold)
	} else {
		g.cur = spawnPiece(g.next)
		g.next = g.bag.pop()
	}
	g.hold, g.hasHold, g.holdUsed, g.lastRot = k, true, true, false
	if !g.board.fits(g.cur) {
		g.done = true
	}
	return true
}

// Gravity performs one gravity tick and returns the reward if the piece locked.
func (g *Game) Gravity() int {
	if g.done || g.tryMove(0, -1) {
		return 0
	}
	return g.lock()
}

func (g *Game) tryMove(dx, dy int) bool {
	p := g.cur
	p.X += dx
	p.Y += dy
	if !g.board.fits(p) {
		return false
	}
	g.cur = p
	g.lastRot = false
	return true
}

// tryRotate rotates by turns quarter turns clockwise, using the SRS kick tests.
func (g *Game) tryRotate(turns int) {
	if p, ok := g.board.rotate(g.cur, turns); ok {
		g.cur = p
		g.lastRot = true
	}
}

// lock fixes the current piece, clears lines, spawns the next piece and
// returns the reward. The game ends on lock out (piece entirely above the
// visible field) or block out (the new piece overlaps the stack).
func (g *Game) lock() int {
	tSpin := g.cur.Kind == T && g.lastRot && g.board.tCorners(g.cur) >= 3
	g.board.place(g.cur)
	lockOut := true
	for _, c := range g.cur.Cells() {
		if c.Y < VisibleHeight {
			lockOut = false
		}
	}
	cleared := g.board.clearLines()
	reward := lineScores[cleared]
	if tSpin {
		reward = tSpinScores[cleared]
	}
	g.lastClear, g.tSpin, g.holdUsed = cleared, tSpin, false
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
