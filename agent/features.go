package agent

import (
	"github.com/nweber23/tetri_ml/engine"
	"github.com/nweber23/tetri_ml/heuristic"
)

// NumMoveFeatures is the input size of a move agent:
//
//	[0:4]     aggregate height, holes, bumpiness, lines cleared by the move
//	[4:204]   the 20 visible rows after the move, row 0 first, 1 = filled
//	[204:211] one-hot of the held piece after the move, all zero if none
//	[211:218] one-hot of the piece that comes next, all zero if unknown
//
// Every value fits a byte, which is how the server sends them.
const NumMoveFeatures = 4 + engine.VisibleHeight*engine.Width + 7 + 7

// MoveFeatures describes the state after playing m from g. After a hold from an
// empty slot the next piece is not known yet, so its one-hot stays zero.
func MoveFeatures(g *engine.Game, m engine.Move) [NumMoveFeatures]uint8 {
	st := g.State()
	var out [NumMoveFeatures]uint8
	f := heuristic.Extract(m.Board, m.Lines)
	out[0], out[1], out[2], out[3] = uint8(f.AggregateHeight), uint8(f.Holes), uint8(f.Bumpiness), uint8(f.Lines)
	for y := 0; y < engine.VisibleHeight; y++ {
		for x := 0; x < engine.Width; x++ {
			if m.Board[y][x] != 0 {
				out[4+y*engine.Width+x] = 1
			}
		}
	}
	held, hasHeld, upcoming, known := st.Hold, st.HasHold, st.Next, true
	if m.Hold {
		held, hasHeld = st.Piece.Kind, true
		known = st.HasHold
		if known {
			upcoming = st.Next
		}
	}
	const holdAt, nextAt = 4 + engine.VisibleHeight*engine.Width, 4 + engine.VisibleHeight*engine.Width + 7
	if hasHeld {
		out[holdAt+int(held)] = 1
	}
	if known {
		out[nextAt+int(upcoming)] = 1
	}
	return out
}
