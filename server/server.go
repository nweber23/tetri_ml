// Package server exposes the engine over gRPC, one game per Play stream.
package server

import (
	"io"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/nweber23/tetri_ml/agent"
	"github.com/nweber23/tetri_ml/engine"
	"github.com/nweber23/tetri_ml/heuristic"
	pb "github.com/nweber23/tetri_ml/proto/tetris/v1"
)

type Server struct {
	pb.UnimplementedTetrisServer
}

// Play serves one game session until the client closes the stream.
func (*Server) Play(stream pb.Tetris_PlayServer) error {
	var g *engine.Game
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		reward := 0
		switch a := req.Action.(type) {
		case *pb.Request_Start:
			if g == nil {
				g = engine.New(a.Start.Seed)
			} else {
				g.Reset(a.Start.Seed)
			}
		case *pb.Request_Place:
			if g == nil {
				return status.Error(codes.FailedPrecondition, "send Reset before Place")
			}
			pl := engine.Placement{Rot: int(a.Place.Rot), X: int(a.Place.X)}
			if !contains(g.Placements(), pl) {
				return status.Error(codes.InvalidArgument, "placement is not valid for the current piece")
			}
			reward, _ = g.Place(pl)
		default:
			return status.Error(codes.InvalidArgument, "request has no action")
		}

		if err := stream.Send(response(g, reward)); err != nil {
			return err
		}
	}
}

func contains(pls []engine.Placement, pl engine.Placement) bool {
	for _, p := range pls {
		if p == pl {
			return true
		}
	}
	return false
}

func stateMsg(s engine.State) *pb.State {
	board := make([]byte, 0, engine.Height*engine.Width)
	for y := range s.Board {
		board = append(board, s.Board[y][:]...)
	}
	return &pb.State{
		Board:  board,
		Piece:  int32(s.Piece.Kind),
		Next:   int32(s.Next),
		Score:  int32(s.Score),
		Lines:  int32(s.Lines),
		Pieces: int32(s.Pieces),
		Done:   s.Done,
	}
}

func response(g *engine.Game, reward int) *pb.Response {
	res := &pb.Response{Reward: int32(reward), State: stateMsg(g.State())}
	for _, pl := range g.Placements() {
		res.Placements = append(res.Placements, &pb.Placement{Rot: int32(pl.Rot), X: int32(pl.X)})
		b, lines := g.Afterstate(pl)
		f := heuristic.Extract(b, lines)
		res.Afterstates = append(res.Afterstates, &pb.Afterstate{
			AggregateHeight: int32(f.AggregateHeight),
			Holes:           int32(f.Holes),
			Bumpiness:       int32(f.Bumpiness),
			Lines:           int32(f.Lines),
		})
	}
	return res
}

// PlayMoves serves one full-game session: the client picks a move by index.
func (*Server) PlayMoves(stream pb.Tetris_PlayMovesServer) error {
	var g *engine.Game
	var moves []engine.Move
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		reward := 0
		switch a := req.Action.(type) {
		case *pb.MoveRequest_Start:
			if g == nil {
				g = engine.New(a.Start.Seed)
			} else {
				g.Reset(a.Start.Seed)
			}
		case *pb.MoveRequest_Move:
			if g == nil {
				return status.Error(codes.FailedPrecondition, "send Reset before Move")
			}
			if a.Move < 0 || int(a.Move) >= len(moves) {
				return status.Error(codes.InvalidArgument, "move index out of range")
			}
			reward, _ = g.Do(moves[a.Move])
		default:
			return status.Error(codes.InvalidArgument, "request has no action")
		}
		moves = g.Moves()
		if len(moves) > MaxMoves {
			moves = moves[:MaxMoves]
		}
		if err := stream.Send(moveResponse(g, moves, reward)); err != nil {
			return err
		}
	}
}

// MaxMoves caps the moves offered per state so clients can use a fixed-size action space.
// The search lists shortest key sequences first, and the most moves seen in play is about 70.
const MaxMoves = 128

func moveResponse(g *engine.Game, moves []engine.Move, reward int) *pb.MoveResponse {
	res := &pb.MoveResponse{State: stateMsg(g.State()), Reward: int32(reward)}
	for _, m := range moves {
		f := agent.MoveFeatures(g, m)
		res.Moves = append(res.Moves, &pb.Move{Hold: m.Hold, TSpin: m.TSpin, Lines: int32(m.Lines), Reward: int32(m.Reward), Features: f[:]})
	}
	return res
}
