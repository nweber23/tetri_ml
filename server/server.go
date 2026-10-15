// Package server exposes the engine over gRPC, one game per Play stream.
package server

import (
	"io"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/nweber23/tetri_ml/engine"
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

func response(g *engine.Game, reward int) *pb.Response {
	s := g.State()
	board := make([]byte, 0, engine.Height*engine.Width)
	for y := range s.Board {
		board = append(board, s.Board[y][:]...)
	}
	res := &pb.Response{
		Reward: int32(reward),
		State: &pb.State{
			Board:  board,
			Piece:  int32(s.Piece.Kind),
			Next:   int32(s.Next),
			Score:  int32(s.Score),
			Lines:  int32(s.Lines),
			Pieces: int32(s.Pieces),
			Done:   s.Done,
		},
	}
	for _, pl := range g.Placements() {
		res.Placements = append(res.Placements, &pb.Placement{Rot: int32(pl.Rot), X: int32(pl.X)})
	}
	return res
}
