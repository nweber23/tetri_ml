package server

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"github.com/nweber23/tetri_ml/agent"
	pb "github.com/nweber23/tetri_ml/proto/tetris/v1"
)

func play(t *testing.T) pb.Tetris_PlayClient {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := grpc.NewServer()
	pb.RegisterTetrisServer(s, &Server{})
	go s.Serve(lis)
	t.Cleanup(s.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	stream, err := pb.NewTetrisClient(conn).Play(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func reset(seed uint64) *pb.Request {
	return &pb.Request{Action: &pb.Request_Start{Start: &pb.Reset{Seed: seed}}}
}

func place(p *pb.Placement) *pb.Request {
	return &pb.Request{Action: &pb.Request_Place{Place: p}}
}

func TestPlayFullGame(t *testing.T) {
	stream := play(t)
	stream.Send(reset(1))
	res, err := stream.Recv()
	if err != nil || res.State.Done || len(res.Placements) == 0 {
		t.Fatalf("reset: %v %+v", err, res)
	}
	if len(res.Afterstates) != len(res.Placements) {
		t.Fatalf("%d afterstates for %d placements", len(res.Afterstates), len(res.Placements))
	}
	if len(res.State.Board) != 24*10 {
		t.Fatalf("board has %d bytes", len(res.State.Board))
	}
	for i := 0; !res.State.Done; i++ {
		if err := stream.Send(place(res.Placements[0])); err != nil {
			t.Fatal(err)
		}
		if res, err = stream.Recv(); err != nil {
			t.Fatal(err)
		}
		if i > 10000 {
			t.Fatal("game did not end")
		}
	}
	if len(res.Placements) != 0 || res.State.Pieces == 0 {
		t.Errorf("final response %+v", res)
	}
}

func TestPlaceBeforeReset(t *testing.T) {
	stream := play(t)
	stream.Send(place(&pb.Placement{}))
	if _, err := stream.Recv(); status.Code(err) != codes.FailedPrecondition {
		t.Errorf("got %v, want FailedPrecondition", err)
	}
}

func TestInvalidPlacement(t *testing.T) {
	stream := play(t)
	stream.Send(reset(1))
	stream.Recv()
	stream.Send(place(&pb.Placement{Rot: 0, X: 99}))
	if _, err := stream.Recv(); status.Code(err) != codes.InvalidArgument {
		t.Errorf("got %v, want InvalidArgument", err)
	}
}

func TestPlayMoves(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := grpc.NewServer()
	pb.RegisterTetrisServer(s, &Server{})
	go s.Serve(lis)
	t.Cleanup(s.Stop)
	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	stream, err := pb.NewTetrisClient(conn).PlayMoves(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	stream.Send(&pb.MoveRequest{Action: &pb.MoveRequest_Start{Start: &pb.Reset{Seed: 1}}})
	res, err := stream.Recv()
	if err != nil || len(res.Moves) == 0 || len(res.Moves) > MaxMoves {
		t.Fatalf("reset: %v, %d moves", err, len(res.Moves))
	}
	for _, m := range res.Moves {
		if len(m.Features) != agent.NumMoveFeatures {
			t.Fatalf("%d features, want %d", len(m.Features), agent.NumMoveFeatures)
		}
	}
	for i := 0; i < 200 && !res.State.Done; i++ { // play the last move each time, including holds and slides
		stream.Send(&pb.MoveRequest{Action: &pb.MoveRequest_Move{Move: int32(len(res.Moves) - 1)}})
		if res, err = stream.Recv(); err != nil {
			t.Fatal(err)
		}
	}
	stream.Send(&pb.MoveRequest{Action: &pb.MoveRequest_Move{Move: int32(len(res.Moves))}})
	if _, err := stream.Recv(); status.Code(err) != codes.InvalidArgument {
		t.Errorf("out-of-range move: %v, want InvalidArgument", err)
	}
}
