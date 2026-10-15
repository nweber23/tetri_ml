// Command server serves the Tetris engine over gRPC.
package main

import (
	"flag"
	"log"
	"net"

	"google.golang.org/grpc"

	pb "github.com/nweber23/tetri_ml/proto/tetris/v1"
	"github.com/nweber23/tetri_ml/server"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:50051", "listen address; keep it on loopback, the server has no authentication")
	flag.Parse()

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatal(err)
	}
	s := grpc.NewServer()
	pb.RegisterTetrisServer(s, &server.Server{})
	log.Printf("listening on %s", lis.Addr())
	log.Fatal(s.Serve(lis))
}
