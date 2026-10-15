#!/usr/bin/env bash
# Regenerates the Go and Python gRPC stubs from proto/tetris/v1/tetris.proto.
# Needs protoc-gen-go and protoc-gen-go-grpc on PATH or in ~/go/bin, and the
# training venv (grpcio-tools bundles protoc).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:$HOME/go/bin:$HOME/go/bin/windows_amd64"
PY=training/.venv/Scripts/python
[ -x "$PY" ] || PY=training/.venv/bin/python

"$PY" -m grpc_tools.protoc -I proto \
  --go_out=proto --go_opt=paths=source_relative \
  --go-grpc_out=proto --go-grpc_opt=paths=source_relative \
  --python_out=training --grpc_python_out=training \
  proto/tetris/v1/tetris.proto
