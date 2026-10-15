// Package replay stores a finished game as its seed and placements; the engine is deterministic, so that is enough to replay it.
package replay

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/nweber23/tetri_ml/engine"
)

type Replay struct {
	Seed       uint64             `json:"seed"`
	Agent      string             `json:"agent"`
	Placements []engine.Placement `json:"placements"`
}

func Save(path string, r Replay) error {
	data, err := json.Marshal(r)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func Load(path string) (Replay, error) {
	var r Replay
	data, err := os.ReadFile(path)
	if err != nil {
		return r, err
	}
	if err := json.Unmarshal(data, &r); err != nil {
		return r, fmt.Errorf("%s: %w", path, err)
	}
	return r, nil
}
