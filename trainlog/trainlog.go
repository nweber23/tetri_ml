// Package trainlog reads the metrics and agent snapshots that training/viewlog.py writes,
// including while a run is still in progress.
package trainlog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/nweber23/tetri_ml/agent"
)

type Episode struct {
	Step                          int
	Score, Lines, Pieces, Epsilon float64
}

type Point struct {
	Step  int
	Value float64
}

type Validation struct {
	Step                 int
	Score, Lines, Pieces float64
}

// Snapshot is the network at one training step.
type Snapshot struct {
	Step  int
	Agent *agent.Agent
}

type Run struct {
	Name        string
	Episodes    []Episode
	Losses      []Point
	Validations []Validation
	Snapshots   []Snapshot // ordered by step
	Done        bool       // the training process wrote its final line
	Updated     time.Time  // when Refresh last found new data

	dir    string
	offset int64
}

// LoadAll returns every run under dir (subdirectories holding metrics.jsonl), ordered by name.
func LoadAll(dir string) ([]*Run, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var runs []*Run
	for _, e := range entries {
		if _, err := os.Stat(filepath.Join(dir, e.Name(), "metrics.jsonl")); err != nil {
			continue
		}
		r := &Run{Name: e.Name(), dir: filepath.Join(dir, e.Name())}
		if _, err := r.Refresh(); err != nil {
			return nil, err
		}
		runs = append(runs, r)
	}
	return runs, nil
}

// LastStep is the latest training step seen in the metrics.
func (r *Run) LastStep() int {
	last := 0
	if n := len(r.Episodes); n > 0 {
		last = r.Episodes[n-1].Step
	}
	if n := len(r.Losses); n > 0 {
		last = max(last, r.Losses[n-1].Step)
	}
	return last
}

type record struct {
	T       string  `json:"t"`
	Step    int     `json:"step"`
	Score   float64 `json:"score"`
	Lines   float64 `json:"lines"`
	Pieces  float64 `json:"pieces"`
	Epsilon float64 `json:"epsilon"`
	Value   float64 `json:"value"`
}

// Refresh reads whatever training has written since the last call and reports whether anything was new.
func (r *Run) Refresh() (bool, error) {
	changed, err := r.readMetrics()
	if err != nil {
		return false, err
	}
	snaps, err := r.readSnapshots()
	if err != nil {
		return false, err
	}
	if changed || snaps {
		r.Updated = time.Now()
	}
	return changed || snaps, nil
}

func (r *Run) readMetrics() (bool, error) {
	f, err := os.Open(filepath.Join(r.dir, "metrics.jsonl"))
	if err != nil {
		return false, err
	}
	defer f.Close()
	if _, err := f.Seek(r.offset, io.SeekStart); err != nil {
		return false, err
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return false, err
	}
	end := bytes.LastIndexByte(data, '\n') + 1 // ignore a line the trainer is still writing
	data = data[:end]
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		var rec record
		if err := json.Unmarshal(line, &rec); err != nil {
			return false, fmt.Errorf("%s: %w", r.Name, err)
		}
		switch rec.T {
		case "episode":
			r.Episodes = append(r.Episodes, Episode{rec.Step, rec.Score, rec.Lines, rec.Pieces, rec.Epsilon})
		case "loss":
			r.Losses = append(r.Losses, Point{rec.Step, rec.Value})
		case "validation":
			r.Validations = append(r.Validations, Validation{rec.Step, rec.Score, rec.Lines, rec.Pieces})
		case "done":
			r.Done = true
		}
	}
	r.offset += int64(end)
	return end > 0, nil
}

func (r *Run) readSnapshots() (bool, error) {
	files, err := filepath.Glob(filepath.Join(r.dir, "snapshots", "step_*.json"))
	if err != nil {
		return false, err
	}
	known := map[int]bool{}
	for _, s := range r.Snapshots {
		known[s.Step] = true
	}
	added := false
	for _, f := range files {
		step, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(filepath.Base(f), "step_"), ".json"))
		if err != nil {
			return false, fmt.Errorf("%s: bad snapshot name %q", r.Name, f)
		}
		if known[step] {
			continue
		}
		a, err := agent.Load(f)
		if err != nil {
			return false, err
		}
		r.Snapshots = append(r.Snapshots, Snapshot{step, a})
		added = true
	}
	sort.Slice(r.Snapshots, func(i, j int) bool { return r.Snapshots[i].Step < r.Snapshots[j].Step })
	return added, nil
}

// Smooth returns the trailing moving average of v over window points.
func Smooth(v []float64, window int) []float64 {
	out := make([]float64, len(v))
	sum := 0.0
	for i, x := range v {
		sum += x
		if i >= window {
			sum -= v[i-window]
		}
		out[i] = sum / float64(min(i+1, window))
	}
	return out
}
