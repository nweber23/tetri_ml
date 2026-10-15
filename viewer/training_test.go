package viewer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/nweber23/tetri_ml/agent"
	"github.com/nweber23/tetri_ml/trainlog"
)

const flatAgent = `{"name":"x","scale":[1,1,1,1],"layers":[{"w":[[-1,-5,-1,3]],"b":[0]}]}`

func makeRun(t *testing.T, root, name string, done bool, steps ...string) {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Join(dir, "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	metrics := `{"t":"episode","step":10,"score":1,"lines":0,"pieces":20,"epsilon":0.9}` + "\n"
	if done {
		metrics += `{"t":"done","step":10}` + "\n"
	}
	if err := os.WriteFile(filepath.Join(dir, "metrics.jsonl"), []byte(metrics), 0o644); err != nil {
		t.Fatal(err)
	}
	addSnapshots(t, dir, steps...)
}

func addSnapshots(t *testing.T, dir string, steps ...string) {
	t.Helper()
	for _, st := range steps {
		if err := os.WriteFile(filepath.Join(dir, "snapshots", "step_"+st+".json"), []byte(flatAgent), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func trainingSession(t *testing.T, root string) *Session {
	t.Helper()
	runs, err := trainlog.LoadAll(root)
	if err != nil {
		t.Fatal(err)
	}
	s := New(1, []*agent.Agent{keepFlat()}, nil)
	s.SetRuns(runs)
	return s
}

func TestTrainingModeSwapsAgents(t *testing.T) {
	root := t.TempDir()
	makeRun(t, root, "a", true, "000000000", "000005000", "000010000")
	s := trainingSession(t, root)
	s.SetMode(Training)
	if s.Mode != Training || len(s.Agents) != 3 || s.AgentIdx != 2 || s.Follow || s.SnapshotStep() != 10000 {
		t.Fatalf("mode %v agents %d idx %d follow %v step %d", s.Mode, len(s.Agents), s.AgentIdx, s.Follow, s.SnapshotStep())
	}
	s.SelectAgent(-1)
	if s.SnapshotStep() != 5000 {
		t.Errorf("step %d, want 5000", s.SnapshotStep())
	}
	s.SetMode(AgentPlay)
	if len(s.Agents) != 1 || s.AgentIdx != 0 || s.SnapshotStep() != -1 {
		t.Errorf("leaving training should restore the startup agents, got %d agents idx %d", len(s.Agents), s.AgentIdx)
	}
}

func TestFollowLiveRun(t *testing.T) {
	root := t.TempDir()
	makeRun(t, root, "live", false, "000000000")
	s := trainingSession(t, root)
	s.SetMode(Training)
	if !s.Follow || s.AgentIdx != 0 {
		t.Fatalf("a live run should be followed, follow=%v idx=%d", s.Follow, s.AgentIdx)
	}
	addSnapshots(t, filepath.Join(root, "live"), "000005000")
	if err := s.RefreshRuns(); err != nil {
		t.Fatal(err)
	}
	if s.AgentIdx != 1 || s.SnapshotStep() != 5000 {
		t.Errorf("following should jump to the new snapshot, idx=%d step=%d", s.AgentIdx, s.SnapshotStep())
	}

	s.SelectAgent(-1) // the user steps back by hand: stop following
	s.Update(stepInterval, Input{})
	pieces := s.Game.State().Pieces
	addSnapshots(t, filepath.Join(root, "live"), "000010000")
	if err := s.RefreshRuns(); err != nil {
		t.Fatal(err)
	}
	if s.Follow || s.SnapshotStep() != 0 || len(s.Agents) != 3 {
		t.Errorf("follow=%v step=%d agents=%d, want the viewer to stay on step 0 with the new snapshot available", s.Follow, s.SnapshotStep(), len(s.Agents))
	}
	if s.Game.State().Pieces != pieces {
		t.Error("a new snapshot restarted a game that was not following")
	}
}

func TestSelectRun(t *testing.T) {
	root := t.TempDir()
	makeRun(t, root, "a", true, "000000000")
	makeRun(t, root, "b", true, "000000000", "000005000")
	s := trainingSession(t, root)
	s.SetMode(Training)
	s.SelectRun(1)
	if s.Run().Name != "b" || len(s.Agents) != 2 {
		t.Errorf("run %s agents %d", s.Run().Name, len(s.Agents))
	}
	s.SelectRun(1)
	if s.Run().Name != "a" {
		t.Errorf("run %s, want wrap-around to a", s.Run().Name)
	}
}

func TestTrainingNeedsRuns(t *testing.T) {
	s := New(1, nil, nil)
	s.SetMode(Training)
	if s.Mode != Human {
		t.Errorf("mode %v, want human without runs", s.Mode)
	}
}

func TestSelectSnapshot(t *testing.T) {
	root := t.TempDir()
	makeRun(t, root, "a", true, "000000000", "000005000", "000010000")
	s := trainingSession(t, root)
	s.SetMode(Training)
	for _, c := range []struct{ ask, want int }{{0, 0}, {7000, 5000}, {10000, 10000}, {999999, 10000}} {
		s.SelectSnapshot(c.ask)
		if got := s.SnapshotStep(); got != c.want {
			t.Errorf("SelectSnapshot(%d) = %d, want %d", c.ask, got, c.want)
		}
	}
}
