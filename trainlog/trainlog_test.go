package trainlog

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

const agentJSON = `{"name":"x","scale":[1,1,1,1],"layers":[{"w":[[0,0,0,0]],"b":[0]}]}`

func writeRun(t *testing.T, dir, metrics string, snapshotSteps ...string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(dir, "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "metrics.jsonl"), []byte(metrics), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, s := range snapshotSteps {
		if err := os.WriteFile(filepath.Join(dir, "snapshots", "step_"+s+".json"), []byte(agentJSON), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func appendFile(t *testing.T, path, text string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	f.WriteString(text)
}

func TestLoadAllAndLive(t *testing.T) {
	root := t.TempDir()
	run := filepath.Join(root, "a")
	writeRun(t, run, `{"t":"episode","step":10,"score":100,"lines":1,"pieces":30,"epsilon":0.9}`+"\n"+
		`{"t":"loss","step":100,"value":0.5}`+"\n", "000000000")
	if err := os.MkdirAll(filepath.Join(root, "not-a-run"), 0o755); err != nil {
		t.Fatal(err)
	}

	runs, err := LoadAll(root)
	if err != nil || len(runs) != 1 {
		t.Fatalf("runs=%d err=%v", len(runs), err)
	}
	r := runs[0]
	if len(r.Episodes) != 1 || len(r.Losses) != 1 || len(r.Snapshots) != 1 || r.Done || r.LastStep() != 100 {
		t.Fatalf("unexpected run %+v", r)
	}

	// The trainer appends a full line, a half-written line, and a new snapshot.
	appendFile(t, filepath.Join(run, "metrics.jsonl"), `{"t":"validation","step":200,"score":5,"lines":1,"pieces":9}`+"\n"+`{"t":"epis`)
	os.WriteFile(filepath.Join(run, "snapshots", "step_000000200.json"), []byte(agentJSON), 0o644)
	changed, err := r.Refresh()
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
	if len(r.Validations) != 1 || len(r.Episodes) != 1 || len(r.Snapshots) != 2 || r.Snapshots[1].Step != 200 {
		t.Fatalf("after refresh %+v", r)
	}

	// The partial line is completed later and then picked up exactly once.
	appendFile(t, filepath.Join(run, "metrics.jsonl"), `ode","step":300,"score":7,"lines":0,"pieces":12,"epsilon":0.5}`+"\n"+`{"t":"done","step":300}`+"\n")
	if _, err := r.Refresh(); err != nil {
		t.Fatal(err)
	}
	if len(r.Episodes) != 2 || !r.Done {
		t.Fatalf("after completing the line %+v", r)
	}
	if changed, _ := r.Refresh(); changed {
		t.Error("nothing new was written, Refresh should report no change")
	}
}

func TestBadMetricsLine(t *testing.T) {
	root := t.TempDir()
	writeRun(t, filepath.Join(root, "bad"), "not json\n")
	if _, err := LoadAll(root); err == nil {
		t.Error("expected an error for a malformed metrics line")
	}
}

func TestSmooth(t *testing.T) {
	got := Smooth([]float64{2, 4, 6, 8}, 2)
	want := []float64{2, 3, 5, 7}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}
