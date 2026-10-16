package viewer

import (
	"testing"
	"time"

	"github.com/nweber23/tetri_ml/agent"
	"github.com/nweber23/tetri_ml/replay"
)

// keepFlat prefers few holes and low, flat stacks.
func keepFlat() *agent.Agent {
	return &agent.Agent{Name: "flat", Scale: []float64{1, 1, 1, 1},
		Layers: []agent.Layer{{W: [][]float64{{-1, -5, -1, 3}}, B: []float64{0}}}}
}

func TestHumanLeftDoesNotDrop(t *testing.T) {
	s := New(1, nil, nil)
	y := s.Game.State().Piece.Y
	s.Update(time.Millisecond, Input{Left: true})
	if p := s.Game.State().Piece; p.Y != y || p.X != 2 {
		t.Errorf("piece %+v, want x=2 y=%d", p, y)
	}
	s.Update(gravityInterval, Input{})
	if s.Game.State().Piece.Y != y-1 {
		t.Error("gravity did not move the piece down after the interval")
	}
}

func TestAgentPlaysAndReplayMatches(t *testing.T) {
	s := New(3, []*agent.Agent{keepFlat()}, nil)
	if s.Mode != AgentPlay || s.BestIdx < 0 {
		t.Fatalf("mode %v best %d", s.Mode, s.BestIdx)
	}
	for i := 0; i < 50; i++ {
		s.Update(stepInterval, Input{})
	}
	if len(s.Played) != 50 || s.Game.State().Pieces != 50 {
		t.Fatalf("played %d, pieces %d", len(s.Played), s.Game.State().Pieces)
	}
	path, err := s.SaveReplay(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r, err := replay.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	want := s.Game.State()

	v := New(99, nil, &r)
	v.SetMode(Replaying)
	for i := 0; i < 50; i++ {
		v.Update(stepInterval, Input{})
	}
	if v.Game.State() != want {
		t.Error("replay ended in a different state than the recorded game")
	}
}

func TestSelectAgentWrapsAndKeepsSeed(t *testing.T) {
	s := New(5, []*agent.Agent{keepFlat(), keepFlat()}, nil)
	s.SelectAgent(-1)
	if s.AgentIdx != 1 || s.Seed != 5 {
		t.Errorf("idx %d seed %d", s.AgentIdx, s.Seed)
	}
	s.SelectAgent(1)
	if s.AgentIdx != 0 {
		t.Errorf("idx %d", s.AgentIdx)
	}
}

func TestSetModeNeedsResources(t *testing.T) {
	s := New(1, nil, nil)
	s.SetMode(AgentPlay)
	s.SetMode(Replaying)
	if s.Mode != Human {
		t.Errorf("mode %v, want human without agents or replay", s.Mode)
	}
}

func TestSaveReplayOnlyForAgentGames(t *testing.T) {
	s := New(1, nil, nil)
	if _, err := s.SaveReplay(t.TempDir()); err == nil {
		t.Error("human games should not be saved")
	}
}

// ground drops the piece until it rests on the floor, without locking it.
func ground(s *Session) {
	for s.Game.SoftDrop() {
	}
}

func TestLockDelay(t *testing.T) {
	s := New(1, nil, nil)
	ground(s)
	s.Update(lockDelay-time.Millisecond, Input{})
	if s.Game.State().Pieces != 0 {
		t.Fatal("piece locked before the delay passed")
	}
	s.Update(2*time.Millisecond, Input{})
	if s.Game.State().Pieces != 1 {
		t.Fatal("piece did not lock after the delay")
	}
}

func TestLockDelayResetsOnMoveUpToCap(t *testing.T) {
	s := New(1, nil, nil)
	ground(s)
	for i := 0; i < maxLockResets; i++ {
		s.Update(lockDelay-time.Millisecond, Input{Left: i%2 == 0, Right: i%2 == 1})
		if s.Game.State().Pieces != 0 {
			t.Fatalf("locked despite a move %d", i)
		}
	}
	s.Update(lockDelay-time.Millisecond, Input{Left: true}) // no resets left, so the time adds up
	if s.Game.State().Pieces != 1 {
		t.Error("piece never locked after the resets ran out")
	}
}

// zeroMoveAgent is a move agent that values every state the same, so it plays the first move it finds.
func zeroMoveAgent() *agent.Agent {
	return &agent.Agent{Name: "zero", Scale: make([]float64, agent.NumMoveFeatures), // scales are replaced below
		Layers: []agent.Layer{{W: [][]float64{make([]float64, agent.NumMoveFeatures)}, B: []float64{0}}}}
}

func TestMoveAgentSteersKeyByKey(t *testing.T) {
	a := zeroMoveAgent()
	for i := range a.Scale {
		a.Scale[i] = 1
	}
	s := New(1, []*agent.Agent{a}, nil)
	if s.Plan == nil || len(s.Plan.Keys) == 0 {
		t.Fatalf("no plan: %+v", s.Plan)
	}
	s.Update(keyInterval, Input{})
	if s.Game.State().Pieces != 0 {
		t.Fatal("the piece locked after one key")
	}
	for i := 0; i < 2000 && !s.Game.State().Done; i++ { // it stacks up and dies, which is fine
		s.Update(keyInterval, Input{})
	}
	if s.Game.State().Pieces < 5 {
		t.Errorf("only %d pieces locked", s.Game.State().Pieces)
	}
}
