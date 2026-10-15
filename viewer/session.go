// Package viewer holds the viewer's game logic: modes, timing, agent playback and replays.
// It does no drawing, so it can be tested without a window; cmd/viewer renders it.
package viewer

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/nweber23/tetri_ml/agent"
	"github.com/nweber23/tetri_ml/engine"
	"github.com/nweber23/tetri_ml/replay"
	"github.com/nweber23/tetri_ml/trainlog"
)

type Mode int

const (
	Human Mode = iota
	AgentPlay
	Replaying
	Training // agent playback over the snapshots of a training run
)

// NumModes is the number of modes, for cycling through them.
const NumModes = 4

func (m Mode) String() string {
	return [...]string{"human", "agent", "replay", "training"}[m]
}

const (
	gravityInterval = 500 * time.Millisecond
	stepInterval    = 300 * time.Millisecond // between agent or replay placements at speed 1
)

// Input is what the player did this frame; key repeat is handled by the caller.
type Input struct {
	Left, Right, RotateCW, RotateCCW, HardDrop bool
}

type Session struct {
	Game   *engine.Game
	Mode   Mode
	Seed   uint64
	Paused bool
	Speed  float64 // playback speed multiplier for agent and replay modes

	Agents   []*agent.Agent
	AgentIdx int
	Scores   []agent.Scored // the agent's value for each candidate placement of the current piece
	BestIdx  int            // index into Scores of the placement the agent will play, or -1

	Runs   []*trainlog.Run // training runs the viewer can follow
	RunIdx int
	Follow bool // in training mode, jump to each new snapshot as training writes it

	base []*agent.Agent // the agents loaded at startup; Agents is swapped for a run's snapshots in training mode

	Replay    *replay.Replay // loaded replay, if any
	ReplayPos int
	Played    []engine.Placement // placements of the current agent game, for saving

	gravityAcc, stepAcc time.Duration
}

// New starts a session in agent mode if there are agents, otherwise human mode.
// A non-nil r makes replay mode available.
func New(seed uint64, agents []*agent.Agent, r *replay.Replay) *Session {
	s := &Session{Game: engine.New(seed), Seed: seed, Agents: agents, base: agents, Replay: r, Speed: 1}
	if len(agents) > 0 {
		s.Mode = AgentPlay
	}
	s.prepare()
	return s
}

// SetRuns makes training mode available.
func (s *Session) SetRuns(runs []*trainlog.Run) {
	s.Runs = runs
}

// Run returns the selected training run, or nil if there are none.
func (s *Session) Run() *trainlog.Run {
	if len(s.Runs) == 0 {
		return nil
	}
	return s.Runs[s.RunIdx]
}

// SnapshotStep is the training step of the agent being shown in training mode, or -1.
func (s *Session) SnapshotStep() int {
	if s.Mode != Training || s.Run() == nil || s.AgentIdx >= len(s.Run().Snapshots) {
		return -1
	}
	return s.Run().Snapshots[s.AgentIdx].Step
}

// SelectRun moves to another training run and shows its latest snapshot.
func (s *Session) SelectRun(delta int) {
	if len(s.Runs) == 0 {
		return
	}
	n := len(s.Runs)
	s.RunIdx = ((s.RunIdx+delta)%n + n) % n
	if s.Mode == Training {
		s.useRun()
		s.Restart(s.Seed)
	}
}

// SelectSnapshot shows the latest snapshot at or before step and restarts the same seed.
func (s *Session) SelectSnapshot(step int) {
	if s.Mode != Training || s.Run() == nil {
		return
	}
	for i, snap := range s.Run().Snapshots {
		if snap.Step <= step {
			s.AgentIdx = i
		}
	}
	s.Follow = false
	s.Restart(s.Seed)
}

// useRun swaps the agent list for the selected run's snapshots, following it if training is still going.
func (s *Session) useRun() {
	r := s.Run()
	s.Agents = s.Agents[:0:0]
	for _, snap := range r.Snapshots {
		s.Agents = append(s.Agents, snap.Agent)
	}
	s.AgentIdx = max(len(s.Agents)-1, 0)
	s.Follow = !r.Done
}

// RefreshRuns picks up new metrics and snapshots from training processes that are still running.
// In training mode with Follow on, a new snapshot restarts the game with that agent.
func (s *Session) RefreshRuns() error {
	var firstErr error
	for i, r := range s.Runs {
		if r.Done {
			continue
		}
		before := len(r.Snapshots)
		if _, err := r.Refresh(); err != nil && firstErr == nil {
			firstErr = err
		}
		if i == s.RunIdx && s.Mode == Training && len(r.Snapshots) > before {
			follow, shown := s.Follow, s.AgentIdx
			s.useRun()
			if s.Follow = follow; follow {
				s.Restart(s.Seed) // a new agent to show
			} else {
				s.AgentIdx = shown // keep playing the same game
			}
		}
	}
	return firstErr
}

// Agent returns the selected agent, or nil if none is loaded.
func (s *Session) Agent() *agent.Agent {
	if len(s.Agents) == 0 {
		return nil
	}
	return s.Agents[s.AgentIdx]
}

// SetMode switches mode and restarts. Agent mode needs an agent and replay mode a loaded replay.
func (s *Session) SetMode(m Mode) {
	if (m == AgentPlay && len(s.base) == 0) || (m == Replaying && s.Replay == nil) || (m == Training && s.Run() == nil) {
		return
	}
	if s.Mode == Training && m != Training {
		s.Agents, s.AgentIdx = s.base, 0
	}
	if m == Training && s.Mode != Training {
		s.useRun()
	}
	s.Mode = m
	if m == Replaying {
		s.Restart(s.Replay.Seed)
	} else {
		s.Restart(s.Seed)
	}
}

// SelectAgent moves the checkpoint selection by delta and restarts the same seed, so checkpoints can be compared on one game.
func (s *Session) SelectAgent(delta int) {
	if len(s.Agents) == 0 {
		return
	}
	n := len(s.Agents)
	s.AgentIdx = ((s.AgentIdx+delta)%n + n) % n
	s.Follow = false // stepping by hand stops following a live run
	s.Restart(s.Seed)
}

func (s *Session) Restart(seed uint64) {
	if s.Mode != Replaying {
		s.Seed = seed
	}
	s.Game.Reset(seed)
	s.Played, s.ReplayPos = nil, 0
	s.gravityAcc, s.stepAcc = 0, 0
	s.prepare()
}

// prepare scores the current piece's placements so the viewer can show them before the agent plays.
func (s *Session) prepare() {
	s.Scores, s.BestIdx = nil, -1
	if (s.Mode == AgentPlay || s.Mode == Training) && s.Agent() != nil && !s.Game.State().Done {
		s.Scores = s.Agent().Evaluate(s.Game)
		s.BestIdx = agent.Best(s.Scores)
	}
}

func (s *Session) Update(dt time.Duration, in Input) {
	if s.Paused || s.Game.State().Done {
		return
	}
	switch s.Mode {
	case Human:
		s.updateHuman(dt, in)
	case AgentPlay, Training:
		s.stepAcc += time.Duration(float64(dt) * s.Speed)
		for s.stepAcc >= stepInterval && s.BestIdx >= 0 {
			s.stepAcc -= stepInterval
			pl := s.Scores[s.BestIdx].Placement
			s.Game.Place(pl)
			s.Played = append(s.Played, pl)
			s.prepare()
		}
	case Replaying:
		s.stepAcc += time.Duration(float64(dt) * s.Speed)
		for s.stepAcc >= stepInterval && s.ReplayPos < len(s.Replay.Placements) {
			s.stepAcc -= stepInterval
			s.Game.Place(s.Replay.Placements[s.ReplayPos])
			s.ReplayPos++
		}
	}
}

func (s *Session) updateHuman(dt time.Duration, in Input) {
	for _, a := range []struct {
		pressed bool
		action  engine.Action
	}{{in.Left, engine.Left}, {in.Right, engine.Right}, {in.RotateCW, engine.RotateCW}, {in.RotateCCW, engine.RotateCCW}, {in.HardDrop, engine.HardDrop}} {
		if a.pressed {
			s.Game.Apply(a.action)
		}
	}
	s.gravityAcc += dt
	for s.gravityAcc >= gravityInterval {
		s.gravityAcc -= gravityInterval
		s.Game.Gravity()
	}
}

// SaveReplay writes the current agent game to dir and returns the file path.
func (s *Session) SaveReplay(dir string) (string, error) {
	if s.Mode != AgentPlay || len(s.Played) == 0 {
		return "", fmt.Errorf("only agent games with at least one placement can be saved")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, fmt.Sprintf("%s_%d.json", s.Agent().Name, s.Seed))
	return path, replay.Save(path, replay.Replay{Seed: s.Seed, Agent: s.Agent().Name, Placements: s.Played})
}
