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
)

type Mode int

const (
	Human Mode = iota
	AgentPlay
	Replaying
)

func (m Mode) String() string {
	return [...]string{"human", "agent", "replay"}[m]
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

	Replay    *replay.Replay // loaded replay, if any
	ReplayPos int
	Played    []engine.Placement // placements of the current agent game, for saving

	gravityAcc, stepAcc time.Duration
}

// New starts a session in agent mode if there are agents, otherwise human mode.
// A non-nil r makes replay mode available.
func New(seed uint64, agents []*agent.Agent, r *replay.Replay) *Session {
	s := &Session{Game: engine.New(seed), Seed: seed, Agents: agents, Replay: r, Speed: 1}
	if len(agents) > 0 {
		s.Mode = AgentPlay
	}
	s.prepare()
	return s
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
	if (m == AgentPlay && s.Agent() == nil) || (m == Replaying && s.Replay == nil) {
		return
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
	if s.Mode == AgentPlay && !s.Game.State().Done {
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
	case AgentPlay:
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
