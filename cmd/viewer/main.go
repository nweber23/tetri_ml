// Command viewer plays Tetris by hand, replays saved agent games and shows trained agents
// playing with a value heatmap over every candidate placement.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"

	"github.com/nweber23/tetri_ml/agent"
	"github.com/nweber23/tetri_ml/engine"
	"github.com/nweber23/tetri_ml/replay"
	"github.com/nweber23/tetri_ml/trainlog"
	"github.com/nweber23/tetri_ml/viewer"
)

const (
	screenW, screenH = 760, 540
	cell             = 22
	boardX, boardY   = 20, 20
	mapX, mapY       = 300, 110 // placement value heatmap
	mapCell          = 28
	curveX, curveY   = 300, 300 // learning curve panel
	curveW, curveH   = 430, 180
)

var (
	background = color.RGBA{18, 18, 24, 255}
	panel      = color.RGBA{30, 30, 40, 255}
	gridLine   = color.RGBA{45, 45, 58, 255}
	textDim    = color.RGBA{150, 150, 165, 255}
	kindColors = [engine.NumKinds]color.RGBA{
		engine.I: {0, 220, 230, 255}, engine.O: {240, 220, 0, 255}, engine.T: {170, 60, 220, 255},
		engine.S: {60, 210, 70, 255}, engine.Z: {230, 50, 50, 255}, engine.J: {50, 80, 230, 255},
		engine.L: {240, 150, 30, 255},
	}
)

type app struct {
	s                         *viewer.Session
	replaysDir                string
	message                   string
	frame                     int
	shotPath                  string
	shotFrame                 int
	recordPath                string
	recordEvery, recordFrames int
	recording                 *gif.GIF
}

func main() {
	agentsDir := flag.String("agents", "agents", "directory of exported agents (*.json)")
	runsDir := flag.String("runs", "viewdata", "directory of training runs written by train.py (live or finished)")
	runName := flag.String("run", "", "training run to show first (default: the first one)")
	replayFile := flag.String("replay", "", "replay file to load")
	replaysDir := flag.String("replays", "replays", "where S saves replays")
	seed := flag.Uint64("seed", 1, "game seed")
	mode := flag.String("mode", "", "start mode: human, agent, replay or training")
	speed := flag.Float64("speed", 1, "agent and replay speed multiplier")
	shot := flag.String("screenshot", "", "save a PNG of the window after -screenshot-frame frames and exit")
	shotFrame := flag.Int("screenshot-frame", 120, "frame to capture")
	snapshot := flag.Int("snapshot", -1, "in training mode, show the latest snapshot at or before this step")
	record := flag.String("record", "", "record the window to this animated GIF and exit")
	recordEvery := flag.Int("record-every", 6, "capture every Nth frame (60 frames per second)")
	recordFrames := flag.Int("record-frames", 600, "stop recording after this many frames")
	flag.Parse()

	agents, err := loadAgents(*agentsDir)
	if err != nil {
		log.Fatal(err)
	}
	var r *replay.Replay
	if *replayFile != "" {
		loaded, err := replay.Load(*replayFile)
		if err != nil {
			log.Fatal(err)
		}
		r = &loaded
	}

	s := viewer.New(*seed, agents, r)
	if _, err := os.Stat(*runsDir); err == nil {
		runs, err := trainlog.LoadAll(*runsDir)
		if err != nil {
			log.Fatal(err)
		}
		s.SetRuns(runs)
		for i := 0; *runName != "" && s.Run().Name != *runName; i++ {
			if i == len(runs) {
				log.Fatalf("no run %q in %s", *runName, *runsDir)
			}
			s.SelectRun(1)
		}
	}
	s.Speed = *speed
	switch *mode {
	case "human":
		s.SetMode(viewer.Human)
	case "agent":
		s.SetMode(viewer.AgentPlay)
	case "replay":
		s.SetMode(viewer.Replaying)
	case "training":
		s.SetMode(viewer.Training)
	case "":
		if r != nil {
			s.SetMode(viewer.Replaying)
		}
	default:
		log.Fatalf("unknown mode %q", *mode)
	}

	if *snapshot >= 0 {
		s.SelectSnapshot(*snapshot)
	}

	a := &app{s: s, replaysDir: *replaysDir, shotPath: *shot, shotFrame: *shotFrame,
		recordPath: *record, recordEvery: max(*recordEvery, 1), recordFrames: *recordFrames}
	if *record != "" {
		a.recording = &gif.GIF{}
	}
	ebiten.SetWindowSize(screenW, screenH)
	ebiten.SetWindowTitle("Tetris ML viewer")
	if err := ebiten.RunGame(a); err != nil && err != ebiten.Termination {
		log.Fatal(err)
	}
	if a.recording != nil {
		f, err := os.Create(*record)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		if err := gif.EncodeAll(f, a.recording); err != nil {
			log.Fatal(err)
		}
	}
}

// gifPalette is the UI's own flat colours, so the dark background, grid and pieces stay exact, plus the
// 216 web-safe colours for everything else (text edges, anti-aliased chart lines). A GIF has 256 colours.
func gifPalette() color.Palette {
	exact := []color.Color{background, panel, gridLine, textDim, color.White, color.Black}
	for _, c := range kindColors {
		exact = append(exact, c)
	}
	return append(exact, palette.WebSafe...)
}

// capture copies the screen into an image the encoders can read quickly.
func capture(screen *ebiten.Image) *image.RGBA {
	b := screen.Bounds()
	img := image.NewRGBA(b)
	screen.ReadPixels(img.Pix)
	return img
}

func loadAgents(dir string) ([]*agent.Agent, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	var agents []*agent.Agent
	for _, f := range files {
		a, err := agent.Load(f)
		if err != nil {
			return nil, err
		}
		agents = append(agents, a)
	}
	return agents, nil
}

// held reports a key press with auto-repeat: on the first frame, then every other frame after a short delay.
func held(k ebiten.Key) bool {
	d := inpututil.KeyPressDuration(k)
	return d == 1 || (d > 12 && d%3 == 0)
}

func (a *app) Update() error {
	a.frame++
	if (a.shotPath != "" && a.frame > a.shotFrame) || (a.recording != nil && a.frame > a.recordFrames) {
		return ebiten.Termination
	}
	s := a.s
	switch {
	case inpututil.IsKeyJustPressed(ebiten.KeyTab):
		for i := 1; i < viewer.NumModes; i++ { // next mode that has what it needs (agents, a replay or runs)
			if m := (s.Mode + viewer.Mode(i)) % viewer.NumModes; a.modeAvailable(m) {
				s.SetMode(m)
				break
			}
		}
	case inpututil.IsKeyJustPressed(ebiten.KeyN):
		s.Restart(s.Seed + 1)
	case inpututil.IsKeyJustPressed(ebiten.KeyBracketLeft):
		s.SelectAgent(-1)
	case inpututil.IsKeyJustPressed(ebiten.KeyBracketRight):
		s.SelectAgent(1)
	case inpututil.IsKeyJustPressed(ebiten.KeyR):
		s.SelectRun(1)
	case inpututil.IsKeyJustPressed(ebiten.KeyF):
		s.Follow = !s.Follow
	case inpututil.IsKeyJustPressed(ebiten.KeyP):
		s.Paused = !s.Paused
	case inpututil.IsKeyJustPressed(ebiten.KeyEqual), inpututil.IsKeyJustPressed(ebiten.KeyKPAdd):
		s.Speed = min(s.Speed*2, 64)
	case inpututil.IsKeyJustPressed(ebiten.KeyMinus), inpututil.IsKeyJustPressed(ebiten.KeyKPSubtract):
		s.Speed = max(s.Speed/2, 0.25)
	case inpututil.IsKeyJustPressed(ebiten.KeyS):
		if path, err := s.SaveReplay(a.replaysDir); err != nil {
			a.message = err.Error()
		} else {
			a.message = "saved " + path
		}
	}

	if a.frame%60 == 0 { // pick up what a running training process has written
		if err := s.RefreshRuns(); err != nil {
			a.message = err.Error()
		}
	}

	in := viewer.Input{
		Left:      held(ebiten.KeyArrowLeft),
		Right:     held(ebiten.KeyArrowRight),
		RotateCW:  inpututil.IsKeyJustPressed(ebiten.KeyArrowUp) || inpututil.IsKeyJustPressed(ebiten.KeyX),
		RotateCCW: inpututil.IsKeyJustPressed(ebiten.KeyZ),
		HardDrop:  inpututil.IsKeyJustPressed(ebiten.KeySpace),
		SoftDrop:  ebiten.IsKeyPressed(ebiten.KeyArrowDown),
		Hold:      inpututil.IsKeyJustPressed(ebiten.KeyC),
	}
	s.Update(time.Second/time.Duration(ebiten.TPS()), in)
	return nil
}

func (a *app) modeAvailable(m viewer.Mode) bool {
	switch m {
	case viewer.AgentPlay:
		return a.s.Agent() != nil
	case viewer.Replaying:
		return a.s.Replay != nil
	case viewer.Training:
		return a.s.Run() != nil
	}
	return true
}

func (a *app) Layout(int, int) (int, int) { return screenW, screenH }

func (a *app) Draw(screen *ebiten.Image) {
	screen.Fill(background)
	a.drawBoard(screen)
	a.drawInfo(screen)
	if a.s.Mode == viewer.Training {
		a.drawTraining(screen)
	} else {
		a.drawHeatmap(screen)
		a.drawCurve(screen)
	}
	if a.shotPath != "" && a.frame == a.shotFrame {
		f, err := os.Create(a.shotPath)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		if err := png.Encode(f, capture(screen)); err != nil {
			log.Fatal(err)
		}
	}
	if a.recording != nil && a.frame%a.recordEvery == 0 {
		img := image.NewPaletted(screen.Bounds(), gifPalette())
		draw.Draw(img, img.Bounds(), capture(screen), image.Point{}, draw.Src)
		a.recording.Image = append(a.recording.Image, img)
		a.recording.Delay = append(a.recording.Delay, a.recordEvery*100/60)
	}
}

func cellRect(screen *ebiten.Image, x, y int, c color.Color) {
	// y is a board row; row 19 is drawn at the top of the visible field
	px, py := float32(boardX+x*cell), float32(boardY+(engine.VisibleHeight-1-y)*cell)
	vector.FillRect(screen, px+1, py+1, cell-2, cell-2, c, false)
}

func (a *app) drawBoard(screen *ebiten.Image) {
	st := a.s.Game.State()
	vector.FillRect(screen, boardX, boardY, engine.Width*cell, engine.VisibleHeight*cell, panel, false)
	for i := 0; i <= engine.Width; i++ {
		vector.StrokeLine(screen, float32(boardX+i*cell), boardY, float32(boardX+i*cell), boardY+engine.VisibleHeight*cell, 1, gridLine, false)
	}
	for i := 0; i <= engine.VisibleHeight; i++ {
		vector.StrokeLine(screen, boardX, float32(boardY+i*cell), boardX+engine.Width*cell, float32(boardY+i*cell), 1, gridLine, false)
	}
	for y := 0; y < engine.VisibleHeight; y++ {
		for x := 0; x < engine.Width; x++ {
			if k := st.Board[y][x]; k != 0 {
				cellRect(screen, x, y, kindColors[k-1])
			}
		}
	}
	if m := a.s.Plan; m != nil { // outline where the move agent will lock the piece
		for _, c := range m.Piece.Cells() {
			if c.Y < engine.VisibleHeight {
				vector.StrokeRect(screen, float32(boardX+c.X*cell)+2, float32(boardY+(engine.VisibleHeight-1-c.Y)*cell)+2, cell-4, cell-4, 2, color.White, false)
			}
		}
	}
	if a.s.BestIdx >= 0 { // outline where the agent will drop the piece
		for _, c := range a.s.Game.Landing(a.s.Scores[a.s.BestIdx].Placement).Cells() {
			if c.Y < engine.VisibleHeight {
				vector.StrokeRect(screen, float32(boardX+c.X*cell)+2, float32(boardY+(engine.VisibleHeight-1-c.Y)*cell)+2, cell-4, cell-4, 2, color.White, false)
			}
		}
	}
	if !st.Done && a.s.Mode == viewer.Human { // ghost: where the piece lands if dropped now
		for _, c := range a.s.Game.Landing(engine.Placement{Rot: st.Piece.Rot, X: st.Piece.X}).Cells() {
			if c.Y < engine.VisibleHeight {
				vector.StrokeRect(screen, float32(boardX+c.X*cell)+2, float32(boardY+(engine.VisibleHeight-1-c.Y)*cell)+2, cell-4, cell-4, 1, kindColors[st.Piece.Kind], false)
			}
		}
	}
	if !st.Done {
		for _, c := range st.Piece.Cells() {
			if c.Y < engine.VisibleHeight {
				cellRect(screen, c.X, c.Y, kindColors[st.Piece.Kind])
			}
		}
	}
	drawQueue(screen, st.Preview)
	if st.HasHold {
		drawPreview(screen, "hold", boardX, st.Hold)
	} else {
		ebitenutil.DebugPrintAt(screen, "hold", boardX, boardY+engine.VisibleHeight*cell+8)
	}
}

// drawQueue draws the upcoming pieces in a column right of the board, the next one on top.
func drawQueue(screen *ebiten.Image, queue [engine.PreviewSize]engine.Kind) {
	const x, y0, c, step = boardX + engine.Width*cell + 12, boardY + 16, 11, 34 // c: preview cell size
	ebitenutil.DebugPrintAt(screen, "next", x, boardY)
	for i, k := range queue {
		cells := nextCells(k)
		minY := min(cells[0].Y, cells[1].Y, cells[2].Y, cells[3].Y)
		for _, p := range cells {
			vector.FillRect(screen, float32(x+p.X*c), float32(y0+i*step+(1-(p.Y-minY))*c), c-1, c-1, kindColors[k], false)
		}
	}
}

// drawPreview draws a small piece under the board next to a label.
func drawPreview(screen *ebiten.Image, label string, x int, k engine.Kind) {
	y := boardY + engine.VisibleHeight*cell + 8
	ebitenutil.DebugPrintAt(screen, label, x, y)
	cells := nextCells(k)
	minY := min(cells[0].Y, cells[1].Y, cells[2].Y, cells[3].Y)
	for _, c := range cells {
		vector.FillRect(screen, float32(x+50+c.X*14), float32(y+(1-(c.Y-minY))*14), 13, 13, kindColors[k], false)
	}
}

// nextCells returns the spawn-orientation cells of a kind, for the preview.
func nextCells(k engine.Kind) [4]engine.Point {
	return engine.Piece{Kind: k}.Cells()
}

func (a *app) drawInfo(screen *ebiten.Image) {
	s := a.s
	st := s.Game.State()
	name := "-"
	if ag := s.Agent(); ag != nil && (s.Mode == viewer.AgentPlay || s.Mode == viewer.Training) {
		name = fmt.Sprintf("%s (%d/%d)", ag.Name, s.AgentIdx+1, len(s.Agents))
	}
	status := ""
	switch {
	case st.Done:
		status = "  GAME OVER"
	case s.Paused:
		status = "  paused"
	case s.Mode == viewer.Replaying && s.ReplayPos >= len(s.Replay.Placements):
		status = "  replay finished"
	}
	text := fmt.Sprintf("mode: %s%s\nagent: %s\nseed: %d   speed: x%g\nscore %d   lines %d   pieces %d",
		s.Mode, status, name, seedOf(s), s.Speed, st.Score, st.Lines, st.Pieces)
	if st.TSpin {
		text += fmt.Sprintf("\nlast clear: T-SPIN, %d lines", st.LastClear)
	} else if st.LastClear > 0 {
		text += fmt.Sprintf("\nlast clear: %d lines", st.LastClear)
	}
	ebitenutil.DebugPrintAt(screen, text, 300, 10)
	help := "Tab mode  N new seed  [ ] agent/snapshot  R next run  F follow live  P pause  +/- speed  S save replay\nHuman: arrows move/soft drop, up/X rotate, Z rotate back, C hold, space drop"
	ebitenutil.DebugPrintAt(screen, help, 20, screenH-30)
	if a.message != "" {
		ebitenutil.DebugPrintAt(screen, a.message, 300, screenH-50)
	}
}

func seedOf(s *viewer.Session) uint64 {
	if s.Mode == viewer.Replaying {
		return s.Replay.Seed
	}
	return s.Seed
}

func (a *app) drawHeatmap(screen *ebiten.Image) {
	ebitenutil.DebugPrintAt(screen, "agent value per placement (row = rotation, column = x); move agents: current plan", mapX, mapY-16)
	vector.FillRect(screen, mapX, mapY, 12*mapCell, 4*mapCell, panel, false)
	s := a.s
	if m := s.Plan; m != nil {
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("move agent: %d candidate moves\nplaying: %d keys%s%s\nscore for the move: %d",
			len(s.MoveScores), len(m.Keys), map[bool]string{true: ", hold", false: ""}[m.Hold], map[bool]string{true: ", T-SPIN", false: ""}[m.TSpin], m.Reward), mapX+10, mapY+20)
		return
	}
	if len(s.Scores) == 0 {
		ebitenutil.DebugPrintAt(screen, "agent mode only", mapX+10, mapY+50)
		return
	}
	lo, hi := s.Scores[0].Value, s.Scores[0].Value
	for _, sc := range s.Scores {
		lo, hi = min(lo, sc.Value), max(hi, sc.Value)
	}
	for i, sc := range s.Scores {
		t := 1.0
		if hi > lo {
			t = (sc.Value - lo) / (hi - lo)
		}
		x, y := float32(mapX+(sc.Placement.X+2)*mapCell), float32(mapY+sc.Placement.Rot*mapCell)
		vector.FillRect(screen, x+1, y+1, mapCell-2, mapCell-2, heat(t), false)
		if i == s.BestIdx {
			vector.StrokeRect(screen, x+1, y+1, mapCell-2, mapCell-2, 2, color.White, false)
		}
	}
	best := s.Scores[s.BestIdx]
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("best value %.2f  (holes %d, bumpiness %d, lines %d)   range %.2f .. %.2f",
		best.Value, best.Features.Holes, best.Features.Bumpiness, best.Features.Lines, lo, hi), mapX, mapY+4*mapCell+6)
	for i := 0; i < 12; i++ {
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%d", i-2), mapX+i*mapCell+10, mapY-2+4*mapCell+22)
	}
}

// heat maps 0..1 from dark red through orange to green.
func heat(t float64) color.RGBA {
	return color.RGBA{uint8(200 * (1 - t)), uint8(40 + 170*t), 60, 255}
}

func (a *app) drawCurve(screen *ebiten.Image) {
	ebitenutil.DebugPrintAt(screen, "validation score during training", curveX, curveY-16)
	vector.FillRect(screen, curveX, curveY, curveW, curveH, panel, false)
	ag := a.s.Agent()
	if ag == nil || len(ag.Curve) == 0 {
		ebitenutil.DebugPrintAt(screen, "no agent loaded", curveX+10, curveY+10)
		return
	}
	maxStep, maxScore := 0, 0.0
	for _, p := range ag.Curve {
		maxStep, maxScore = max(maxStep, p.Step), max(maxScore, p.Score)
	}
	if maxStep == 0 || maxScore == 0 {
		return
	}
	pt := func(p agent.CurvePoint) (float32, float32) {
		return float32(curveX+10) + float32(p.Step)/float32(maxStep)*(curveW-20),
			float32(curveY+curveH-12) - float32(p.Score/maxScore)*(curveH-30)
	}
	for i := 1; i < len(ag.Curve); i++ {
		x0, y0 := pt(ag.Curve[i-1])
		x1, y1 := pt(ag.Curve[i])
		vector.StrokeLine(screen, x0, y0, x1, y1, 2, color.RGBA{90, 190, 255, 255}, true)
	}
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%s: %.0f at step %d (peak axis %.0f)", ag.Name, ag.Curve[len(ag.Curve)-1].Score, maxStep, maxScore), curveX+8, curveY+4)
}
