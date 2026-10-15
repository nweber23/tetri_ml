package main

import (
	"fmt"
	"image/color"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"

	"github.com/nweber23/tetri_ml/trainlog"
)

const (
	trainX = 300
	trainW = 440
	smooth = 50 // episodes per moving-average point
)

var runColors = []color.RGBA{
	{90, 190, 255, 255}, {255, 150, 70, 255}, {120, 220, 120, 255}, {230, 110, 200, 255},
	{240, 220, 90, 255}, {150, 130, 255, 255}, {90, 220, 210, 255}, {230, 90, 90, 255},
}

type series struct {
	points []trainlog.Point
	color  color.RGBA
	width  float32
}

// drawChart plots series against training step, with a vertical marker at the shown snapshot.
func drawChart(screen *ebiten.Image, x, y, w, h float32, title string, marker int, all ...series) {
	ebitenutil.DebugPrintAt(screen, title, int(x), int(y)-16)
	vector.FillRect(screen, x, y, w, h, panel, false)
	maxStep, maxVal := 1, 0.0
	for _, s := range all {
		for _, p := range s.points {
			maxStep, maxVal = max(maxStep, p.Step), max(maxVal, p.Value)
		}
	}
	if maxVal == 0 {
		ebitenutil.DebugPrintAt(screen, "no data yet", int(x)+8, int(y)+8)
		return
	}
	px := func(step int) float32 { return x + 6 + float32(step)/float32(maxStep)*(w-12) }
	py := func(v float64) float32 { return y + h - 6 - float32(v/maxVal)*(h-24) }
	if marker >= 0 && marker <= maxStep {
		vector.StrokeLine(screen, px(marker), y+2, px(marker), y+h-2, 1, color.RGBA{255, 255, 255, 120}, false)
	}
	for _, s := range all {
		stride := max(1, len(s.points)/int(w)) // keep drawing cheap for long runs
		for i := stride; i < len(s.points); i += stride {
			a, b := s.points[i-stride], s.points[i]
			vector.StrokeLine(screen, px(a.Step), py(a.Value), px(b.Step), py(b.Value), s.width, s.color, true)
		}
	}
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("max %s   step %s", short(maxVal), short(float64(maxStep))), int(x)+6, int(y)+2)
}

// short formats 12345 as "12.3k".
func short(v float64) string {
	switch {
	case v >= 1e6:
		return fmt.Sprintf("%.1fM", v/1e6)
	case v >= 1e3:
		return fmt.Sprintf("%.1fk", v/1e3)
	case v >= 10:
		return fmt.Sprintf("%.0f", v)
	}
	return fmt.Sprintf("%.3g", v)
}

func smoothedEpisodes(r *trainlog.Run, value func(trainlog.Episode) float64) []trainlog.Point {
	v := make([]float64, len(r.Episodes))
	for i, e := range r.Episodes {
		v[i] = value(e)
	}
	v = trainlog.Smooth(v, smooth)
	pts := make([]trainlog.Point, len(v))
	for i, e := range r.Episodes {
		pts[i] = trainlog.Point{Step: e.Step, Value: v[i]}
	}
	return pts
}

func smoothedLoss(r *trainlog.Run) []trainlog.Point {
	v := make([]float64, len(r.Losses))
	for i, p := range r.Losses {
		v[i] = p.Value
	}
	v = trainlog.Smooth(v, 10)
	pts := make([]trainlog.Point, len(v))
	for i, p := range r.Losses {
		pts[i] = trainlog.Point{Step: p.Step, Value: v[i]}
	}
	return pts
}

func validationPoints(r *trainlog.Run) []trainlog.Point {
	pts := make([]trainlog.Point, len(r.Validations))
	for i, v := range r.Validations {
		pts[i] = trainlog.Point{Step: v.Step, Value: v.Score}
	}
	return pts
}

func (a *app) drawTraining(screen *ebiten.Image) {
	s := a.s
	r := s.Run()
	state := "finished"
	if !r.Done {
		state = fmt.Sprintf("LIVE, updated %s ago", time.Since(r.Updated).Round(time.Second))
	}
	follow := ""
	if s.Follow {
		follow = "  following"
	}
	last := "-"
	if n := len(r.Episodes); n > 0 {
		e := r.Episodes[n-1]
		last = fmt.Sprintf("last episode: score %.0f lines %.0f pieces %.0f, epsilon %.2f", e.Score, e.Lines, e.Pieces, e.Epsilon)
	}
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("run %s (%d/%d): %s%s\nstep %d, %d episodes, snapshot %d/%d\n%s",
		r.Name, s.RunIdx+1, len(s.Runs), state, follow, r.LastStep(), len(r.Episodes), s.AgentIdx+1, len(r.Snapshots), last), trainX, 84)

	marker := s.SnapshotStep()
	drawChart(screen, trainX, 150, trainW, 120, "training score per episode (moving average)", marker,
		series{smoothedEpisodes(r, func(e trainlog.Episode) float64 { return e.Score }), runColors[s.RunIdx%len(runColors)], 2})

	var all []series
	for i, other := range s.Runs {
		c, w := runColors[i%len(runColors)], float32(1)
		if i == s.RunIdx {
			w = 3
		} else {
			c.A = 110
		}
		all = append(all, series{validationPoints(other), c, w})
	}
	drawChart(screen, trainX, 305, trainW, 100, "validation score, all runs (bold = this run)", marker, all...)

	half := float32(trainW-10) / 2
	drawChart(screen, trainX, 445, half, 58, "loss (smoothed)", marker, series{smoothedLoss(r), color.RGBA{255, 150, 70, 255}, 2})
	drawChart(screen, trainX+half+10, 445, half, 58, "epsilon", marker,
		series{smoothedEpisodes(r, func(e trainlog.Episode) float64 { return e.Epsilon }), color.RGBA{120, 220, 120, 255}, 2})
}
