package heuristic

import (
	"testing"

	"github.com/nweber23/tetri_ml/engine"
)

func TestExtract(t *testing.T) {
	var b engine.Board
	b[0][0], b[1][0], b[2][0] = 1, 1, 1 // column 0: height 3
	b[1][1] = 1                         // column 1: height 2 with a hole below
	b[0][2] = 1                         // column 2: height 1
	got := Extract(b, 2)
	// heights 3,2,1,0...: aggregate 6, bumpiness 1+1+1, one hole at (1,0)
	want := Features{AggregateHeight: 6, Holes: 1, Bumpiness: 3, Lines: 2}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestBestIsValidAndDeterministic(t *testing.T) {
	g := engine.New(1)
	a, b := Best(g, Default), Best(g, Default)
	if a != b {
		t.Error("Best is not deterministic")
	}
	found := false
	for _, pl := range g.Placements() {
		found = found || pl == a
	}
	if !found {
		t.Errorf("Best returned %+v, not in Placements", a)
	}
}

func TestPlaysManyPieces(t *testing.T) {
	g := engine.New(1)
	for s := g.State(); !s.Done && s.Pieces < 200; s = g.State() {
		g.Place(Best(g, Default))
	}
	if s := g.State(); s.Lines == 0 {
		t.Errorf("heuristic cleared no lines in %d pieces", s.Pieces)
	}
}
