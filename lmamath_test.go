package lmamath

import (
	"math"
	"testing"
)

func approxEq(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func TestExact2D(t *testing.T) {
	pos := [][]float64{{0, 0}, {10, 0}, {0, 10}}
	d := []float64{math.Hypot(3, 4), math.Hypot(7, 4), math.Hypot(3, 6)}
	got, err := Solve_LMA(pos, d)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !approxEq(got, []float64{3, 4}, 1e-6) {
		t.Fatalf("want [3,4], got %v", got)
	}
}

func TestTooFewAnchors(t *testing.T) {
	if _, err := Solve_LMA([][]float64{{0, 0}, {1, 1}}, []float64{1, 1}); err == nil {
		t.Fatal("want error for <3 anchors")
	}
}

func TestCollinearAnchors(t *testing.T) {
	pos := [][]float64{{0, 0}, {5, 0}, {10, 0}}
	d := []float64{3.1, 2.1, 7.1} // point (3,0.1) but ill-conditioned
	got, err := Solve_LMA(pos, d)
	t.Logf("collinear: got=%v err=%v", got, err)
}

func TestInconsistentDimension(t *testing.T) {
	pos := [][]float64{{0, 0}, {1, 0, 0}, {0, 1, 0}}
	if _, err := Solve_LMA(pos, []float64{1, 1, 1}); err == nil {
		t.Fatal("want error for mixed dimensions")
	}
}

func TestNegativeDistance(t *testing.T) {
	pos := [][]float64{{0, 0}, {10, 0}, {0, 10}}
	if _, err := Solve_LMA(pos, []float64{-1, 2, 3}); err == nil {
		t.Fatal("want error for negative distance")
	}
}

func TestZeroDistance(t *testing.T) {
	pos := [][]float64{{0, 0}, {10, 0}, {0, 10}, {10, 10}}
	got, err := Solve_LMA(pos, []float64{0, math.Hypot(5, 5), math.Hypot(5, 5), math.Hypot(5, 5)})
	t.Logf("zero-distance anchor: got=%v err=%v", got, err)
}

func TestDegenerateAllZeroDistances(t *testing.T) {
	pos := [][]float64{{0, 0}, {10, 0}, {0, 10}}
	if _, err := Solve_LMA(pos, []float64{0, 0, 0}); err == nil {
		t.Fatal("want error for all-zero distances (division by zero in weights)")
	}
}

func TestDuplicateAnchors(t *testing.T) {
	pos := [][]float64{{0, 0}, {0, 0}, {10, 10}}
	if _, err := Solve_LMA(pos, []float64{1, 1, 5}); err == nil {
		t.Fatal("want error for duplicate anchors (singular)")
	}
}

func TestHugeDistances(t *testing.T) {
	pos := [][]float64{{0, 0}, {10, 0}, {0, 10}}
	got, err := Solve_LMA(pos, []float64{1e8, 1e8 + 5, 1e8 + 3})
	t.Logf("huge: got=%v err=%v", got, err)
}

func TestNaNInput(t *testing.T) {
	pos := [][]float64{{0, 0}, {math.NaN(), 0}, {0, 10}}
	got, err := Solve_LMA(pos, []float64{1, 1, 1})
	t.Logf("nan input: got=%v err=%v", got, err)
}

func TestPackageGlobalsRace(t *testing.T) {
	// Solve_LMA uses package-level lmaPositions/lmaDistances.
	// Two concurrent solves must corrupt each other if globals are used.
	a := [][]float64{{0, 0}, {10, 0}, {0, 10}}
	b := [][]float64{{100, 100}, {200, 100}, {100, 200}}
	done := make(chan []float64, 2)
	run := func(anchors [][]float64, target []float64) {
		d := make([]float64, len(anchors))
		for i, p := range anchors {
			d[i] = math.Hypot(target[0]-p[0], target[1]-p[1])
		}
		r, err := Solve_LMA(anchors, d)
		if err != nil {
			r = nil
		}
		done <- r
	}
	go run(a, []float64{5, 5})
	go run(b, []float64{150, 150})
	r1, r2 := <-done, <-done
	t.Logf("concurrent: r1=%v r2=%v", r1, r2)
}
