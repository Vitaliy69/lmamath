package lmamath

import (
	"math"
	"testing"
)

func TestCalculateRealDistance(t *testing.T) {
	cases := []struct {
		name string
		tx   int
		rssi int
		want float64
	}{
		{"at the 1-meter reference", -59, -59, 1.0},
		{"free-space example from the README", -59, -70, math.Pow(10, 11.0/20)},
		{"closer than 1 m", -59, -50, math.Pow(10, -9.0/20)},
		{"factory default reference", -59, -75, math.Pow(10, 16.0/20)},
	}
	for _, c := range cases {
		got := CalculateRealDistance(c.tx, c.rssi)
		if math.Abs(got-c.want) > 1e-9 {
			t.Fatalf("%s: CalculateRealDistance(%d, %d) = %v, want %v",
				c.name, c.tx, c.rssi, got, c.want)
		}
	}
}
