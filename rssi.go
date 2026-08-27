package lmamath

import "math"

// CalculateRealDistance converts a smoothed RSSI reading into an estimated
// distance in meters, using the log-distance path-loss model anchored at the
// 1-meter reference with the free-space exponent n = 2:
// distance = 10^((txCalibratedPower - rssi) / 20).
//
// It is kept in this package because trilateration consumers typically derive
// their distance inputs this way. For measured indoor environments, prefer
// CalculateRealDistanceN with a per-zone fitted exponent.
func CalculateRealDistance(onMeterRssi int, currenRssi int) float64 {
	return CalculateRealDistanceN(onMeterRssi, currenRssi, 2.0)
}

// CalculateRealDistanceN is the general form of the path-loss conversion:
// distance = 10^((txCalibratedPower - rssi) / (10 * n)).
//
// The exponent n is the path-loss exponent: 2 is free space, published indoor
// measurements run from about 2.7 to 3.5 in offices and 4 or higher in
// cluttered industrial halls. Fit n per zone from baseline readings and pass
// it here for anchors in that zone. Returns NaN when n <= 0.
func CalculateRealDistanceN(onMeterRssi int, currenRssi int, n float64) float64 {
	if n <= 0 {
		return math.NaN()
	}
	ratioDB := onMeterRssi - currenRssi
	return math.Pow(10.0, float64(ratioDB)/(10.0*n))
}
