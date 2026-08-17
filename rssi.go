package lmamath

import "math"

// CalculateRealDistance converts a smoothed RSSI reading into an estimated
// distance in meters, using the log-distance path-loss model anchored at the
// 1-meter reference: distance = 10^((txCalibratedPower - rssi) / 20).
//
// onMeterRssi is the RSSI measured at exactly 1 meter from the beacon
// (txCalibratedPower). It is kept in this package because trilateration
// consumers typically derive their distance inputs this way.
func CalculateRealDistance(onMeterRssi int, currenRssi int) float64 {
	ratioDB := onMeterRssi - currenRssi
	return math.Pow(10.0, float64(ratioDB)/20)
}
