# lmamath

![Go](https://img.shields.io/github/go-mod/go-version/Vitaliy69/lmamath)
![Build](https://img.shields.io/github/actions/workflow/status/Vitaliy69/lmamath/go.yml)
![License](https://img.shields.io/github/license/Vitaliy69/lmamath)
![Go Reference](https://pkg.go.dev/badge/github.com/Vitaliy69/lmamath.svg)

n-Dimensional trilateration in Go using the Levenberg–Marquardt Algorithm (LMA).

The library solves a non-linear least-squares problem: given a set of anchor
points with known coordinates and measured distances to an unknown point, it
estimates the most likely coordinates of that point.

## Features

- Works in spaces of arbitrary dimension (2D and higher).
- Robust to noise in distance measurements.
- Minimal dependencies, pure Go.

## Use cases

Trilateration appears anywhere a position must be recovered from distances to
known reference points:

- Indoor positioning (BLE beacons, UWB, Wi-Fi RTT) where satellite positioning cannot reach.
- Asset and equipment tracking in IoT networks.
- Localization of warehouse robots and delivery fleet vehicles.
## Installation

```bash
go get github.com/Vitaliy69/lmamath
```

## Usage

```go
package main

import (
	"fmt"

	"github.com/Vitaliy69/lmamath"
)

func main() {
	// Anchor coordinates (example in three axes; any consistent dimension works)
	positions := [][]float64{
		{1.5, 5.0, 0.5},
		{-4.5, -6.7, 3.0},
		{18.5, 12.5, 0.5},
		{10.5, 15.6, 2.75},
	}

	// Measured distances from each anchor to the unknown point
	distances := []float64{3.0, 4.0, 5.9, 13.1}

	coordinates, err := lmamath.Solve_LMA(positions, distances)
	if err != nil {
		fmt.Printf("solve error: %s\n", err)
		return
	}

	fmt.Printf("Estimated coordinates: %v\n", coordinates)
}
```

### Input requirements

- `positions` — a slice of points; all points must have the same dimension `n`.
- `distances` — one distance per anchor; its length must match the number of points.
- For a well-defined solution, provide at least `n + 1` anchors.

## How it works

The problem is reduced to minimizing the sum of squared residuals:

​```
f(x) = Σ ( ||x - pᵢ||² - dᵢ² )²
​```

where `pᵢ` are the coordinates of the i-th anchor, `dᵢ` is the measured distance
to it, and `x` is the unknown position. The squared form (difference of squared
distances rather than of distances) shares its minimizer with the direct form
whenever an exact solution exists, and yields a simpler Jacobian. The
Levenberg–Marquardt algorithm
iteratively refines `x` by blending Gauss–Newton and gradient-descent steps: the
damping factor is adjusted automatically, which helps avoid getting stuck in
local minima and ensures stable convergence even with noisy measurements.

## License

Released under the MIT License. See the [LICENSE](LICENSE) file.
