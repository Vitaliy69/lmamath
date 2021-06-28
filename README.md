# lmamath

Solves a formulation of n-D space trilateration problem using a nonlinear least squares optimizer.

**Input:** positions, distances  
**Output:** centroid 

Uses [Levenberg-Marquardt algorithm](http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).

## Usage 
```
import "github.com/Vitaliy69/lmamath"

//positions := [][]float64{{1.5, 5.0}, {-4.5, -6.7}, {18.5, 12.5}, {10.5, 15.6}} // 2-D space
positions := [][]float64{{1.5, 5.0, 0.5}, {-4.5, -6.7, 3.0}, {18.5, 12.5, 0.5}, {10.5, 15.6, 2.75}} // 3-D space
distances := []float64{3.0, 4.0, 5.9, 13.1}

if coordinates, e := Solve_LMA(positions, distances); e == nil {
	fmt.Printf("Coordinates are: %v\n", coordinates)
} else {
	fmt.Printf("Calculation error: %s\n", e)
}
```
