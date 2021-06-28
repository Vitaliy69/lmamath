//
//  LMAMath
//
//  Created by Vitaliy Gribko on 10.06.2021.
//
//  Solves a formulation of n-D space trilateration problem using a nonlinear
//  least squares optimizer. Uses Levenberg-Marquardt algorithm.
//

package lmamath

import (
	"errors"
	"math"
)

type lmaOptimum struct {
	target           []float64
	weightSquareRoot []float64
	start            []float64
}

type lmaEvaluation struct {
	jacobian  [][]float64
	residuals []float64
	point     []float64
}

type lmaInternalData struct {
	weightedJacobian [][]float64
	permutation      []int
	rank             int
	diagR            []float64
	jacNorm          []float64
	beta             []float64
}

const maxIteration = 1000
const maxEvaluation = 1000

var lmaPositions [][]float64
var lmaDistances []float64

func createArray(size int) []float64 {
	s := make([]float64, size)
	for i := range s {
		s[i] = 0.0
	}

	return s
}

func uniquesArray(intSlice []float64) []float64 {
	keys := make(map[float64]bool)
	list := []float64{}
	for _, entry := range intSlice {
		if _, value := keys[entry]; !value {
			keys[entry] = true
			list = append(list, entry)
		}
	}

	return list
}

func CalculateRealDistance(onMeterRssi int, currenRssi int) float64 {
	ratioDB := onMeterRssi - currenRssi
	return math.Pow(10.0, float64(ratioDB)/20)
}

func Solve_LMA(positions [][]float64, distances []float64) ([]float64, error) {
	if len(positions) < 2 || len(distances) < 2 || len(positions) != len(distances) {
		return nil, errors.New("input arguments error")
	}

	for i := 0; i < len(positions[0]); i++ {
		var values []float64
		for j := 0; j < len(positions); j++ {
			values = append(values, positions[j][i])
		}

		uniques := uniquesArray(values)
		if len(uniques) == 1 {
			return nil, errors.New("there is no solution for this input data")
		}
	}

	lmaPositions = positions
	lmaDistances = distances

	var numberOfPositions = len(positions)
	var positionDimension = len(positions[0])

	var initialPoint = createArray(positionDimension)
	for i := 0; i < len(positions); i++ {
		vertex := positions[i]
		for j := 0; j < len(vertex); j++ {
			initialPoint[j] += vertex[j]
		}
	}

	for j := 0; j < len(initialPoint); j++ {
		initialPoint[j] /= float64(numberOfPositions)
	}

	var target = createArray(numberOfPositions)
	var weights = make([]float64, len(distances))
	for i := range distances {
		weights[i] = inverseSquareLaw(distances[i])
	}

	optimum := solveProblem(target, weights, initialPoint)
	if evaluation, e := optimize(optimum); e != nil {
		return nil, errors.New("optimize error")
	} else {
		return evaluation.point, nil
	}
}

func inverseSquareLaw(distance float64) float64 {
	return 1 / (distance * distance)
}

func solveProblem(target []float64, weights []float64, initialPoint []float64) lmaOptimum {
	var optimun lmaOptimum
	optimun.target = target
	optimun.weightSquareRoot = createArray(len(weights))
	for i, value := range weights {
		optimun.weightSquareRoot[i] = math.Sqrt(value)
	}
	optimun.start = createArray(len(initialPoint))
	for i, value := range initialPoint {
		optimun.start[i] = value
	}

	return optimun
}

func optimize(optium lmaOptimum) (lmaEvaluation, error) {
	const initialStepBoundFactor = 100.0
	const orthoTolerance = 1.0e-10
	const costRelativeTolerance = 1.0e-10
	const parRelativeTolerance = 1.0e-10
	const two_eps = 2.220446049250313e-16

	// Pull in relevant data from the problem as locals
	nR := len(optium.target) // Number of observed data
	nC := len(optium.start)  // Number of parameters
	iterationCounter := 0
	evaluationCounter := 1

	// Levenberg-Marquardt parameters
	solvedCols := int(math.Min(float64(nR), float64(nC)))
	lmPar := 0.0
	lmDir := createArray(nC)

	// Local point
	delta := 0.0
	xNorm := 0.0
	diag := createArray(nC)
	oldX := createArray(nC)
	oldRes := createArray(nR)
	qtf := createArray(nR)
	work1 := createArray(nC)
	work2 := createArray(nC)
	work3 := createArray(nC)

	// Evaluate the function at the starting point and calculate its norm
	var current lmaEvaluation
	current.jacobian = jacobian(optium.start)
	current.residuals = value(optium.start)
	current.point = optium.start

	currentResiduals := getResiduals(current.residuals, optium.weightSquareRoot)
	currentCost := getCost(currentResiduals)
	currentPoint := optium.start

	firstIteration := true
	for {
		iterationCounter += 1
		if iterationCounter > maxIteration {
			return current, errors.New("there is no decision")
		}

		previous := current

		// QR decomposition of the jacobian matrix
		internalData := qrDecomposition(current.jacobian, optium.weightSquareRoot, solvedCols)
		weightedJacobian := internalData.weightedJacobian
		permutation := internalData.permutation
		diagR := internalData.diagR
		jacNorm := internalData.jacNorm

		// Residuals already have weights applied
		weightedResidual := currentResiduals
		for i := 0; i < nR; i++ {
			qtf[i] = weightedResidual[i]
		}

		// Compute Qt.res
		qTy(qtf, internalData)

		// Now we don't need Q anymore,
		// So let jacobian contain the R matrix with its diagonal elements
		for k := 0; k < solvedCols; k++ {
			pk := permutation[k]
			weightedJacobian[k][pk] = diagR[pk]
		}
		internalData.weightedJacobian = weightedJacobian

		if firstIteration {
			// Scale the point according to the norms of the columns
			// Of the initial jacobian
			xNorm = 0
			for k := 0; k < nC; k++ {
				dk := jacNorm[k]
				if dk == 0 {
					dk = 1.0
				}
				xk := dk * currentPoint[k]
				xNorm += xk * xk
				diag[k] = dk
			}

			xNorm = math.Sqrt(xNorm)
			// Initialize the step bound delta
			delta = initialStepBoundFactor * xNorm
			if xNorm == 0 {
				delta = initialStepBoundFactor
			}
		}

		// Check orthogonality between function vector and jacobian columns
		maxCosine := 0.0
		if currentCost != 0 {
			for j := 0; j < solvedCols; j++ {
				pj := permutation[j]
				s := jacNorm[pj]
				if s != 0 {
					sum := 0.0
					for i := 0; i <= j; i++ {
						sum += weightedJacobian[i][pj] * qtf[i]
					}
					maxCosine = math.Max(maxCosine, math.Abs(sum)/(s*currentCost))
				}
			}
		}

		if maxCosine <= orthoTolerance {
			// Convergence has been reached
			return current, nil
		}

		// Rescale if necessary
		for j := 0; j < nC; j++ {
			diag[j] = math.Max(diag[j], jacNorm[j])
		}

		// Inner loop
		ratio := 0.0
		for ratio < 1.0e-4 {
			// Save the state
			for j := 0; j < solvedCols; j++ {
				pj := permutation[j]
				oldX[pj] = currentPoint[pj]
			}

			previousCost := currentCost
			tmpVec := weightedResidual
			weightedResidual = oldRes
			oldRes = tmpVec

			// Determine the Levenberg-Marquardt parameter
			lmPar = determineLMParameter(qtf, delta, diag, internalData, solvedCols, work1, work2, work3, lmDir, &lmPar)

			// Compute the new point and the norm of the evolution direction
			lmNorm := 0.0
			for j := 0; j < solvedCols; j++ {
				pj := permutation[j]
				lmDir[pj] = -lmDir[pj]
				currentPoint[pj] = oldX[pj] + lmDir[pj]
				s := diag[pj] * lmDir[pj]
				lmNorm += s * s
			}
			lmNorm = math.Sqrt(lmNorm)
			// On the first iteration, adjust the initial step bound
			if firstIteration {
				delta = math.Min(delta, lmNorm)
			}

			// Evaluate the function at x + p and calculate its norm
			evaluationCounter += 1
			if evaluationCounter > maxEvaluation {
				return current, errors.New("there is no decision")
			}

			current.jacobian = jacobian(currentPoint)
			current.residuals = value(currentPoint)
			current.point = currentPoint

			currentResiduals = getResiduals(current.residuals, optium.weightSquareRoot)
			currentCost = getCost(currentResiduals)

			// Compute the scaled actual reduction
			var actRed = -1.0
			if 0.1*currentCost < previousCost {
				r := currentCost / previousCost
				actRed = 1.0 - r*r
			}

			// Compute the scaled predicted reduction
			// and the scaled directional derivative
			for j := 0; j < solvedCols; j++ {
				pj := permutation[j]
				dirJ := lmDir[pj]
				work1[j] = 0
				for i := 0; i < j; i++ {
					work1[i] += weightedJacobian[i][pj] * dirJ
				}
			}
			coeff1 := 0.0
			for j := 0; j < solvedCols; j++ {
				coeff1 += work1[j] * work1[j]
			}
			pc2 := previousCost * previousCost
			coeff1 /= pc2
			coeff2 := lmPar * lmNorm * lmNorm / pc2
			preRed := coeff1 + 2*coeff2
			dirDer := -(coeff1 + coeff2)

			// Ratio of the actual to the predicted reduction
			if preRed == 0 {
				ratio = 0
			} else {
				ratio = actRed / preRed
			}

			// Update the step bound
			if ratio <= 0.25 {
				tmp := 0.5
				if actRed < 0 {
					tmp = 0.5 * dirDer / (dirDer + 0.5*actRed)
				}

				if (0.1*currentCost >= previousCost) || (tmp < 0.1) {
					tmp = 0.1
				}
				delta = tmp * math.Min(delta, 10.0*lmNorm)
				lmPar /= tmp
			} else if (lmPar == 0) || (ratio >= 0.75) {
				delta = 2 * lmNorm
				lmPar *= 0.5
			}

			// Test for successful iteration
			if ratio >= 1.0e-4 {
				// Successful iteration, update the norm
				firstIteration = false
				xNorm = 0
				for k := 0; k < nC; k++ {
					xK := diag[k] * currentPoint[k]
					xNorm += xK * xK
				}
				xNorm = math.Sqrt(xNorm)
			} else {
				// Failed iteration, reset the previous values
				currentCost = previousCost
				for j := 0; j < solvedCols; j++ {
					pj := permutation[j]
					currentPoint[pj] = oldX[pj]
				}
				tmpVec = weightedResidual
				weightedResidual = oldRes
				oldRes = tmpVec
				// Reset "current" to previous values
				current = previous
			}

			// Default convergence criteria
			if (math.Abs(actRed) <= costRelativeTolerance &&
				preRed <= costRelativeTolerance &&
				ratio <= 2.0) ||
				delta <= parRelativeTolerance*xNorm {
				return current, nil
			}

			// Tests for termination and stringent tolerances
			if math.Abs(actRed) <= two_eps &&
				preRed <= two_eps &&
				ratio <= 2.0 {
				return current, errors.New("cost relative tolerance error")
			} else if delta <= two_eps*xNorm {
				return current, errors.New("par relative tolerance error")
				//throw LMAMathError.parRelativeTolerance
			} else if maxCosine <= two_eps {
				return current, errors.New("ortho tolerance error")
			}
		}
	}
}
func jacobian(point []float64) [][]float64 {
	var jacobian [][]float64
	for range lmaDistances {
		s := createArray(len(point))
		jacobian = append(jacobian, s)
	}

	for i := 0; i < len(jacobian); i++ {
		for j := 0; j < len(point); j++ {
			jacobian[i][j] = 2*point[j] - 2*lmaPositions[i][j]
		}
	}

	return jacobian
}

func value(point []float64) []float64 {
	resultPoint := createArray(len(lmaDistances))

	// Compute least squares
	for i := 0; i < len(resultPoint); i++ {
		resultPoint[i] = 0.0

		// Calculate sum, add to overall
		for j := 0; j < len(point); j++ {
			resultPoint[i] += (point[j] - lmaPositions[i][j]) * (point[j] - lmaPositions[i][j])
		}

		resultPoint[i] -= lmaDistances[i] * lmaDistances[i]
		resultPoint[i] *= -1
	}

	return resultPoint
}

func getResiduals(residuals []float64, weightSquareRoot []float64) []float64 {
	resultResiduals := createArray(len(residuals))
	for i := 0; i < len(residuals); i++ {
		resultResiduals[i] = residuals[i] * weightSquareRoot[i]
	}

	return resultResiduals
}

func getCost(residuals []float64) float64 {
	var dot float64
	for _, value := range residuals {
		dot += value * value
	}

	dot = math.Sqrt(dot)
	return dot
}

func qrDecomposition(jacobian [][]float64, weightSquareRoot []float64, solvedCols int) lmaInternalData {
	// Code in this function assumes that the weighted Jacobian is -(W^(1/2) J), hence the multiplication by -1
	weightedJacobian := jacobian

	for index, value := range jacobian {
		s := make([]float64, len(value))
		for i := range s {
			s[i] = value[i] * -weightSquareRoot[index]
		}

		weightedJacobian[index] = s
	}

	nR := len(weightedJacobian)
	nC := len(weightedJacobian[0])

	permutation := make([]int, nC)
	for i := range permutation {
		permutation[i] = 0
	}

	diagR := createArray(nC)
	jacNorm := createArray(nC)
	beta := createArray(nC)

	// Initializations
	for k := 0; k < nC; k++ {
		permutation[k] = k
		norm2 := 0.0
		for i := 0; i < nR; i++ {
			akk := weightedJacobian[i][k]
			norm2 += akk * akk
		}
		jacNorm[k] = math.Sqrt(norm2)
	}

	// Transform the matrix column after column
	for k := 0; k < nC; k++ {
		// Select the column with the greatest norm on active components
		nextColumn := -1
		ak2 := math.Inf(-1)
		for i := k; i < nC; i++ {
			norm2 := 0.0
			for j := k; j < nR; j++ {
				aki := weightedJacobian[j][permutation[i]]
				norm2 += aki * aki
			}

			if norm2 > ak2 {
				nextColumn = i
				ak2 = norm2
			}
		}

		if nextColumn == -1 {
			break
		}

		pk := permutation[nextColumn]
		permutation[nextColumn] = permutation[k]
		permutation[k] = pk

		// Choose alpha such that Hk.u = alpha ek
		akk := weightedJacobian[k][pk]
		alpha := math.Sqrt(ak2)
		if akk > 0 {
			alpha *= -1
		}

		betak := 1.0 / (ak2 - akk*alpha)
		beta[pk] = betak

		// Transform the current column
		diagR[pk] = alpha
		weightedJacobian[k][pk] -= alpha

		for dk := nC - 1 - k; dk > 0; dk-- {
			gamma := 0.0
			for j := k; j < nR; j++ {
				gamma += weightedJacobian[j][pk] * weightedJacobian[j][permutation[k+dk]]
			}
			gamma *= betak
			for j := k; j < nR; j++ {
				weightedJacobian[j][permutation[k+dk]] -= gamma * weightedJacobian[j][pk]
			}
		}
	}

	var internalData lmaInternalData
	internalData.weightedJacobian = weightedJacobian
	internalData.permutation = permutation
	internalData.rank = solvedCols
	internalData.diagR = diagR
	internalData.jacNorm = jacNorm
	internalData.beta = beta

	return internalData
}

func qTy(y []float64, internalData lmaInternalData) {
	weightedJacobian := internalData.weightedJacobian
	permutation := internalData.permutation
	beta := internalData.beta

	nR := len(weightedJacobian)
	nC := len(weightedJacobian[0])

	for k := 0; k < nC; k++ {
		pk := permutation[k]
		gamma := 0.0

		for i := k; i < nR; i++ {
			gamma += weightedJacobian[i][pk] * y[i]
		}
		gamma *= beta[pk]
		for i := k; i < nR; i++ {
			y[i] -= gamma * weightedJacobian[i][pk]
		}
	}
}

func determineLMParameter(qy []float64, delta float64, diag []float64, internalData lmaInternalData, solvedCols int, work1 []float64, work2 []float64, work3 []float64, lmDir []float64, lmPar *float64) float64 {
	safeMin := 2.2250738585072014e-308
	weightedJacobian := internalData.weightedJacobian
	permutation := internalData.permutation
	rank := internalData.rank
	diagR := internalData.diagR

	nC := len(weightedJacobian[0])

	// Compute and store in x the gauss-newton direction, if the
	// jacobian is rank-deficient, obtain a least squares solution
	for j := 0; j < rank; j++ {
		lmDir[permutation[j]] = qy[j]
	}
	for j := rank; j < nC; j++ {
		lmDir[permutation[j]] = 0
	}
	for k := rank - 1; k >= 0; k-- {
		pk := permutation[k]
		ypk := lmDir[pk] / diagR[pk]
		for i := 0; i < k; i++ {
			lmDir[permutation[i]] -= ypk * weightedJacobian[i][pk]
		}
		lmDir[pk] = ypk
	}

	// Evaluate the function at the origin, and test
	// for acceptance of the Gauss-Newton direction
	dxNorm := 0.0
	for j := 0; j < solvedCols; j++ {
		pj := permutation[j]
		s := diag[pj] * lmDir[pj]
		work1[pj] = s
		dxNorm += s * s
	}

	dxNorm = math.Sqrt(dxNorm)
	fp := dxNorm - delta
	if fp <= 0.1*delta {
		*lmPar = 0
		return *lmPar
	}

	// If the jacobian is not rank deficient, the Newton step provides
	// a lower bound, parl, for the zero of the function,
	// otherwise set this bound to zero
	sum2 := 0.0
	parl := 0.0
	if rank == solvedCols {
		for j := 0; j < solvedCols; j++ {
			pj := permutation[j]
			work1[pj] *= diag[pj] / dxNorm
		}
		sum2 = 0.0
		for j := 0; j < solvedCols; j++ {
			pj := permutation[j]
			var sum = 0.0
			for i := 0; i < j; i++ {
				sum += weightedJacobian[i][pj] * work1[permutation[i]]
			}
			s := (work1[pj] - sum) / diagR[pj]
			work1[pj] = s
			sum2 += s * s
		}
		parl = fp / (delta * sum2)
	}

	// Calculate an upper bound, paru, for the zero of the function
	sum2 = 0.0
	for j := 0; j < solvedCols; j++ {
		pj := permutation[j]
		sum := 0.0
		for i := 0; i < j; i++ {
			sum += weightedJacobian[i][pj] * qy[i]
		}
		sum /= diag[pj]
		sum2 += sum * sum
	}
	gNorm := math.Sqrt(sum2)
	paru := gNorm / delta
	if paru == 0 {
		paru = safeMin / math.Min(delta, 0.1)
	}

	// If the input par lies outside of the interval (parl,paru),
	// set par to the closer endpoint
	*lmPar = math.Min(paru, math.Max(*lmPar, parl))
	if *lmPar == 0 {
		*lmPar = gNorm / dxNorm
	}

	for k := 10; k >= 0; k-- {

		// Evaluate the function at the current value of lmPar
		if *lmPar == 0 {
			*lmPar = math.Max(safeMin, 0.001*paru)
		}
		sPar := math.Sqrt(*lmPar)
		for j := 0; j < solvedCols; j++ {
			pj := permutation[j]
			work1[pj] = sPar * diag[pj]
		}
		determineLMDirection(qy, work1, work2, internalData, solvedCols, work3, lmDir)

		dxNorm = 0.0
		for j := 0; j < solvedCols; j++ {
			pj := permutation[j]
			s := diag[pj] * lmDir[pj]
			work3[pj] = s
			dxNorm += s * s
		}
		dxNorm = math.Sqrt(dxNorm)
		previousFP := fp
		fp = dxNorm - delta

		// If the function is small enough, accept the current value
		// of lmPar, also test for the exceptional cases where parl is zero
		if math.Abs(fp) <= 0.1*delta ||
			(parl == 0 &&
				fp <= previousFP &&
				previousFP < 0) {
			return *lmPar
		}

		// Compute the Newton correction
		for j := 0; j < solvedCols; j++ {
			pj := permutation[j]
			work1[pj] = work3[pj] * diag[pj] / dxNorm
		}
		for j := 0; j < solvedCols; j++ {
			pj := permutation[j]
			work1[pj] /= work2[j]
			tmp := work1[pj]
			for i := j + 1; i < solvedCols; i++ {
				work1[permutation[i]] -= weightedJacobian[i][pj] * tmp
			}
		}
		sum2 = 0.0
		for j := 0; j < solvedCols; j++ {
			s := work1[permutation[j]]
			sum2 += s * s
		}
		correction := fp / (delta * sum2)

		// Depending on the sign of the function, update parl or paru
		if fp > 0 {
			parl = math.Max(parl, *lmPar)
		} else if fp < 0 {
			paru = math.Min(paru, *lmPar)
		}

		// Compute an improved estimate for lmPar
		*lmPar = math.Max(parl, *lmPar+correction)
	}

	return *lmPar
}

func determineLMDirection(qy []float64, diag []float64, lmDiag []float64, internalData lmaInternalData, solvedCols int, work []float64, lmDir []float64) {
	permutation := internalData.permutation
	weightedJacobian := internalData.weightedJacobian
	diagR := internalData.diagR

	// Copy R and Qty to preserve input and initialize s
	// in particular, save the diagonal elements of R in lmDir
	for j := 0; j < solvedCols; j++ {
		pj := permutation[j]
		for i := j + 1; i < solvedCols; i++ {
			weightedJacobian[i][pj] = weightedJacobian[j][permutation[i]]
		}
		lmDir[j] = diagR[pj]
		work[j] = qy[j]
	}

	// Eliminate the diagonal matrix d using a Givens rotation
	for j := 0; j < solvedCols; j++ {

		// Prepare the row of d to be eliminated, locating the
		// diagonal element using p from the Q.R. factorization
		pj := permutation[j]
		dpj := diag[pj]
		if dpj != 0 {
			for k := j + 1; k < len(lmDiag); k++ {
				lmDiag[k] = 0
			}
		}
		lmDiag[j] = dpj

		// The transformations to eliminate the row of d
		// modify only a single element of Qty
		// beyond the first n, which is initially zero.
		qtbpj := 0.0
		for k := j; k < solvedCols; k++ {
			pk := permutation[k]

			// Determine a Givens rotation which eliminates the
			// appropriate element in the current row of d
			if lmDiag[k] != 0 {

				sin := 0.0
				cos := 0.0
				rkk := weightedJacobian[k][pk]
				if math.Abs(rkk) < math.Abs(lmDiag[k]) {
					cotan := rkk / lmDiag[k]
					sin = 1.0 / math.Sqrt(1.0+cotan*cotan)
					cos = sin * cotan
				} else {
					tan := lmDiag[k] / rkk
					cos = 1.0 / math.Sqrt(1.0+tan*tan)
					sin = cos * tan
				}

				// Compute the modified diagonal element of R and
				// the modified element of (Qty,0)
				weightedJacobian[k][pk] = cos*rkk + sin*lmDiag[k]
				temp := cos*work[k] + sin*qtbpj
				qtbpj = -sin*work[k] + cos*qtbpj
				work[k] = temp

				// Accumulate the tranformation in the row of s
				for i := k + 1; i < solvedCols; i++ {
					rik := weightedJacobian[i][pk]
					temp2 := cos*rik + sin*lmDiag[i]
					lmDiag[i] = -sin*rik + cos*lmDiag[i]
					weightedJacobian[i][pk] = temp2
				}
			}
		}

		// Store the diagonal element of s and restore
		// the corresponding diagonal element of R
		lmDiag[j] = weightedJacobian[j][permutation[j]]
		weightedJacobian[j][permutation[j]] = lmDir[j]
	}

	// Solve the triangular system for z, if the system is
	// singular, then obtain a least squares solution
	nSing := solvedCols
	for j := 0; j < solvedCols; j++ {
		if (lmDiag[j] == 0) && (nSing == solvedCols) {
			nSing = j
		}
		if nSing < solvedCols {
			work[j] = 0
		}
	}
	if nSing > 0 {
		for j := nSing - 1; j >= 0; j-- {
			pj := permutation[j]
			sum := 0.0
			for i := j + 1; i < nSing; i++ {
				sum += weightedJacobian[i][pj] * work[i]
			}
			work[j] = (work[j] - sum) / lmDiag[j]
		}
	}

	// Permute the components of z back to components of lmDir
	for j := 0; j < len(lmDir); j++ {
		lmDir[permutation[j]] = work[j]
	}
}
