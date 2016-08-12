/* ****************************************************************************
A simulated annealing algorithm to solve sudoku puzzles.

Copyright (c) 2016 Everett Robinson

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
* ****************************************************************************/

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// Modified from https://stackoverflow.com/questions/9862443/golang-is-there-a-better-way-read-a-file-of-integers-into-an-array
// Read in the start state of the sudoku puzzle (of arbitrary dimension)
func readInOneLine(r io.Reader, line int, blockXDim int, blockYDim int) (puzzle [][]int, e error) {

	scanner := bufio.NewScanner(r)
	scanner.Split(bufio.ScanLines)

	// Start puzzle at line 1 (more user friendly)
	lineCounter := 1

	// For each line
	for scanner.Scan() {

		// Check if it's the line we selected
		if line == lineCounter {

			// Read the puzzle text in and split it into it's components
			puzzleText := scanner.Text()
			puzzleElements := strings.Split(puzzleText, "")
			puzzleDim := blockXDim * blockYDim
			puzzle = make([][]int, puzzleDim)

			for i := 0; i < puzzleDim; i++ {
				puzzle[i] = make([]int, puzzleDim)
				for j := 0; j < puzzleDim; j++ {
					value, err := strconv.Atoi(puzzleElements[(i*puzzleDim)+j])
					if err == nil {
						puzzle[i][j] = value
					} else {
						puzzle[i][j] = 0
					}
				}
			}
		}

		lineCounter++
	}

	return puzzle, scanner.Err()
}


func printPuzzle(puzzle [][]int, blockXDim int, blockYDim int) {

	for r := range puzzle {
		if r > 0 && r % blockYDim == 0 {
			fmt.Printf("---------------------\n")
		}
		for c := range puzzle[r] {
			if c > 0 && c % blockXDim == 0 {
				fmt.Printf("| ")
			}
			if puzzle[r][c] > 0 {
				fmt.Printf("%v ", puzzle[r][c])
			} else {
				fmt.Printf("  ")
			}
		}
		fmt.Printf("\n")
	}

}

// Starts n annealing goroutines at exponentially increasing temperatures 2^n where n is defined by the
// concurrentAnnealerCount value passed to the function. Once each annealing goroutine is returned any
// hotter goroutines with lower costs than their cooler neighbours will trade their candidate solutions
// with that neighbour.
func anneal(originalPuzzle [][]int, blockXDim int, blockYDim int, baseTemperature float64, coolingRate float64, internalIterations int, swapCount int, concurrentAnnealerCount int) (solvedPuzzle [][]int, solutionFound bool) {

	initialSolution := randomInitialization(originalPuzzle)

	baseTemperature = baseTemperature
	finalTemperature := 0.00001
	coolingRate = coolingRate

	// Create a channel for the concurrent annealers of differing temperatures
	annealerSolution := make(chan [][]int)
	annealerCost := make(chan float64)

	annealerSolutions := make([][][]int, concurrentAnnealerCount)
	annealerCosts := make([]float64, concurrentAnnealerCount)

	for i := 0; i < concurrentAnnealerCount; i++ {
		annealerSolutions[i] = copyPuzzle(initialSolution)
		annealerCosts[i] = costFunction(initialSolution, blockXDim, blockYDim)
	}

	// While the cost is not zero and we haven't hit our final temperature
	for baseTemperature > finalTemperature {

		for i := 0; i < concurrentAnnealerCount; i++ {
			go annealerInternalIterator(originalPuzzle, annealerSolutions[i], blockXDim, blockYDim, baseTemperature*math.Pow(2, float64(i)), internalIterations, swapCount, annealerSolution, annealerCost)
			annealerSolutions[i] = <- annealerSolution
			annealerCosts[i] = <- annealerCost
		}

		// If a hotter goroutine has a better solution than a colder one then we swap the solutions
		for i := concurrentAnnealerCount - 1; i > 0; i-- {
			if annealerCosts[i] < annealerCosts[i-1] {
				annealerSolutions[i], annealerSolutions[i-1] = annealerSolutions[i-1], annealerSolutions[i]
				annealerCosts[i], annealerCosts[i-1] = annealerCosts[i-1], annealerCosts[i]
			}
		}

		// If the coldest goroutine has cost zero then we have solved the puzzle
		if annealerCosts[0] == 0 {
			return annealerSolutions[0], true
		}

		// Cool all of the goroutines
		baseTemperature = baseTemperature * coolingRate
	}

	return annealerSolutions[0], false
}

// Gets a neighbouring candidate solution and runs the probibalistic steps of the annealing process as many times as
// specified by the internalIterations count.
func annealerInternalIterator(originalPuzzle [][]int, candidateSolution [][]int, blockXDim int, blockYDim int, temperature float64, internalIterations int, swapCount int, as chan [][]int, ac chan float64) {

	// Set updatedSolution and updatedCost to the current values associated with candidateSolution
	updatedSolution := copyPuzzle(candidateSolution)
	updatedCost := costFunction(updatedSolution, blockXDim, blockYDim)

	for i := 0; i < internalIterations; i++ {
		newCandidateSolution := getNeighbour(updatedSolution, swapCount, originalPuzzle)
		newCandidateCost := costFunction(newCandidateSolution, blockXDim, blockYDim)

		// If the cost is zero, then we found a viable solution. exit!
		if newCandidateCost == 0 {
			as <- newCandidateSolution
			ac <- 0
			return
		}

		// Otherwise, if the cost is less then switch to that solution
		if newCandidateCost < updatedCost {
			updatedSolution = newCandidateSolution
			updatedCost = newCandidateCost

		// And finally switch to a more costly solution randomly based on the acceptance probablity
		} else {
			ap := acceptanceProbability(updatedCost, newCandidateCost, temperature)

			if ap > rand.Float64() {
				updatedSolution = newCandidateSolution
				updatedCost = newCandidateCost
			}
		}
	}

	as <- updatedSolution
	ac <- updatedCost
	return
}

// Gets a neighbouring candidate solution to the current one by randomly swapping two numbers in the puzzle.
// It also ensures that the neighbouring solution created does not modify or swap one of the clues in the
// original puzzle.
func getNeighbour(currentPuzzle [][]int, swapCount int, originalPuzzle [][]int) (neighbourPuzzle [][]int) {

	puzzleDim := len(originalPuzzle)

	// Copy the current puzzle into neighbourPuzzle
	neighbourPuzzle = copyPuzzle(currentPuzzle)

	for i := 0; i < swapCount; i++ {
		randomXIndex1 := rand.Intn(puzzleDim)
		randomYIndex1 := rand.Intn(puzzleDim)

		randomXIndex2 := rand.Intn(puzzleDim)
		randomYIndex2 := rand.Intn(puzzleDim)

		// Keep randomly reassigning the index until we get one that wasn't defined in the
		// original puzzle.
		for originalPuzzle[randomXIndex1][randomYIndex1] > 0 {
			randomXIndex1 = rand.Intn(puzzleDim)
			randomYIndex1 = rand.Intn(puzzleDim)
		}

		for originalPuzzle[randomXIndex2][randomYIndex2] > 0 {
			randomXIndex2 = rand.Intn(puzzleDim)
			randomYIndex2 = rand.Intn(puzzleDim)
		}

		// Swap the two randomly selected elements
		neighbourPuzzle[randomXIndex1][randomYIndex1], neighbourPuzzle[randomXIndex2][randomYIndex2] = neighbourPuzzle[randomXIndex2][randomYIndex2], neighbourPuzzle[randomXIndex1][randomYIndex1]
	}

	return neighbourPuzzle
}

// A cost function for the provided sudoku puzzle. The cost is defined as the sum over all rows, columns
// and blocks of the  absolute difference between the occurances of a number in that row block or column
// and it's expected occurance of 1. A cost of zero for the whole puzzle indicates that it has been solved.
func costFunction(puzzle [][]int, blockXDim int, blockYDim int) (cost float64) {

	// Figure out the full dimension of the puzzle from the passed block dimensions
	puzzleDim := blockXDim * blockYDim

	// Initialize the cost to zero
	cost = 0.0

	// For each row and column: (takes advantage of the square nature of the puzzle by swapping the
	// two iterators)
	for dim1 := 0; dim1 < puzzleDim; dim1++ {

		// Create two slices to track the occurances of each number by row and by column.
		// Numbers are shifted down by one, so 1 is stored in index 0, 2 in index 1, and so forth.
		rowCounts := make([]int, puzzleDim, puzzleDim)
		columnCounts := make([]int, puzzleDim, puzzleDim)

		// For each entry in this row or column
		for dim2 := 0; dim2 < puzzleDim; dim2++ {

			// rows
			if puzzle[dim1][dim2] > 0 {
				number := puzzle[dim1][dim2]
				rowCounts[number-1]++
			}

			// columns
			if puzzle[dim2][dim1] > 0 {
				number := puzzle[dim2][dim1]
				columnCounts[number-1]++
			}
		}

		// Figure out the cost for this row
		for _, count := range rowCounts {
			cost += math.Abs(float64(count - 1))
		}

		// And the cost for this column
		for _, count := range columnCounts {
			cost += math.Abs(float64(count - 1))
		}
	}

	// Also figure out the cost for each block in the puzzle (puzzleDim = number of blocks)
	horizontalBlockCount := blockYDim
	verticalBlockCount := blockXDim

	// For each block in the horizontal
	for i := 0; i < horizontalBlockCount; i++ {

		// For each block in the vertical
		for j := 0; j < verticalBlockCount; j++ {

			// Keep track of the occurances of each number for the given block
			blockCounts := make([]int, puzzleDim, puzzleDim)

			for k := 0; k < puzzleDim; k++ {

				horizontalIndex := i * blockXDim + k % blockXDim
				verticalIndex := j * blockYDim + k / blockXDim

				if puzzle[horizontalIndex][verticalIndex] > 0 {
					number := puzzle[horizontalIndex][verticalIndex]
					blockCounts[number-1]++
				}
			}

			// The cost for this block
			for _, count := range blockCounts {
				cost += math.Abs(float64(count - 1))
			}
		}
	}

	return cost
}


func acceptanceProbability(oldCost float64, newCost float64, temperature float64) (probability float64) {
	return math.Exp((oldCost - newCost) / temperature)
}

// Randomly sets all blank values in the original puzzle to number within the
// dimension of the puzzle so the anneaing function has a complete (but incorrect)
// base to start from. It ensures that the occurances of each number is correct
// for the puzzle. Eg. for a standard sudoku, there will be 9 of each number.
func randomInitialization(originalPuzzle [][]int) (initializedPuzzle [][]int) {

	puzzleDim := len(originalPuzzle)

	remainingNumbers := make(map[int]int)

	var emptySpots [][]int
	emptySpots = make([][]int,0)

	initializedPuzzle = make([][]int, puzzleDim)

	// Set all of the remainingNumbers to the maximum possible (puzzleDim)
	for i := 1; i <= puzzleDim; i++ {
		remainingNumbers[i] = puzzleDim
	}

	// For each occurance of a number in the originalPuzzle, subtract one from
	// remainingNumbers and duplicate it in the initializedPuzzle
	for i := 0; i < puzzleDim; i++ {
		initializedPuzzle[i] = make([]int, puzzleDim)
		for j := 0; j < puzzleDim; j++ {
			if originalPuzzle[i][j] > 0 {
				remainingNumbers[originalPuzzle[i][j]]--
				initializedPuzzle[i][j] = originalPuzzle[i][j]

			} else {
				emptySpots = append(emptySpots, []int{i, j})
			}
		}
	}

	// For every remaining number, randomly assign it to one of the remaining empty spots
	// then delete that empty spot from the slice
	for remainingNumber, count := range remainingNumbers {
		for i := 0; i < count; i++ {
			spotIndex := rand.Intn(len(emptySpots))
			spot := emptySpots[spotIndex]
			initializedPuzzle[spot[0]][spot[1]] = remainingNumber
			emptySpots[spotIndex] = emptySpots[len(emptySpots)-1]
			emptySpots = emptySpots[0 : len(emptySpots)-1]
		}
	}

	return initializedPuzzle
}


func copyPuzzle(originalPuzzle [][]int) (copiedPuzzle [][]int) {

	puzzleDim := len(originalPuzzle)

	copiedPuzzle = make([][]int, puzzleDim, puzzleDim)

	for i := 0; i < puzzleDim; i++ {
		copiedPuzzle[i] = make([]int, puzzleDim, puzzleDim)
		copy(copiedPuzzle[i], originalPuzzle[i])
	}

	return copiedPuzzle
}

func main() {
	// Seed the random number generator for use throughout the program.
	rand.Seed(time.Now().Unix())

	start := time.Now()

	inputModePtr := flag.String("m", "one-line", "An input mode used to interpret the input file")
	dimPtr := flag.String("d", "3x3", "The dimensions of one of the puzzle blocks (eg. standard sudoku is 3x3)")
	filePtr := flag.String("f", "puzzles.txt", "The filename to be checked")
	linePtr := flag.String("l", "1", "The line of the puzzle to be solved")
	temperaturePtr := flag.String("t", "1.0", "The lowest base temperature for the concurrent annealers (temperature increases by 2^i for each goroutine i)")
	coolingRatePtr := flag.String("c", "0.9", "The rate of cooling for each step in the annealing process (a number greater than 0 and less than 1)")
	iterationPtr := flag.String("i", "1000", "The number of iterations at each step of the annealing process")
	swapPtr := flag.String("s", "1", "The number of swaps in each iteration of the anneling process")
	concurrentAnnealerPtr := flag.String("a", "6", "The number of concurrent annealing goroutines")
	trainingModePtr := flag.Bool("training-mode", false, "Enables a minimal output indicating only if a solution was found and how long that result took in seconds."+
		" Intended for collecting data to determine the optimal combination of the other flags.")

	flag.Parse()

	puzzleLine, _ := strconv.Atoi(*linePtr)
	puzzleDim := strings.Split(*dimPtr, "x")
	baseTemperature, _ := strconv.ParseFloat(*temperaturePtr, 64)
	coolingRate,_ := strconv.ParseFloat(*coolingRatePtr, 64)
	internalIterations, _ := strconv.Atoi(*iterationPtr)
	swapCount,_ := strconv.Atoi(*swapPtr)
	annealerCount, _ := strconv.Atoi(*concurrentAnnealerPtr)
	blockXDim, _ := strconv.Atoi(puzzleDim[0])
	blockYDim, _ := strconv.Atoi(puzzleDim[1])

	inFile, err := os.Open(*filePtr)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	var originalPuzzle [][]int

	if *inputModePtr == "one-line" {
		// Read the file into an array
		originalPuzzle, err = readInOneLine(inFile, puzzleLine, blockXDim, blockYDim)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	} else {
		fmt.Println("No appropriate input mode for the puzzle was entered.")
		os.Exit(1)
	}

	if !*trainingModePtr {
		fmt.Println()
		fmt.Println("Original Puzzle:")
		printPuzzle(originalPuzzle, blockXDim, blockYDim)
		fmt.Printf("\nPuzzle cost: %v\n", costFunction(originalPuzzle, blockXDim, blockYDim))
	}

	solvedPuzzle, successfullySolved := anneal(originalPuzzle, blockXDim, blockYDim, baseTemperature, coolingRate, internalIterations, swapCount, annealerCount)

	if !*trainingModePtr {
		if successfullySolved {
			fmt.Println()
			fmt.Println("Solved Puzzle:")
			printPuzzle(solvedPuzzle, blockXDim, blockYDim)
		} else {
			fmt.Println()
			fmt.Println("No viable solution to the puzzle was found.\n")
			fmt.Printf("Final puzzle candidate:\n")
			printPuzzle(solvedPuzzle, blockXDim, blockYDim)
			fmt.Println()
			fmt.Printf("Cost at end: %v\n\n", costFunction(solvedPuzzle, blockXDim, blockYDim))
		}
	}

	elapsed := time.Since(start)

	if !*trainingModePtr {
		fmt.Printf("Execution completed in %s \n", elapsed)
	} else {
		// Return a csv line of the form
		// puzzleLine, baseTemperature, coolingRate, internalIterations, swapCount, annealerCount, solved, time
		fmt.Printf("%v,%v,%v,%v,%v,%v,%v,%v\n", puzzleLine, baseTemperature, coolingRate, internalIterations, swapCount, annealerCount, successfullySolved, elapsed.Seconds())
	}
}
