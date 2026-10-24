package server

import (
	"math"
	"time"
)

// Lower value means less recent work is lost per byte released.
func runnerEvictionValue(runner *runnerRef, now time.Time) float64 {
	if runner.loadDuration <= 0 || runner.lastUsed.IsZero() {
		return -1
	}
	if runner.sessionDuration == 0 {
		return 0
	}
	age := max(0, now.Sub(runner.lastUsed).Seconds())
	reuse := 1 + math.Log2(1+float64(runner.useCount))
	bytes := max(runner.vramSize, runner.totalSize)
	if bytes == 0 {
		return -1
	}
	value := runner.loadDuration.Seconds() * reuse / (1 + age/60) / float64(bytes)
	if runner.sessionDuration < 0 {
		value *= 2
	}
	return value
}
