package server

import (
	"math"
	"slices"
	"testing"
)

func TestNormalizeFiniteOverflow(t *testing.T) {
	for _, input := range [][]float32{
		{math.MaxFloat32, math.MaxFloat32},
		{math.MaxFloat32, -math.MaxFloat32},
		{math.MaxFloat32, 1, -2},
		{1e19, 1e19, -1e19, -1e19},
	} {
		actual, err := normalize(slices.Clone(input))
		if err != nil {
			t.Fatal(err)
		}
		var squared float64
		for i, value := range actual {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("finite input yielded invalid result: %v", actual)
			}
			if value == 0 && input[i] != 0 {
				t.Fatalf("finite nonzero component disappeared: %v", actual)
			}
			if math.Signbit(float64(value)) != math.Signbit(float64(input[i])) {
				t.Fatalf("component sign changed: %v", actual)
			}
			squared += float64(value) * float64(value)
		}
		if math.Abs(squared-1) > 2e-7 {
			t.Fatalf("large finite vector is not unit length: norm2=%g vector=%v", squared, actual)
		}
	}
}

func TestNormalizeOrdinaryBitsPreserved(t *testing.T) {
	inputs := [][]float32{
		{}, {0, 0, 0}, {float32(math.Copysign(0, -1)), 0},
		{3, 4}, {-0.1, 0.2, 0.3, -0.4},
		{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32},
		{1e-30, -2e-30, 3e-30}, {1e-15, 1e-12, -1e-13},
		{1e18, -1e18, 3},
	}
	for n := 1; n <= 1536; n *= 2 {
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(math.Sin(float64(i+1)) * 0.75)
		}
		inputs = append(inputs, input)
	}
	for _, input := range inputs {
		// Freeze the previous float32 path, including its tiny-vector floor.
		want := slices.Clone(input)
		var sum float32
		for _, value := range want {
			sum += value * value
		}
		norm := float32(1.0 / max(math.Sqrt(float64(sum)), 1e-12))
		for i := range want {
			want[i] *= norm
		}
		actual, err := normalize(slices.Clone(input))
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Float32bits(actual[i]) != math.Float32bits(want[i]) {
				t.Fatalf("ordinary normalization bits changed at%d: got%g want%g", i, actual[i], want[i])
			}
		}
	}
}

func TestNormalizeNonfiniteRejectedBeforeMutation(t *testing.T) {
	for _, invalid := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		input := []float32{math.MaxFloat32, invalid, 1}
		actual, err := normalize(input)
		if err == nil || actual != nil {
			t.Fatalf("nonfinite input accepted: %v %v", actual, err)
		}
		if input[0] != math.MaxFloat32 || input[2] != 1 {
			t.Fatal("invalid input was partially normalized")
		}
	}
}
