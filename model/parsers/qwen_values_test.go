package parsers

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestTypedToolNumbersRemainRepresentable(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want any
	}{
		{"9007199254740993", int64(9007199254740993)},
		{"9223372036854775807", int64(math.MaxInt64)},
		{"-9223372036854775808", int64(math.MinInt64)},
		{"9223372036854775808", float64(0x1p63)},
		{"-9223372036854777856", float64(-0x1p63 - 2048)},
		{"9.223372036854775808e18", float64(0x1p63)},
		{"1e100", float64(1e100)},
		{"-1e100", float64(-1e100)},
		{"NaN", "NaN"},
		{"+Inf", "+Inf"},
		{"-Infinity", "-Infinity"},
		{"1e400", "1e400"},
		{"100.0", int(100)},
		{"1.23e5", int(123000)},
		{"0.125", float64(0.125)},
	} {
		for _, types := range []api.PropertyType{{"number"}, {"string", "number"}, {"integer", "number"}} {
			t.Run(tc.raw+"/"+strings.Join(types, "+"), func(t *testing.T) {
				got := parseTypedToolValue(tc.raw, types)
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("got %v (%T), want %v (%T)", got, got, tc.want, tc.want)
				}
				if _, err := json.Marshal(got); err != nil {
					t.Fatalf("tool value cannot be sent as JSON: %v", err)
				}
			})
		}
	}
}

// Keep a literal reference for invalid UTF-8 as well as ordinary tool text.
// The parser historically replaces each invalid byte with a replacement rune.
func referenceEscapeToolText(s string) string {
	var out strings.Builder
	for _, r := range s {
		switch r {
		case '&':
			out.WriteString("&amp;")
		case '<':
			out.WriteString("&lt;")
		case '>':
			out.WriteString("&gt;")
		default:
			out.WriteRune(r)
		}
	}
	return out.String()
}

func TestEscapeToolTextPreservesBytesAndMalformedUTF8(t *testing.T) {
	inputs := []string{"", "plain text", "ação 日本語 🙂\t\r\n", "<tag>&amp;</tag>", "\x00\xff\xfe&\xc3<\xa9>", strings.Repeat("x", 8192)}
	rng := rand.New(rand.NewPCG(1, 2))
	for range 256 {
		input := make([]byte, rng.IntN(256))
		for i := range input {
			input[i] = byte(rng.Uint32())
		}
		inputs = append(inputs, string(input))
	}
	for i, input := range inputs {
		var got strings.Builder
		got.WriteString("prefix:")
		escapeTextNode(&got, input)
		if want := "prefix:" + referenceEscapeToolText(input); got.String() != want {
			t.Fatalf("case %d: got %q, want %q", i, got.String(), want)
		}
	}
}

func BenchmarkEscapeToolText(b *testing.B) {
	for _, size := range []int{8 << 10, 256 << 10} {
		for _, tc := range []struct{ name, unit string }{{"ascii", "abcdefgh"}, {"unicode", "ação 🙂 "}, {"escaped", "a<&>bc"}} {
			b.Run(fmt.Sprintf("%s/%dKiB", tc.name, size>>10), func(b *testing.B) {
				input := strings.Repeat(tc.unit, size/len(tc.unit))
				b.SetBytes(int64(len(input)))
				b.ReportAllocs()
				for b.Loop() {
					var out strings.Builder
					escapeTextNode(&out, input)
				}
			})
		}
	}
}
