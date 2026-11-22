package jpegxs

import (
	"bytes"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Real Big Buck Bunny output frame 31, strip 25, phase 16. At Q=4 its
// entropy contains ff20, which is data rather than a new slice marker.
// The expected mosaic was decoded independently by the Rust bridge.
func TestDecodeQ4EntropyMarkerBytes(t *testing.T) {
	raw, err := readHexFixture("testdata/q4-entropy-marker.hex")
	if err != nil {
		t.Fatal(err)
	}
	want, err := readHexFixture("testdata/q4-entropy-marker-decoded.hex")
	if err != nil {
		t.Fatal(err)
	}
	coded, err := EncodeRGGB8Block(raw, &Options{Quantization: 4, LevelsX: 1, LevelsY: 0,
		StarTetrix: true, Br: 4, CTS: CTSParams{CT: 0}})
	if err != nil {
		t.Fatal(err)
	}
	cs, err := parseCodestream(coded)
	if err != nil {
		t.Fatal(err)
	}
	if len(cs.slices) != 1 || !bytes.Contains(cs.slices[0].data, []byte{0xff, 0x20}) {
		t.Fatal("regression fixture must contain ff20 inside entropy")
	}
	got, err := DecodeRGGB8Block(coded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("Q=4 decoded mosaic differs from the Rust reference")
	}
	start := len(coded) - 2 - len(cs.slices[0].data)
	for _, marker := range []byte{0x11, 0x20} {
		changed := append([]byte(nil), coded...)
		// Change payload only; precinct length and the real final EOC stay intact.
		changed[start+15], changed[start+16] = 0xff, marker
		parsed, e := parseCodestream(changed)
		if e != nil || len(parsed.slices) != 1 || len(parsed.slices[0].data) != len(cs.slices[0].data) {
			t.Fatalf("entropy ff%02x was mistaken for a marker: %v", marker, e)
		}
	}
	if _, err := DecodeRGGB8Block(coded[:len(coded)-3]); err == nil {
		t.Fatal("truncated precinct accepted")
	}
	invalid := append([]byte(nil), coded...)
	invalid[start], invalid[start+1], invalid[start+2] = 0, 0, 0
	if _, err := DecodeRGGB8Block(invalid); err == nil {
		t.Fatal("zero precinct length accepted")
	}
}

// decodeTestBlock returns a deterministic, non-degenerate 160 × 4 RGGB block.
func decodeTestBlock(seed uint32) []byte {
	raw := make([]byte, RawBayerBlockWidth*RawBayerBlockHeight)
	s := seed*1664525 + 1013904223
	for i := range raw {
		s = s*1664525 + 1013904223
		raw[i] = byte(s >> 24)
	}
	return raw
}

// TestDecodeRoundTripLossless checks that Q = 0 survives encode → decode
// bit-exactly. This exercises the whole decode path: header parsing, the
// entropy decoder, the inverse dequantiser, the inverse 5/3 DWT, the
// inverse Star-Tetrix and the Annex G output scaling.
func TestDecodeRoundTripLossless(t *testing.T) {
	for _, seed := range []uint32{1, 7, 123456} {
		raw := decodeTestBlock(seed)
		coded, err := EncodeRGGB8Block(raw, &Options{
			Quantization: 0,
			LevelsX:      1,
			LevelsY:      0,
			StarTetrix:   true,
			CTS:          CTSParams{CT: 0},
		})
		if err != nil {
			t.Fatalf("seed %d: encode: %v", seed, err)
		}
		got, err := DecodeRGGB8Block(coded)
		if err != nil {
			t.Fatalf("seed %d: decode: %v", seed, err)
		}
		if !bytes.Equal(got, raw) {
			diff := 0
			for i := range raw {
				if raw[i] != got[i] {
					diff++
				}
			}
			t.Fatalf("seed %d: Q=0 round trip is not bit-exact (%d of %d bytes differ)", seed, diff, len(raw))
		}
	}
}

// TestDecodeLossyIsClose checks that a lossy decode stays in the
// neighbourhood of the source and produces a plausible image.
func TestDecodeLossyIsClose(t *testing.T) {
	raw := decodeTestBlock(7)
	coded, err := EncodeRGGB8Block(raw, &Options{
		Quantization: 6,
		LevelsX:      1,
		LevelsY:      0,
		StarTetrix:   true,
		CTS:          CTSParams{CT: 0},
	})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := DecodeRGGB8Block(coded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(raw) {
		t.Fatalf("decoded length %d != %d", len(got), len(raw))
	}
	var sum int64
	for i := range raw {
		d := int64(raw[i]) - int64(got[i])
		if d < 0 {
			d = -d
		}
		sum += d
	}
	mae := float64(sum) / float64(len(raw))
	if mae > 24 {
		t.Fatalf("Q=6 mean absolute error %.2f is too large for a single-level decode", mae)
	}
	t.Logf("Q=6 mean absolute error: %.2f", mae)
}

// TestDecodeMatchesRustBridge compares the Go decoder against the upstream
// Rust decoder for the same codestream. It needs the bridge binary, so it
// is skipped when JPEgXS_BRIDGE is not set.
func TestDecodeMatchesRustBridge(t *testing.T) {
	bridge := os.Getenv("JPEgXS_BRIDGE")
	if bridge == "" {
		t.Skip("JPEgXS_BRIDGE is not set")
	}
	dir := t.TempDir()
	for _, q := range []uint8{0, 1, 4, 6, 15} {
		for _, br := range []uint8{4, 8} {
			for _, seed := range []uint32{1, 7, 999} {
				raw := decodeTestBlock(seed)
				coded, err := EncodeRGGB8Block(raw, &Options{
					Quantization: q,
					LevelsX:      1,
					LevelsY:      0,
					StarTetrix:   true,
					Br:           br,
					CTS:          CTSParams{CT: 0},
				})
				if err != nil {
					t.Fatalf("q=%d br=%d seed=%d: encode: %v", q, br, seed, err)
				}
				in := filepath.Join(dir, "in.jxs")
				out := filepath.Join(dir, "out.rggb8")
				if err := os.WriteFile(in, coded, 0o600); err != nil {
					t.Fatal(err)
				}
				_ = os.Remove(out)
				cmd := exec.Command(bridge, "decode", "--input", in, "--output", out)
				if b, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("q=%d br=%d seed=%d: bridge decode: %v: %s", q, br, seed, err, b)
				}
				want, err := os.ReadFile(out)
				if err != nil {
					t.Fatal(err)
				}
				got, err := DecodeRGGB8Block(coded)
				if err != nil {
					t.Fatalf("q=%d br=%d seed=%d: go decode: %v", q, br, seed, err)
				}
				if !bytes.Equal(got, want) {
					diff := 0
					for i := range want {
						if i < len(got) && want[i] != got[i] {
							diff++
						}
					}
					t.Fatalf("q=%d br=%d seed=%d: Go and Rust decoders differ in %d of %d bytes", q, br, seed, diff, len(want))
				}
			}
		}
	}
}

// readHexFixture keeps byte-exact regression vectors in Git as reviewable text.
func readHexFixture(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return hex.DecodeString(strings.Join(strings.Fields(string(data)), ""))
}
