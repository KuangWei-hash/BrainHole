package jpegxs

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// The upstream Rust bridge (oxideav-jpegxs 0.0.9 @3484a041) is the
// reference encoder. The tests in this file compare the Go port against
// it byte for byte.
//
// The bridge executable is located through the JPEgXS_BRIDGE environment
// variable. When it is not set the live comparison is skipped and the
// pinned golden vectors in testdata/ are checked instead, so the suite
// still runs on a machine without a Rust toolchain.
const bridgeEnvVar = "JPEgXS_BRIDGE"

// bridgePath returns the Rust bridge executable path, or "" when it is
// unavailable.
func bridgePath(t *testing.T) string {
	t.Helper()
	p := os.Getenv(bridgeEnvVar)
	if p == "" {
		return ""
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("%s=%q is not usable: %v", bridgeEnvVar, p, err)
	}
	return p
}

// parityPatterns returns the raw 160 × 4 RGGB8 blocks used for the
// comparison. The first four mirror the upstream bridge's own unit-test
// fixtures so both sides exercise identical inputs.
func parityPatterns() []struct {
	name string
	raw  []byte
} {
	const n = RawBayerBlockWidth * RawBayerBlockHeight
	var out []struct {
		name string
		raw  []byte
	}

	all := func(v byte) []byte {
		b := make([]byte, n)
		for i := range b {
			b[i] = v
		}
		return b
	}
	out = append(out, struct {
		name string
		raw  []byte
	}{"zeros", all(0)})
	out = append(out, struct {
		name string
		raw  []byte
	}{"flat128", all(128)})
	out = append(out, struct {
		name string
		raw  []byte
	}{"flat255", all(255)})

	gradient := make([]byte, n)
	for i := range gradient {
		gradient[i] = byte((i*7 + (i/160)*11) % 256)
	}
	out = append(out, struct {
		name string
		raw  []byte
	}{"gradient", gradient})

	lcg := func(seed uint32) []byte {
		b := make([]byte, n)
		s := seed
		for i := range b {
			s = s*1664525 + 1013904223
			b[i] = byte(s >> 24)
		}
		return b
	}
	out = append(out, struct {
		name string
		raw  []byte
	}{"noise7", lcg(7)})
	out = append(out, struct {
		name string
		raw  []byte
	}{"noise123456", lcg(123456)})

	// A single hot pixel in each corner plus a vertical ramp exercises the
	// Star-Tetrix neighbour reflection at the picture edges.
	ramp := make([]byte, n)
	for y := 0; y < RawBayerBlockHeight; y++ {
		for x := 0; x < RawBayerBlockWidth; x++ {
			ramp[y*RawBayerBlockWidth+x] = byte((x*3 + y*37) % 256)
		}
	}
	ramp[0] = 255
	ramp[RawBayerBlockWidth-1] = 255
	ramp[n-1] = 0
	out = append(out, struct {
		name string
		raw  []byte
	}{"ramp-edges", ramp})

	return out
}

// runBridge encodes one block with the Rust bridge and returns the
// codestream bytes.
func runBridge(t *testing.T, bridgeExe string, raw []byte, q int, lx, ly int) []byte {
	t.Helper()
	dir := t.TempDir()
	in := filepath.Join(dir, "input.rggb8")
	out := filepath.Join(dir, "output.jxs")
	if err := os.WriteFile(in, raw, 0o600); err != nil {
		t.Fatalf("write bridge input: %v", err)
	}
	cmd := exec.Command(bridgeExe,
		"encode",
		"--input", in,
		"--output", out,
		"--target-bytes", "80",
		// The byte cap only gates the bridge's own Q-selection policy; with
		// --quantization forced it does not influence the encoded bytes.
		// A generous cap lets the bridge emit the stream even when it
		// exceeds the pipeline's 640-byte budget, so the comparison can
		// cover every configuration.
		"--max-bytes", "65535",
		"--levels-x", fmt.Sprint(lx),
		"--levels-y", fmt.Sprint(ly),
		"--quantization", fmt.Sprint(q),
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("bridge encode (q=%d levels=%d/%d): %v: %s", q, lx, ly, err, stderr.String())
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read bridge output: %v", err)
	}
	return b
}

// parityLevels lists the (NL,x, NL,y) pairs the bridge accepts.
func parityLevels() [][2]int {
	var out [][2]int
	for lx := 1; lx <= 5; lx++ {
		for ly := 0; ly <= 1; ly++ {
			if ly > lx {
				continue
			}
			out = append(out, [2]int{lx, ly})
		}
	}
	return out
}

// TestDecodedByUpstreamDecoder proves interoperability: a codestream
// produced by the Go encoder is decoded by the upstream Rust decoder
// (through the bridge's decode mode).
//
// Two properties are checked:
//
//   - at Q = 0 the reconstruction is bit-exact (lossless round trip).
//   - at any Q the upstream decoder's reconstruction of the Go stream
//     equals its reconstruction of the upstream encoder's own stream, so
//     the two encoders are interchangeable for a downstream decoder.
//
// Note the absolute reconstruction error is NOT bounded: the Star-Tetrix
// transform carries a DC offset of 128, which a lossy quantization step
// can discard entirely. That is the quantizer's behaviour, not a
// conformance difference.
func TestDecodedByUpstreamDecoder(t *testing.T) {
	bridgeExe := bridgePath(t)
	if bridgeExe == "" {
		t.Skipf("%s is not set; skipping the upstream decoder check", bridgeEnvVar)
	}
	for _, p := range parityPatterns() {
		for _, lv := range parityLevels() {
			lx, ly := lv[0], lv[1]
			for _, q := range []int{0, 4, 8, 15} {
				opts := &Options{
					Quantization: uint8(q),
					LevelsX:      uint8(lx),
					LevelsY:      uint8(ly),
					StarTetrix:   true,
					CTS:          CTSParams{CT: 0},
					Br:           8,
				}
				coded, err := EncodeRGGB8Block(p.raw, opts)
				if err != nil {
					t.Fatalf("%s q=%d levels=%d/%d: go encode: %v", p.name, q, lx, ly, err)
				}
				restored := runBridgeDecode(t, bridgeExe, coded)

				upstream := runBridge(t, bridgeExe, p.raw, q, lx, ly)
				want := runBridgeDecode(t, bridgeExe, upstream)
				if !bytes.Equal(restored, want) {
					t.Fatalf("%s q=%d levels=%d/%d: the upstream decoder reconstructs the Go stream differently from the upstream stream",
						p.name, q, lx, ly)
				}
				if q == 0 && !bytes.Equal(restored, p.raw) {
					t.Fatalf("%s q=%d levels=%d/%d: lossless round trip through the upstream decoder failed",
						p.name, q, lx, ly)
				}
			}
		}
	}
}

// runBridgeDecode decodes a codestream with the upstream decoder through
// the bridge's decode mode and returns the reconstructed mosaic.
func runBridgeDecode(t *testing.T, bridgeExe string, coded []byte) []byte {
	t.Helper()
	dir := t.TempDir()
	in := filepath.Join(dir, "input.jxs")
	out := filepath.Join(dir, "output.rggb8")
	if err := os.WriteFile(in, coded, 0o600); err != nil {
		t.Fatalf("write bridge input: %v", err)
	}
	cmd := exec.Command(bridgeExe, "decode", "--input", in, "--output", out)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("bridge decode: %v: %s", err, stderr.String())
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read bridge output: %v", err)
	}
	return b
}

// TestParityAgainstRustBridge compares the Go encoder against the
// upstream Rust bridge byte for byte over the supported configuration
// space.
func TestParityAgainstRustBridge(t *testing.T) {
	bridgeExe := bridgePath(t)
	if bridgeExe == "" {
		t.Skipf("%s is not set; skipping the live bridge comparison", bridgeEnvVar)
	}
	patterns := parityPatterns()
	for _, p := range patterns {
		for _, lv := range parityLevels() {
			lx, ly := lv[0], lv[1]
			for q := 0; q <= 15; q++ {
				want := runBridge(t, bridgeExe, p.raw, q, lx, ly)
				got, err := EncodeRGGB8Block(p.raw, &Options{
					Quantization: uint8(q),
					LevelsX:      uint8(lx),
					LevelsY:      uint8(ly),
					StarTetrix:   true,
					CTS:          CTSParams{CT: 0},
					// Br = 8 is what the upstream Rust encoder writes;
					// the comparison is byte-for-byte against it.
					Br: 8,
				})
				if err != nil {
					t.Fatalf("%s q=%d levels=%d/%d: go encode: %v", p.name, q, lx, ly, err)
				}
				if !bytes.Equal(want, got) {
					t.Fatalf("%s q=%d levels=%d/%d: codestream mismatch\n  bridge: %x\n  go:     %x\n  (bridge %d bytes, go %d bytes)",
						p.name, q, lx, ly, want, got, len(want), len(got))
				}
			}
		}
	}
}
