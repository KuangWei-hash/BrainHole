package jpegxs

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestBrConformanceAgainstThirdPartyDecoder checks the raw bitplane-count
// width Br against an independent decoder.
//
// ISO/IEC 21122-1:2022 Table A.7 lists exactly one conformant value for
// Br — 4 — next to Ng = 4 and Ss = 8. The upstream Rust encoder writes
// Br = 8; libsvtjpegxs (SVT-AV1's JPEG XS decoder) rejects that.
func TestBrConformanceAgainstThirdPartyDecoder(t *testing.T) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg unavailable")
	}
	// Only a Cpih = 0 picture: third-party decoders are far more likely
	// to support a plain colourless stream than Star-Tetrix.
	const w, h = 64, 32
	plane := make([]byte, w*h)
	for i := range plane {
		plane[i] = byte((i*37 + i/8) % 256)
	}
	comps := &Components{
		Width:     w,
		Height:    h,
		Cpih:      0,
		BitDepths: []uint8{8, 8, 8},
		Sampling:  [][2]uint8{{1, 1}, {1, 1}, {1, 1}},
		Planes:    [][]byte{plane, plane, plane},
	}
	dir := t.TempDir()
	for _, br := range []uint8{4, 8} {
		coded, err := EncodeComponents(comps, &Options{
			Quantization: 4,
			LevelsX:      1,
			LevelsY:      0,
			Br:           br,
		})
		if err != nil {
			t.Fatalf("Br=%d: encode: %v", br, err)
		}
		path := filepath.Join(dir, "br"+string(rune('0'+br))+".jxs")
		if err = os.WriteFile(path, coded, 0o600); err != nil {
			t.Fatal(err)
		}
		out := filepath.Join(dir, "br"+string(rune('0'+br))+".raw")
		cmd := exec.Command(ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
			"-i", path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv444p", out)
		msg, err := cmd.CombinedOutput()
		t.Logf("Br=%d: ffmpeg/libsvtjpegxs: err=%v msg=%s", br, err, string(msg))
	}
}
