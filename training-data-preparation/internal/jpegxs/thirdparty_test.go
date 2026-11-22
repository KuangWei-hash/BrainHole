package jpegxs

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// The tests here check the encoder's codestreams against independent
// decoders — libsvtjpegxs (SVT-AV1's JPEG XS decoder) through FFmpeg, and
// the upstream Rust crate — rather than only against the Rust code the
// port was written from.
//
// That distinction matters. The upstream encoder writes Br = 8, which
// ISO/IEC 21122-1:2022 Table A.7 does not allow (it lists a single
// conformant value, 4, next to Ng = 4 and Ss = 8). libsvtjpegxs rejects
// such streams outright, so "my encoder matches the Rust encoder" would
// not have established correctness on its own.

// refDecoderPath returns the upstream Rust reference decoder's path, or
// "" when it is unavailable.
//
// The reference decoder is a small test-only binary built from the same
// upstream crate; it decodes arbitrary codestreams, unlike the
// repository's bridge, which only accepts its own 160x4 RGGB
// Star-Tetrix layout.
func refDecoderPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("JPEgXS_REF_DECODER"); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("JPEgXS_REF_DECODER=%q is not usable: %v", p, err)
		}
		return p
	}
	p := filepath.Join("..", "..", "..", ".cache", "jpegxs-ref", "target", "release", "jpegxs-ref.exe")
	if _, err := os.Stat(p); err != nil {
		return ""
	}
	return p
}

// thirdPartyDecoderAvailable reports whether ffmpeg carries a JPEG XS
// decoder.
func thirdPartyDecoderAvailable(t *testing.T) (string, bool) {
	t.Helper()
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		return "", false
	}
	out, err := exec.Command(ffmpeg, "-hide_banner", "-decoders").CombinedOutput()
	if err != nil && len(out) == 0 {
		return "", false
	}
	for _, line := range bytes.Split(out, []byte("\n")) {
		fields := bytes.Fields(line)
		if len(fields) >= 2 && bytes.Contains(fields[1], []byte("jpegxs")) {
			return ffmpeg, true
		}
	}
	return "", false
}

// thirdPartyDecode decodes a codestream with libsvtjpegxs and returns the
// raw planar samples.
func thirdPartyDecode(t *testing.T, ffmpeg string, coded []byte, pixFmt string) ([]byte, error) {
	t.Helper()
	dir := t.TempDir()
	in := filepath.Join(dir, "in.jxs")
	out := filepath.Join(dir, "out.raw")
	if err := os.WriteFile(in, coded, 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
		"-i", in, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pixFmt, out)
	if msg, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("libsvtjpegxs declined the stream: %v: %s", err, bytes.TrimSpace(msg))
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	return b, nil
}

// refDecode decodes a codestream with the upstream Rust decoder.
func refDecode(t *testing.T, refDecoder string, coded []byte) []byte {
	t.Helper()
	dir := t.TempDir()
	in := filepath.Join(dir, "in.jxs")
	out := filepath.Join(dir, "out.raw")
	if err := os.WriteFile(in, coded, 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(refDecoder, in, out)
	if msg, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("upstream decoder: %v: %s", err, bytes.TrimSpace(msg))
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestConformantStreamsDecodeWithThirdPartyDecoder checks that the
// encoder's default (conformant) codestreams are accepted by an
// independent decoder, and that the third-party decoder's reconstruction
// matches the upstream Rust decoder's.
func TestConformantStreamsDecodeWithThirdPartyDecoder(t *testing.T) {
	ffmpeg, ok := thirdPartyDecoderAvailable(t)
	if !ok {
		t.Skip("no third-party JPEG XS decoder on this machine")
	}
	refDecoder := refDecoderPath(t)
	if refDecoder == "" {
		t.Skip("no upstream reference decoder available")
	}

	// A Cpih = 0 picture: third-party decoders are far more likely to
	// support a plain colourless stream than Star-Tetrix.
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

	for _, q := range []int{0, 4, 8} {
		coded, err := EncodeComponents(comps, &Options{
			Quantization: uint8(q),
			LevelsX:      1,
			LevelsY:      0,
		})
		if err != nil {
			t.Fatalf("q=%d: encode: %v", q, err)
		}
		theirs, err := thirdPartyDecode(t, ffmpeg, coded, "yuv444p")
		if err != nil {
			t.Fatalf("q=%d: %v", q, err)
		}
		if len(theirs) != 3*w*h {
			t.Fatalf("q=%d: third-party decoder returned %d bytes, want %d", q, len(theirs), 3*w*h)
		}
		ours := refDecode(t, refDecoder, coded)
		if len(ours) != 3*w*h {
			t.Fatalf("q=%d: upstream decoder returned %d bytes, want %d", q, len(ours), 3*w*h)
		}
		if !bytes.Equal(theirs, ours) {
			for i := range theirs {
				if theirs[i] != ours[i] {
					t.Fatalf("q=%d: sample %d: third-party decoder says %d, upstream says %d",
						q, i, theirs[i], ours[i])
				}
			}
		}
		// At Q = 0 the reconstruction must be the source samples.
		if q == 0 {
			for c := 0; c < 3; c++ {
				for i := 0; i < w*h; i++ {
					if theirs[c*w*h+i] != plane[i] {
						t.Fatalf("q=0: component %d sample %d: third-party decoder says %d, source is %d",
							c, i, theirs[c*w*h+i], plane[i])
					}
				}
			}
		}
	}
}

// TestNonConformantBrIsRejectedByThirdPartyDecoder documents why the
// default Br is 4 and not the upstream's 8.
func TestNonConformantBrIsRejectedByThirdPartyDecoder(t *testing.T) {
	ffmpeg, ok := thirdPartyDecoderAvailable(t)
	if !ok {
		t.Skip("no third-party JPEG XS decoder on this machine")
	}
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
	coded, err := EncodeComponents(comps, &Options{
		Quantization: 4,
		LevelsX:      1,
		LevelsY:      0,
		Br:           8,
	})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, err = thirdPartyDecode(t, ffmpeg, coded, "yuv444p"); err == nil {
		t.Fatal("expected the third-party decoder to reject a Br = 8 codestream")
	}
}
