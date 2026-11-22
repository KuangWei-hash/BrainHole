package jpegxs

import (
	"bytes"
	"testing"
)

// TestConformantStreamsRoundTripThroughUpstreamDecoder runs the whole
// supported configuration space through the upstream Rust decoder.
//
// The encoder writes Br = 4 — the only conformant bitplane-count width
// (ISO/IEC 21122-1:2022 Table A.7) — and the upstream decoder reads it
// back. At Q = 0 the reconstruction must reproduce the source mosaic
// exactly, which exercises the whole chain: Star-Tetrix, the forward
// 5/3 DWT, the quantizer, the entropy coder and the inverse of each.
func TestConformantStreamsRoundTripThroughUpstreamDecoder(t *testing.T) {
	refDecoder := refDecoderPath(t)
	if refDecoder == "" {
		t.Skip("no upstream reference decoder available")
	}
	for _, p := range parityPatterns() {
		for _, lv := range parityLevels() {
			lx, ly := lv[0], lv[1]
			for q := 0; q <= 15; q++ {
				coded, err := EncodeRGGB8Block(p.raw, &Options{
					Quantization: uint8(q),
					LevelsX:      uint8(lx),
					LevelsY:      uint8(ly),
					StarTetrix:   true,
					CTS:          CTSParams{CT: 0},
				})
				if err != nil {
					t.Fatalf("%s q=%d levels=%d/%d: encode: %v", p.name, q, lx, ly, err)
				}
				decoded := refDecode(t, refDecoder, coded)
				if len(decoded) != len(p.raw) {
					t.Fatalf("%s q=%d levels=%d/%d: decoder returned %d bytes, want %d",
						p.name, q, lx, ly, len(decoded), len(p.raw))
				}
				if q != 0 {
					continue
				}
				mosaic := planesToMosaic(decoded, RawBayerBlockWidth, RawBayerBlockHeight)
				if !bytes.Equal(mosaic, p.raw) {
					for i := range mosaic {
						if mosaic[i] != p.raw[i] {
							t.Fatalf("%s q=0 levels=%d/%d: sample %d reconstructed as %d, want %d (lossless round trip)",
								p.name, lx, ly, i, mosaic[i], p.raw[i])
						}
					}
				}
			}
		}
	}
}

// planesToMosaic reassembles the RGGB mosaic from the four
// de-interleaved component planes, reversing the encoder's Annex F
// component registration.
func planesToMosaic(planes []byte, width, height int) []byte {
	out := make([]byte, width*height)
	pw, ph := width/2, height/2
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := (y%2)*2 + x%2
			out[y*width+x] = planes[c*pw*ph+(y/2)*pw+x/2]
		}
	}
	return out
}
