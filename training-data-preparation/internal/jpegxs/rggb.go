package jpegxs

import (
	"fmt"
)

// RawBayerBlockWidth and RawBayerBlockHeight are the dimensions of one
// training-data block: a 160 × 4 RAW Bayer slice of a 640 × 480 frame.
const (
	RawBayerBlockWidth  = 160
	RawBayerBlockHeight = 4
)

// RGGB8Components de-interleaves a raw RGGB8 Bayer mosaic into the four
// Star-Tetrix component planes.
//
// The CFA sample grid is de-interleaved into R, G1, G2, B, each
// (width/2) × (height/2). The codestream picture grid is therefore
// (width/2) × (height/2) with four registered components — it is *not*
// four (width × height) planes, and no RGB de-mosaicking happens.
//
// This mirrors the layout the upstream Rust bridge uses, so the bytes
// this produces are interchangeable with the bridge's output.
func RGGB8Components(raw []byte, width, height int) (*Components, error) {
	if width%2 != 0 || height%2 != 0 {
		return nil, fmt.Errorf("jpegxs RGGB8: mosaic dimensions must be even, got %dx%d", width, height)
	}
	if len(raw) != width*height {
		return nil, fmt.Errorf("jpegxs RGGB8: expected %d bytes of %dx%d RGGB8, got %d", width*height, width, height, len(raw))
	}
	planeW := width / 2
	planeH := height / 2
	planes := make([][]byte, 4)
	for i := range planes {
		planes[i] = make([]byte, planeW*planeH)
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := (y%2)*2 + x%2
			planes[c][(y/2)*planeW+x/2] = raw[y*width+x]
		}
	}
	return &Components{
		Width:     uint32(planeW),
		Height:    uint32(planeH),
		Cpih:      3,
		BitDepths: []uint8{8, 8, 8, 8},
		Sampling:  [][2]uint8{{1, 1}, {1, 1}, {1, 1}, {1, 1}},
		Planes:    planes,
	}, nil
}

// EncodeRGGB8Block encodes one raw RGGB8 Bayer block as a bare JPEG XS
// codestream using the Star-Tetrix transform.
func EncodeRGGB8Block(raw []byte, opts *Options) ([]byte, error) {
	comps, err := RGGB8Components(raw, RawBayerBlockWidth, RawBayerBlockHeight)
	if err != nil {
		return nil, err
	}
	return EncodeComponents(comps, opts)
}
