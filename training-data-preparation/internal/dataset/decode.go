package dataset

import (
	"fmt"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/jpegxs"
)

// DecodeBlock reconstructs the low-resolution mosaic of one encoded block.
//
// The decoding always runs on the in-process Go JPEG XS decoder — the same
// implementation the encoder was ported from — so no external decoder
// process, temporary file or Rust toolchain is involved. The Go decoder
// reproduces the upstream Rust decoder byte for byte, so the stored decoded
// mosaic is identical to what the upstream decoder produces.
//
// The result is the lossy reconstruction: at Q = 6 it differs from the
// encoder input, which is exactly what the second model input needs.
func DecodeBlock(codestream []byte) ([]byte, error) {
	if len(codestream) < 4 || len(codestream) > MaxEncodedBytes {
		return nil, fmt.Errorf("codestream length %d outside 4..%d bytes", len(codestream), MaxEncodedBytes)
	}
	decoded, err := jpegxs.DecodeRGGB8Block(codestream)
	if err != nil {
		return nil, err
	}
	if len(decoded) != DecodedBytes {
		return nil, fmt.Errorf("decoded mosaic is %d bytes, want %d", len(decoded), DecodedBytes)
	}
	return decoded, nil
}
