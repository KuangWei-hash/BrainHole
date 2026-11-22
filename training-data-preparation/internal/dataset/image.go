package dataset

import "fmt"

const (
	Width             = 640
	Height            = 480
	BlockHeight       = 16
	BlockCount        = Height / BlockHeight
	InputWidth        = 160
	InputHeight       = 4
	FPS               = 30
	MaxEncodedBytes   = 640
	RGBBlockBytes     = Width * BlockHeight * 3
	RawBlockBytes     = Width * BlockHeight
	SchemaVersion     = 4
	TargetDescription = "RAW8 Bayer RGGB row-major [16,640], 10240 bytes; shared-header-v1 payload padded to 640; schema v4 also stores the 640-byte low-resolution decoded mosaic"
	RGBFrameBytes     = Width * Height * 3
	// DecodedBytes is the size of the low-resolution decoded mosaic stored
	// alongside every encoded sample: the 160x4 RGGB block the JPEG XS
	// codestream reconstructs, which becomes the second model input.
	DecodedBytes = InputWidth * InputHeight
)

// Phase uses the output frame index, including frames duplicated to reach 30 FPS.
func Phase(frame int64) int { return int(frame%16) + 1 }

// Bayer selects channels directly from RGB8. No linearization or sensor ISP is simulated.
func Bayer(rgb []byte) ([]byte, error) {
	if len(rgb) != RGBBlockBytes {
		return nil, fmt.Errorf("RGB block: got %d bytes, want %d", len(rgb), RGBBlockBytes)
	}
	out := make([]byte, Width*BlockHeight)
	for y := 0; y < BlockHeight; y++ {
		for x := 0; x < Width; x++ {
			channel := (y % 2) + (x % 2) // RG / GB => channel 0,1 / 1,2
			out[y*Width+x] = rgb[(y*Width+x)*3+channel]
		}
	}
	return out, nil
}

// Downsample retains one complete RGGB tile per 8x8 pixels, preserving RGGB.
func Downsample(raw []byte, phase int) ([]byte, error) {
	if len(raw) != Width*BlockHeight || phase < 1 || phase > 16 {
		return nil, fmt.Errorf("invalid Bayer block size or phase %d", phase)
	}
	out := make([]byte, InputWidth*InputHeight)
	px, py := ((phase-1)%4)*2, ((phase-1)/4)*2
	for j := 0; j < 2; j++ {
		for i := 0; i < 80; i++ {
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					out[(2*j+dy)*InputWidth+2*i+dx] = raw[(8*j+py+dy)*Width+8*i+px+dx]
				}
			}
		}
	}
	return out, nil
}
