package jpegxs

import (
	"sync"
	"testing"
)

// benchBlock is a representative RGGB block for the encoder benchmark.
func benchBlock() []byte {
	const n = RawBayerBlockWidth * RawBayerBlockHeight
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((i*7 + (i/160)*11) % 256)
	}
	return b
}

// BenchmarkEncodeRGGB8Block measures the in-process encoder on one
// 160 × 4 RGGB8 block.
func BenchmarkEncodeRGGB8Block(b *testing.B) {
	raw := benchBlock()
	opts := &Options{
		Quantization: 8,
		LevelsX:      1,
		LevelsY:      0,
		StarTetrix:   true,
		CTS:          CTSParams{CT: 0},
	}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := EncodeRGGB8Block(raw, opts); err != nil {
			b.Fatal(err)
		}
	}
}

// benchFrame returns the 30 blocks of one 640 × 480 frame: the production
// unit of work, which must be encoded within one 30 FPS frame period.
func benchFrame() [][]byte {
	const blocks = 30
	out := make([][]byte, blocks)
	for i := range out {
		raw := make([]byte, RawBayerBlockWidth*RawBayerBlockHeight)
		for j := range raw {
			raw[j] = byte((i*31 + j*7 + j/160*11) % 256)
		}
		out[i] = raw
	}
	return out
}

// benchOpts returns the production encoder settings: Q = 6, levels 1/0.
func benchOpts() *Options {
	return &Options{
		Quantization: 6,
		LevelsX:      1,
		LevelsY:      0,
		StarTetrix:   true,
		CTS:          CTSParams{CT: 0},
	}
}

// BenchmarkEncodeFrameSerial measures one whole frame's 30 blocks encoded
// one after another. This is the number that has to fit in a frame period.
func BenchmarkEncodeFrameSerial(b *testing.B) {
	frame := benchFrame()
	opts := benchOpts()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, raw := range frame {
			if _, err := EncodeRGGB8Block(raw, opts); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkEncodeFrameParallel measures one frame's 30 blocks encoded across
// several goroutines. The blocks share no state, so this shows how well the
// encoder scales with cores.
func BenchmarkEncodeFrameParallel(b *testing.B) {
	frame := benchFrame()
	opts := benchOpts()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		for w := 0; w < 4; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				for j := w; j < len(frame); j += 4 {
					if _, err := EncodeRGGB8Block(frame[j], opts); err != nil {
						b.Error(err)
						return
					}
				}
			}(w)
		}
		wg.Wait()
	}
}

// BenchmarkEncodeRGGB8BlockCascade measures the multi-level path.
func BenchmarkEncodeRGGB8BlockCascade(b *testing.B) {
	raw := benchBlock()
	opts := &Options{
		Quantization: 8,
		LevelsX:      3,
		LevelsY:      1,
		StarTetrix:   true,
		CTS:          CTSParams{CT: 0},
	}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := EncodeRGGB8Block(raw, opts); err != nil {
			b.Fatal(err)
		}
	}
}
