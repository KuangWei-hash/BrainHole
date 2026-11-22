package dataset

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// benchBlock builds a representative RGGB block: a Bayer mosaic of a
// smooth gradient plus sensor-like noise.
func benchBlock() []byte {
	rgb := make([]byte, RGBBlockBytes)
	for i := range rgb {
		rgb[i] = byte((i*7 + i/13) % 256)
	}
	raw, err := Bayer(rgb)
	if err != nil {
		panic(err)
	}
	raw, err = Downsample(raw, 1)
	if err != nil {
		panic(err)
	}
	return raw
}

// BenchmarkNativeEncode measures the in-process Go encoder.
func BenchmarkNativeEncode(b *testing.B) {
	enc, err := LoadEncoder(filepath.Join("..", "..", "encoder.native.json"))
	if err != nil {
		b.Fatal(err)
	}
	raw := benchBlock()
	ctx := context.Background()
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := enc.Encode(ctx, raw); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCommandEncode measures the external Rust bridge path, which
// spawns a process and writes two temporary files per block.
func BenchmarkCommandEncode(b *testing.B) {
	root := filepath.Join("..", "..")
	bridge := os.Getenv("JPEgXS_BRIDGE")
	if bridge == "" {
		bridge = filepath.Join(root, "bin", "brainhole-jpegxs.exe")
	}
	if _, err := os.Stat(bridge); err != nil {
		b.Skipf("external JPEG XS encoder not available at %s: %v", bridge, err)
	}
	rawCfg, err := os.ReadFile(filepath.Join(root, "encoder.oxideav.json"))
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := decodeEncoderConfig(rawCfg)
	if err != nil {
		b.Fatal(err)
	}
	cfg.Executable = bridge
	enc := &CommandEncoder{Config: cfg}

	block := benchBlock()
	ctx := context.Background()
	b.SetBytes(int64(len(block)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := enc.Encode(ctx, block); err != nil {
			b.Fatal(err)
		}
	}
}
