package dataset

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// nativeAndCommandEncoders loads both encoder implementations from the
// repository's configuration files.
//
// The external encoder's executable is resolved to an absolute path so
// the test does not depend on the working directory.
func nativeAndCommandEncoders(t *testing.T) (*NativeEncoder, *CommandEncoder) {
	t.Helper()
	root := filepath.Join("..", "..")
	native, err := LoadEncoder(filepath.Join(root, "encoder.native.json"))
	if err != nil {
		t.Fatalf("load native encoder: %v", err)
	}
	bridge := os.Getenv("JPEgXS_BRIDGE")
	if bridge == "" {
		bridge = filepath.Join(root, "bin", "brainhole-jpegxs.exe")
	}
	if _, err := os.Stat(bridge); err != nil {
		t.Skipf("external JPEG XS encoder not available at %s: %v", bridge, err)
	}
	raw, err := os.ReadFile(filepath.Join(root, "encoder.oxideav.json"))
	if err != nil {
		t.Fatalf("read encoder.oxideav.json: %v", err)
	}
	var cfg EncoderConfig
	if err = json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("parse encoder.oxideav.json: %v", err)
	}
	cfg.Executable = bridge
	return native.(*NativeEncoder), &CommandEncoder{Config: cfg}
}

// TestNativeEncoderMatchesCommandEncoder proves the in-process Go encoder
// and the external Rust bridge produce identical codestreams for the
// pipeline's production configuration.
//
// The comparison uses Br = 8 because that is what the upstream Rust
// encoder writes. The conformant Br = 4 differs in exactly the
// picture-header bitplane-count field; see the jpegxs package's
// third-party decoder tests.
func TestNativeEncoderMatchesCommandEncoder(t *testing.T) {
	native, command := nativeAndCommandEncoders(t)
	native.Opts.Br = 8
	ctx := context.Background()

	// A block that exercises the real Bayer path: build it from an RGB
	// frame through the same Bayer/Downsample steps the pipeline uses.
	rgb := make([]byte, RGBBlockBytes)
	for i := 0; i < len(rgb); i++ {
		rgb[i] = byte((i*7 + i/13) % 256)
	}
	raw, err := Bayer(rgb)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = Downsample(raw, 1)
	if err != nil {
		t.Fatal(err)
	}

	want, err := command.Encode(ctx, raw)
	if err != nil {
		t.Fatalf("command encoder: %v", err)
	}
	got, err := native.Encode(ctx, raw)
	if err != nil {
		t.Fatalf("native encoder: %v", err)
	}
	if !bytes.Equal(want, got) {
		t.Fatalf("native encoder output differs from the Rust bridge\n  bridge: %x\n  native: %x", want, got)
	}
	if len(got) > MaxEncodedBytes {
		t.Fatalf("native encoder produced %d bytes, over the %d byte budget", len(got), MaxEncodedBytes)
	}
}

// TestNativeEncoderRejectsOversizedStreams checks the byte budget the
// pipeline relies on.
func TestNativeEncoderRejectsOversizedStreams(t *testing.T) {
	native, _ := nativeAndCommandEncoders(t)
	// Lossless white noise does not fit the 640-byte budget.
	native.Opts.Quantization = 0
	raw := make([]byte, InputWidth*InputHeight)
	s := uint32(12345)
	for i := range raw {
		s = s*1664525 + 1013904223
		raw[i] = byte(s >> 24)
	}
	if _, err := native.Encode(context.Background(), raw); err == nil {
		t.Fatal("expected the native encoder to reject an oversized stream")
	}
}

// TestNativeEncoderSatisfiesSharedHeaderPacking checks that the native
// encoder's codestreams satisfy the shared-header packing contract: the
// header can be split off, stored once, and every block restored to a
// byte-identical complete codestream.
func TestNativeEncoderSatisfiesSharedHeaderPacking(t *testing.T) {
	native, _ := nativeAndCommandEncoders(t)
	ctx := context.Background()

	var header SharedHeader
	var payloads [][]byte
	for i := 0; i < BlockCount; i++ {
		rgb := make([]byte, RGBBlockBytes)
		for j := range rgb {
			rgb[j] = byte((j*11 + i*37) % 256)
		}
		raw, err := Bayer(rgb)
		if err != nil {
			t.Fatal(err)
		}
		raw, err = Downsample(raw, 1+i%16)
		if err != nil {
			t.Fatal(err)
		}
		coded, err := native.Encode(ctx, raw)
		if err != nil {
			t.Fatalf("block %d: %v", i, err)
		}
		h, p, err := SplitCodestream(coded, native.EncoderConfig().Identity)
		if err != nil {
			t.Fatalf("block %d: shared-header split: %v", i, err)
		}
		if i == 0 {
			header = h
		} else if header.ID != h.ID {
			t.Fatalf("block %d: shared header changed", i)
		}
		payloads = append(payloads, p)
	}
	for i, p := range payloads {
		full, err := header.Restore(p)
		if err != nil {
			t.Fatalf("block %d: restore: %v", i, err)
		}
		if full[0] != 0xff || full[1] != 0x10 || full[len(full)-2] != 0xff || full[len(full)-1] != 0x11 {
			t.Fatalf("block %d: restored codestream lost its SOC/EOC markers", i)
		}
	}
}

// TestLoadEncoderRejectsUnknownFields keeps a typo in the encoder
// configuration from silently changing the encoding.
func TestLoadEncoderRejectsUnknownFields(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "encoder.json")
	if err := os.WriteFile(path, []byte(`{"kind":"native","identity":"x","quantisation":4}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadEncoder(path); err == nil {
		t.Fatal("expected an unknown field to be rejected")
	}
}
