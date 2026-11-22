package dataset

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/jpegxs"
)

// Encoder implementations must produce a complete JPEG XS codestream for one RGGB block.
type Encoder interface {
	// Encode turns one RGGB block into a bare JPEG XS codestream.
	Encode(context.Context, []byte) ([]byte, error)
	// EncoderConfig returns the encoder configuration recorded in the dataset
	// metadata, so a resumed dataset can prove it used the same encoder.
	EncoderConfig() EncoderConfig
}

// EncoderConfig describes an encoder. The same struct covers both the
// external command encoder and the in-process Go encoder; the "kind"
// field selects which.
//
// The JSON is strict — unknown fields are rejected — so a typo cannot
// silently change the encoder's behaviour.
type EncoderConfig struct {
	// Kind selects the encoder implementation: "command" (default) or
	// "native".
	Kind string `json:"kind,omitempty"`
	// Identity is a human-readable description of the exact encoder
	// build and settings. It is recorded in the dataset metadata and
	// compared on resume.
	Identity string `json:"identity"`
	// TimeoutSeconds bounds one external encoder invocation.
	TimeoutSeconds int `json:"timeout_seconds,omitempty"`

	// Executable and Args configure the external command encoder.
	Executable string   `json:"executable,omitempty"`
	Args       []string `json:"args,omitempty"`

	// Quantization, LevelsX, LevelsY and StarTetrix configure the
	// in-process Go encoder.
	Quantization *uint8     `json:"quantization,omitempty"`
	LevelsX      *uint8     `json:"levels_x,omitempty"`
	LevelsY      *uint8     `json:"levels_y,omitempty"`
	StarTetrix   *bool      `json:"star_tetrix,omitempty"`
	CTS          *CTSConfig `json:"cts,omitempty"`
	// Br is the raw bitplane-count width. 4 is the only conformant value
	// (ISO/IEC 21122-1:2022 Table A.7) and the default; 8 reproduces the
	// upstream Rust encoder's bytes, which a conforming third-party
	// decoder rejects.
	Br *uint8 `json:"br,omitempty"`
	// MaxBytes caps the encoded size; the stream is rejected when it
	// exceeds the cap, mirroring the external encoder's byte budget.
	MaxBytes int `json:"max_bytes,omitempty"`
}

// CTSConfig holds the CTS marker parameters of the Star-Tetrix
// transform.
type CTSConfig struct {
	E1 uint8 `json:"e1"`
	E2 uint8 `json:"e2"`
	CF uint8 `json:"cf"`
	CT uint8 `json:"ct"`
}

// CommandEncoder runs an external encoder executable once per block.
type CommandEncoder struct{ Config EncoderConfig }

func (e *CommandEncoder) EncoderConfig() EncoderConfig { return e.Config }

// NativeEncoder encodes blocks in-process with the Go JPEG XS encoder.
//
// It produces byte-identical codestreams to the oxideav-jpegxs Rust
// bridge for the same configuration, but needs no external process, no
// temporary files and no Rust toolchain.
type NativeEncoder struct {
	Config EncoderConfig
	Opts   jpegxs.Options
}

// Config returns the encoder configuration recorded in the dataset
// metadata.
func (e *NativeEncoder) EncoderConfig() EncoderConfig { return e.Config }

// Encode encodes one RGGB block.
func (e *NativeEncoder) Encode(ctx context.Context, raw []byte) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(raw) != InputWidth*InputHeight {
		return nil, fmt.Errorf("encoder input must be 640 bytes")
	}
	b, err := jpegxs.EncodeRGGB8Block(raw, &e.Opts)
	if err != nil {
		return nil, err
	}
	max := e.Config.MaxBytes
	if max <= 0 {
		max = MaxEncodedBytes
	}
	if len(b) < 4 || len(b) > max {
		return nil, fmt.Errorf("encoded length %d outside 4..%d bytes; refusing truncation", len(b), max)
	}
	// SOC FF10 and EOC FF11 reject obvious container/raw/fake outputs. This is not full validation.
	if b[0] != 0xff || b[1] != 0x10 || b[len(b)-2] != 0xff || b[len(b)-1] != 0x11 {
		return nil, fmt.Errorf("encoder output must be a bare complete JPEG XS codestream (SOC/EOC markers)")
	}
	return b, nil
}

// LoadEncoder loads an encoder configuration and returns the matching
// implementation.
func LoadEncoder(path string) (Encoder, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var probe struct {
		Kind string `json:"kind"`
	}
	if err = json.Unmarshal(b, &probe); err != nil {
		return nil, fmt.Errorf("encoder config %s: %w", path, err)
	}
	switch probe.Kind {
	case "", "command":
		return loadCommandEncoder(b)
	case "native":
		return loadNativeEncoder(b)
	default:
		return nil, fmt.Errorf("encoder config %s: unknown kind %q (want \"command\" or \"native\")", path, probe.Kind)
	}
}

// decodeEncoderConfig parses an encoder configuration strictly.
func decodeEncoderConfig(b []byte) (EncoderConfig, error) {
	var c EncoderConfig
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return c, err
	}
	return c, nil
}

// loadCommandEncoder builds the external-command encoder.
func loadCommandEncoder(b []byte) (*CommandEncoder, error) {
	c, err := decodeEncoderConfig(b)
	if err != nil {
		return nil, err
	}
	if c.Executable == "" || c.Identity == "" || c.TimeoutSeconds < 1 {
		return nil, fmt.Errorf("encoder requires executable, identity/version and positive timeout_seconds")
	}
	joined := strings.Join(c.Args, " ")
	if !strings.Contains(joined, "{input}") || !strings.Contains(joined, "{output}") {
		return nil, fmt.Errorf("encoder args must include {input} and {output}")
	}
	resolved, err := exec.LookPath(c.Executable)
	if err != nil {
		return nil, fmt.Errorf("JPEG XS encoder: %w", err)
	}
	c.Executable, err = filepath.Abs(resolved)
	if err != nil {
		return nil, err
	}
	return &CommandEncoder{Config: c}, nil
}

// loadNativeEncoder builds the in-process Go encoder.
func loadNativeEncoder(b []byte) (*NativeEncoder, error) {
	c, err := decodeEncoderConfig(b)
	if err != nil {
		return nil, err
	}
	if c.Identity == "" {
		return nil, fmt.Errorf("encoder requires an identity/version string")
	}
	if c.Executable != "" || len(c.Args) != 0 || c.TimeoutSeconds != 0 {
		return nil, fmt.Errorf("native encoder does not take executable, args or timeout_seconds")
	}
	quant := uint8(4)
	if c.Quantization != nil {
		quant = *c.Quantization
	}
	lx := uint8(1)
	if c.LevelsX != nil {
		lx = *c.LevelsX
	}
	ly := uint8(0)
	if c.LevelsY != nil {
		ly = *c.LevelsY
	}
	starTetrix := true
	if c.StarTetrix != nil {
		starTetrix = *c.StarTetrix
	}
	cts := CTSConfig{}
	if c.CTS != nil {
		cts = *c.CTS
	}
	br := uint8(0) // 0 selects the encoder's conformant default (4).
	if c.Br != nil {
		br = *c.Br
	}
	return &NativeEncoder{
		Config: c,
		Opts: jpegxs.Options{
			Quantization: quant,
			LevelsX:      lx,
			LevelsY:      ly,
			StarTetrix:   starTetrix,
			Br:           br,
			CTS: jpegxs.CTSParams{
				E1: cts.E1,
				E2: cts.E2,
				CF: cts.CF,
				CT: cts.CT,
			},
		},
	}, nil
}

// Encode runs the external encoder once.
func (e *CommandEncoder) Encode(ctx context.Context, raw []byte) ([]byte, error) {
	if len(raw) != InputWidth*InputHeight {
		return nil, fmt.Errorf("encoder input must be 640 bytes")
	}
	dir, err := os.MkdirTemp("", "brainhole-jpegxs-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	input, output := filepath.Join(dir, "input.rggb8"), filepath.Join(dir, "output.jxs")
	if err := os.WriteFile(input, raw, 0600); err != nil {
		return nil, err
	}
	replace := strings.NewReplacer("{input}", input, "{output}", output, "{width}", "160", "{height}", "4", "{bitdepth}", "8", "{ratio}", "8", "{max_bytes}", "640")
	args := make([]string, len(e.Config.Args))
	for i, arg := range e.Config.Args {
		args[i] = replace.Replace(arg)
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(e.Config.TimeoutSeconds)*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, e.Config.Executable, args...)
	var log limitedLog
	cmd.Stdout = &log
	cmd.Stderr = &log
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("JPEG XS encoder: %w: %s", err, log.String())
	}
	f, err := os.Open(output)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < 4 || info.Size() > MaxEncodedBytes {
		return nil, fmt.Errorf("encoded length %d outside 4..640 bytes; refusing truncation", info.Size())
	}
	b, err := os.ReadFile(output)
	if err != nil {
		return nil, err
	}
	// SOC FF10 and EOC FF11 reject obvious container/raw/fake outputs. This is not full validation.
	if b[0] != 0xff || b[1] != 0x10 || b[len(b)-2] != 0xff || b[len(b)-1] != 0x11 {
		return nil, fmt.Errorf("encoder output must be a bare complete JPEG XS codestream (SOC/EOC markers)")
	}
	return b, nil
}

// Capture bounded diagnostics while continuing to drain subprocess output.
type limitedLog struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *limitedLog) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(p)
	remaining := 16384 - b.buf.Len()
	if remaining > 0 {
		if len(p) > remaining {
			p = p[:remaining]
		}
		_, _ = b.buf.Write(p)
	}
	return n, nil
}
func (b *limitedLog) String() string { b.mu.Lock(); defer b.mu.Unlock(); return b.buf.String() }
