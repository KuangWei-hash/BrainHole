package jpegxs

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The golden vectors pin the encoder's output so byte-exactness with the
// upstream Rust encoder can be verified on a machine without a Rust
// toolchain.
//
// The file is regenerated from the upstream bridge with
// JPEgXS_UPDATE_GOLDEN=1; the live comparison in parity_test.go is the
// authority for the vectors' contents.

const goldenPath = "testdata/golden.json"

// goldenEncoderID identifies the upstream encoder build the vectors were
// produced from.
const goldenEncoderID = "oxideav-jpegxs 0.0.9 @3484a041d6d8e893683c1920cb3f68fe6fd3b8b8"

// goldenEntry holds the expected output of one pattern across all
// quantization steps Q = 0..15.
type goldenEntry struct {
	Pattern string `json:"pattern"`
	LevelsX int    `json:"levels_x"`
	LevelsY int    `json:"levels_y"`
	// Br is the raw bitplane-count width the vectors were produced with:
	// 4 (the only conformant value, Table A.7) or 8 (the upstream
	// encoder's choice, kept for byte-parity checks).
	Br uint8 `json:"br"`
	// SHA256 holds one hex digest per Q, indexed by Q.
	SHA256 []string `json:"sha256"`
	// Lengths holds the encoded size per Q, indexed by Q.
	Lengths []int `json:"lengths"`
}

// goldenFile is the on-disk format of testdata/golden.json.
type goldenFile struct {
	Encoder string        `json:"encoder"`
	Vectors []goldenEntry `json:"vectors"`
}

// loadGolden reads the pinned vectors.
func loadGolden(t *testing.T) goldenFile {
	t.Helper()
	b, err := os.ReadFile(filepath.FromSlash(goldenPath))
	if err != nil {
		t.Fatalf("read %s: %v", goldenPath, err)
	}
	var g goldenFile
	if err = json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse %s: %v", goldenPath, err)
	}
	if g.Encoder != goldenEncoderID {
		t.Fatalf("golden vectors are for %q, want %q", g.Encoder, goldenEncoderID)
	}
	return g
}

// goldenBrs lists the bitplane-count widths the golden vectors cover.
func goldenBrs() []uint8 { return []uint8{4, 8} }

// TestGoldenVectors checks the encoder against the pinned upstream
// outputs.
//
// The Br = 8 vectors are the upstream Rust encoder's bytes; the Br = 4
// vectors are the same encodings with the only conformant bitplane-count
// width (Table A.7) — they differ from the upstream bytes and are what
// a third-party decoder accepts.
func TestGoldenVectors(t *testing.T) {
	g := loadGolden(t)
	patterns := parityPatterns()
	byKey := map[string]goldenEntry{}
	for _, e := range g.Vectors {
		byKey[goldenKey(e.Pattern, e.LevelsX, e.LevelsY, e.Br)] = e
	}
	for _, p := range patterns {
		for _, lv := range parityLevels() {
			for _, br := range goldenBrs() {
				key := goldenKey(p.name, lv[0], lv[1], br)
				entry, ok := byKey[key]
				if !ok {
					t.Fatalf("missing golden vector for %s", key)
				}
				for q := 0; q <= 15; q++ {
					got, err := EncodeRGGB8Block(p.raw, &Options{
						Quantization: uint8(q),
						LevelsX:      uint8(lv[0]),
						LevelsY:      uint8(lv[1]),
						StarTetrix:   true,
						CTS:          CTSParams{CT: 0},
						Br:           br,
					})
					if err != nil {
						t.Fatalf("%s q=%d levels=%d/%d br=%d: %v", p.name, q, lv[0], lv[1], br, err)
					}
					sum := sha256.Sum256(got)
					if hex.EncodeToString(sum[:]) != entry.SHA256[q] {
						t.Fatalf("%s q=%d levels=%d/%d br=%d: output differs from the reference\n  got  %d bytes, sha256 %x\n  want %d bytes, sha256 %s",
							p.name, q, lv[0], lv[1], br, len(got), sum, entry.Lengths[q], entry.SHA256[q])
					}
					if len(got) != entry.Lengths[q] {
						t.Fatalf("%s q=%d levels=%d/%d br=%d: length %d, want %d",
							p.name, q, lv[0], lv[1], br, len(got), entry.Lengths[q])
					}
				}
			}
		}
	}
}

// goldenKey builds the lookup key of one vector.
func goldenKey(pattern string, lx, ly int, br uint8) string {
	return pattern + "/" + itoa(lx) + "-" + itoa(ly) + "/br" + itoa(int(br))
}

// itoa formats a small non-negative int without pulling in strconv for
// a handful of call sites.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [8]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// TestUpdateGolden regenerates the pinned vectors from the upstream
// bridge. It only runs when JPEgXS_UPDATE_GOLDEN is set.
func TestUpdateGolden(t *testing.T) {
	if os.Getenv("JPEgXS_UPDATE_GOLDEN") == "" {
		t.Skip("set JPEgXS_UPDATE_GOLDEN=1 to regenerate the golden vectors")
	}
	bridgeExe := bridgePath(t)
	if bridgeExe == "" {
		t.Fatalf("%s is not set; cannot regenerate the golden vectors", bridgeEnvVar)
	}
	var g goldenFile
	g.Encoder = goldenEncoderID
	for _, br := range goldenBrs() {
		for _, p := range parityPatterns() {
			for _, lv := range parityLevels() {
				entry := goldenEntry{
					Pattern: p.name,
					LevelsX: lv[0],
					LevelsY: lv[1],
					Br:      br,
					SHA256:  make([]string, 16),
					Lengths: make([]int, 16),
				}
				for q := 0; q <= 15; q++ {
					var b []byte
					if br == 8 {
						// The upstream Rust encoder's own bytes.
						b = runBridge(t, bridgeExe, p.raw, q, lv[0], lv[1])
					} else {
						// The same encoding with the conformant
						// bitplane-count width, taken from this port.
						var err error
						b, err = EncodeRGGB8Block(p.raw, &Options{
							Quantization: uint8(q),
							LevelsX:      uint8(lv[0]),
							LevelsY:      uint8(lv[1]),
							StarTetrix:   true,
							CTS:          CTSParams{CT: 0},
							Br:           br,
						})
						if err != nil {
							t.Fatalf("%s q=%d levels=%d/%d br=%d: %v", p.name, q, lv[0], lv[1], br, err)
						}
					}
					sum := sha256.Sum256(b)
					entry.SHA256[q] = hex.EncodeToString(sum[:])
					entry.Lengths[q] = len(b)
				}
				g.Vectors = append(g.Vectors, entry)
			}
		}
	}
	out, err := json.MarshalIndent(g, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.MkdirAll(filepath.Dir(filepath.FromSlash(goldenPath)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.FromSlash(goldenPath), append(out, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %d vectors to %s", len(g.Vectors), goldenPath)
}
