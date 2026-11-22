package dataset

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

func TestAllPhasesPreserveRGGBAndCoverSource(t *testing.T) {
	rgb := make([]byte, RGBBlockBytes)
	for i := 0; i < len(rgb); i += 3 {
		rgb[i] = 11
		rgb[i+1] = 77
		rgb[i+2] = 201
	}
	raw, err := Bayer(rgb)
	if err != nil {
		t.Fatal(err)
	}
	seen := make([]int, len(raw))
	for phase := 1; phase <= 16; phase++ {
		out, err := Downsample(raw, phase)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != 640 {
			t.Fatalf("phase %d has %d bytes", phase, len(out))
		}
		for y := 0; y < InputHeight; y++ {
			for x := 0; x < InputWidth; x++ {
				want := []byte{11, 77, 201}[y%2+x%2]
				if out[y*InputWidth+x] != want {
					t.Fatalf("phase %d breaks RGGB at %d,%d", phase, x, y)
				}
				sx := (x/2)*8 + ((phase-1)%4)*2 + x%2
				sy := (y/2)*8 + ((phase-1)/4)*2 + y%2
				seen[sy*Width+sx]++
			}
		}
	}
	for i, count := range seen {
		if count != 1 {
			t.Fatalf("source %d visited %d times", i, count)
		}
	}
	// Nonconstant pixel data ensures matching colors alone cannot hide a wrong tile selection.
	for y := 0; y < BlockHeight; y++ {
		for x := 0; x < Width; x++ {
			raw[y*Width+x] = byte((x/2)*3 + (y/2)*31)
		}
	}
	for p := 1; p <= 16; p++ {
		out, _ := Downsample(raw, p)
		for y := 0; y < InputHeight; y++ {
			for x := 0; x < InputWidth; x++ {
				sx := (x/2)*8 + ((p-1)%4)*2 + x%2
				sy := (y/2)*8 + ((p-1)/4)*2 + y%2
				if out[y*InputWidth+x] != raw[sy*Width+sx] {
					t.Fatalf("wrong spatial sample at phase %d", p)
				}
			}
		}
	}
	if Phase(0) != 1 || Phase(15) != 16 || Phase(16) != 1 {
		t.Fatal("wrong phase cycle")
	}
}

func TestTimingDropsDuplicatesAndNonzeroOrigin(t *testing.T) {
	frames := []SourceFrame{{1_000_000, 1_066_667}, {1_066_667, 1_133_334}}
	for output, want := range []int{0, 0, 1, 1, 1} {
		got, ok := SourceAt(frames, int64(output))
		if !ok || got != want {
			t.Fatalf("frame %d: %d %v", output, got, ok)
		}
	}
	fast := []SourceFrame{{0, 16_667}, {16_667, 33_333}, {33_333, 50_000}, {50_000, 66_667}}
	if got, _ := SourceAt(fast, 1); got != 2 {
		t.Fatalf("60->30 did not drop correct frames: %d", got)
	}
	if _, ok := SourceAt(fast, 3); ok {
		t.Fatal("generated frame beyond source duration")
	}
}

func testRows(frame int64) []Sample {
	rows := make([]Sample, BlockCount)
	for i := range rows {
		rows[i] = Sample{Frame: frame, SourceFrame: 0, Block: i, Phase: Phase(frame), Encoded: testCodestream(), RawBayer: make([]byte, RawBlockBytes), Decoded: testDecoded()}
	}
	return rows
}

// testDecoded is the decoded mosaic of the flat test codestream.
func testDecoded() []byte {
	d, err := DecodeBlock(testCodestream())
	if err != nil {
		panic(err)
	}
	return d
}

func TestStoreAtomicityResumeAndBounds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "video.sqlite")
	s, next, err := OpenStore(path, `{"schema":1}`, false)
	if err != nil {
		t.Fatal(err)
	}
	if next != 0 {
		t.Fatal(next)
	}
	bad := testRows(0)
	bad[20].Encoded = make([]byte, 641)
	if err = s.WriteFrame(context.Background(), bad); err == nil {
		t.Fatal("accepted oversized encoding")
	}
	var count int
	_ = s.DB.QueryRow("SELECT COUNT(*) FROM samples").Scan(&count)
	if count != 0 {
		t.Fatalf("partial frame persisted: %d", count)
	}
	if err = s.WriteFrame(context.Background(), testRows(0)); err != nil {
		t.Fatal(err)
	}
	_ = s.DB.Close()
	readOnly, e := OpenReadStore(path)
	if e != nil {
		t.Fatal(e)
	}
	_ = readOnly.DB.Close()
	if _, _, err = OpenStore(path, "different", true); err == nil {
		t.Fatal("accepted changed configuration")
	}
	s, next, err = OpenStore(path, `{"schema":1}`, true)
	if err != nil {
		t.Fatal(err)
	}
	defer s.DB.Close()
	if next != 1 {
		t.Fatalf("wrong checkpoint %d", next)
	}
	bad = testRows(1)
	bad[0].RawBayer = bad[0].RawBayer[:10]
	if err = s.WriteFrame(context.Background(), bad); err == nil {
		t.Fatal("accepted malformed RGB target")
	}
	bad = testRows(1)
	bad[0].Decoded = bad[0].Decoded[:10]
	if err = s.WriteFrame(context.Background(), bad); err == nil {
		t.Fatal("accepted malformed decoded mosaic")
	}
	bad = testRows(1)
	bad[0].Decoded = nil
	if err = s.WriteFrame(context.Background(), bad); err == nil {
		t.Fatal("accepted missing decoded mosaic")
	}
}

// This subprocess is only a test double for exercising I/O and validation, not a JPEG XS codec.
func TestEncoderSubprocess(t *testing.T) {
	var start int
	for i, a := range os.Args {
		if a == "brainhole-test-helper" {
			start = i
			break
		}
	}
	if start == 0 {
		return
	}
	args := os.Args[start+1:]
	if len(args) != 3 {
		os.Exit(2)
	}
	raw, err := os.ReadFile(args[0])
	if err != nil || len(raw) != 640 {
		os.Exit(3)
	}
	var out []byte
	switch args[2] {
	case "ok":
		out = testCodestream()
	case "large":
		out = make([]byte, 641)
	case "invalid":
		out = []byte("not-a-jpeg-xs-stream")
	}
	if os.WriteFile(args[1], out, 0600) != nil {
		os.Exit(4)
	}
	os.Exit(0)
}

func helperEncoder(t *testing.T, mode string) *CommandEncoder {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return &CommandEncoder{Config: EncoderConfig{Executable: exe, Identity: "TEST DOUBLE ONLY", TimeoutSeconds: 10, Args: []string{"-test.run=^TestEncoderSubprocess$", "--", "brainhole-test-helper", "{input}", "{output}", mode}}}
}

func TestCommandEncoderRejectsInvalidAndOversizedOutput(t *testing.T) {
	for _, mode := range []string{"ok", "large", "invalid"} {
		out, err := helperEncoder(t, mode).Encode(context.Background(), make([]byte, 640))
		if mode == "ok" {
			if err != nil || len(out) != len(testCodestream()) {
				t.Fatalf("valid protocol output: %x %v", out, err)
			}
		} else if err == nil {
			t.Fatalf("accepted %s output", mode)
		}
	}
}

func TestVideoPipelineAndResume(t *testing.T) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg unavailable")
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Skip("ffprobe unavailable")
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "source.mkv")
	output := filepath.Join(dir, "result.sqlite")
	cmd := exec.Command(ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=60:duration=0.1", "-c:v", "ffv1", input)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("test source: %v %s", err, b)
	}
	opts := Options{Input: input, Output: output, FFmpeg: ffmpeg, FFprobe: ffprobe, Split: "train", MaxFrames: 1}
	enc := helperEncoder(t, "ok")
	if err = Run(context.Background(), opts, enc); err != nil {
		t.Fatal(err)
	}
	opts.Resume = true
	opts.MaxFrames = 0
	if err = Run(context.Background(), opts, enc); err != nil {
		t.Fatal(err)
	}
	// Inspect directly; tests do not claim the helper produces real JPEG XS.
	// Re-open with the recorded configuration to validate completion and uniqueness.
	// Reading SQL directly here avoids manufacturing configuration strings.
	db, err := sql.Open("sqlite", output)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var count, phases int
	if err = db.QueryRow("SELECT COUNT(*), COUNT(DISTINCT phase_id) FROM samples").Scan(&count, &phases); err != nil {
		t.Fatal(err)
	}
	if count != 90 || phases != 3 {
		t.Fatalf("got %d samples / %d phases", count, phases)
	}
	var target []byte
	var source int
	if err = db.QueryRow("SELECT raw_bayer_target,source_frame_index FROM samples WHERE frame_index=1 AND block_index=0").Scan(&target, &source); err != nil {
		t.Fatal(err)
	}
	if len(target) != RawBlockBytes || source != 2 {
		t.Fatalf("bad pairing: source %d, size %d", source, len(target))
	}
	var state string
	_ = db.QueryRow("SELECT value FROM dataset_info WHERE key='status'").Scan(&state)
	if state != "complete" {
		t.Fatal("dataset not marked complete")
	}
}

func TestLoadEncoderConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "encoder.json")
	cfg := helperEncoder(t, "ok").Config
	b, _ := json.Marshal(cfg)
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	e, err := LoadEncoder(path)
	if err != nil {
		t.Fatal(err)
	}
	if out, err := e.Encode(context.Background(), bytes.Repeat([]byte{0}, 640)); err != nil || len(out) != len(testCodestream()) {
		t.Fatal(err)
	}
}

// Opt-in validation with the actual Rust encoder, not the protocol test double.
func testCodestream() []byte {
	// Actual synthetic flat RGGB codestream. Mocks return it without encoding input.
	b, err := hex.DecodeString("ff10ff5000044080ff12001a000000000000000000500002000000020404080808031040ff13000a0811081108110811ff14001200000001000200030104010501060107ff1800040000ff19001200000000800000000000800080008000ff200004000000000f0400000000000050000000000000000000000000000f04000000000000500000000000000000000000ff11")
	if err != nil {
		panic(err)
	}
	return b
}

func TestRealJPEGXSPipeline(t *testing.T) {
	config := os.Getenv("BRAINHOLE_JPEGXS_CONFIG")
	if config == "" {
		t.Skip("set BRAINHOLE_JPEGXS_CONFIG to test the real JPEG XS encoder")
	}
	nFrames := 1
	if value := os.Getenv("BRAINHOLE_REAL_FRAMES"); value != "" {
		var e error
		nFrames, e = strconv.Atoi(value)
		if e != nil || nFrames < 1 || nFrames > 300 {
			t.Fatal("BRAINHOLE_REAL_FRAMES must be 1..300")
		}
	}
	encoder, err := LoadEncoder(config)
	if err != nil {
		t.Fatal(err)
	}
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	input, output := filepath.Join(dir, "real-source.mkv"), filepath.Join(dir, "real.sqlite")
	cmd := exec.Command(ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30", "-frames:v", strconv.Itoa(nFrames), "-c:v", "ffv1", input)
	if b, e := cmd.CombinedOutput(); e != nil {
		t.Fatalf("source: %v %s", e, b)
	}
	if err = Run(context.Background(), Options{Input: input, Output: output, FFmpeg: ffmpeg, FFprobe: ffprobe, Split: "train"}, encoder); err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite", output)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var count, minLen, maxLen int
	var mean float64
	if err = db.QueryRow("SELECT COUNT(*),MIN(codestream_length),MAX(codestream_length),AVG(codestream_length) FROM samples").Scan(&count, &minLen, &maxLen, &mean); err != nil {
		t.Fatal(err)
	}
	if count != nFrames*30 || minLen < 4 || maxLen > 640 {
		t.Fatalf("unexpected real dataset: count=%d range=%d..%d", count, minLen, maxLen)
	}
	// Every sample must carry the decoded low-resolution mosaic, and it
	// must be exactly what the Go decoder produces from the stored payload.
	var decodedLen, decodedMismatch int
	if err = db.QueryRow("SELECT COUNT(*), SUM(CASE WHEN length(decoded)!=640 THEN 1 ELSE 0 END) FROM samples").Scan(&decodedLen, &decodedMismatch); err != nil {
		t.Fatal(err)
	}
	if decodedLen != count || decodedMismatch != 0 {
		t.Fatalf("decoded column incomplete: %d rows, %d malformed", decodedLen, decodedMismatch)
	}
	readStore0, err := OpenReadStore(output)
	if err != nil {
		t.Fatal(err)
	}
	defer readStore0.DB.Close()
	for _, sample := range [][2]int64{{0, 0}, {0, 15}, {int64(nFrames - 1), 29}} {
		full, e := readStore0.RestoreSample(context.Background(), sample[0], int(sample[1]))
		if e != nil {
			t.Fatal(e)
		}
		want, e := DecodeBlock(full)
		if e != nil {
			t.Fatal(e)
		}
		got, e := readStore0.RestoreDecoded(context.Background(), sample[0], int(sample[1]))
		if e != nil {
			t.Fatal(e)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("stored decoded mosaic differs from the Go decoder at %v", sample)
		}
	}
	t.Logf("real JPEG XS, synthetic testsrc: %d frames, %d samples; complete codestream %d..%d bytes; mean %.2f bytes; effective image payload %.0f bps at 30 FPS", nFrames, count, minLen, maxLen, mean, mean*30*30*8)
	readStore, err := OpenReadStore(output)
	if err != nil {
		t.Fatal(err)
	}
	defer readStore.DB.Close()
	header, err := readStore.Header()
	if err != nil {
		t.Fatal(err)
	}
	var stream bytes.Buffer
	if err = readStore.ExportStream(context.Background(), &stream, false); err != nil {
		t.Fatal(err)
	}
	reader, err := NewStreamReader(bytes.NewReader(stream.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	var total, payloadMin, payloadMax, n int
	for {
		item, e := reader.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			t.Fatal(e)
		}
		if item.Frame != int64(n/30) || item.Block != n%30 || item.Phase != Phase(item.Frame) {
			t.Fatal("wire sample pairing mismatch")
		}
		full, e := header.Restore(item.Payload)
		if e != nil {
			t.Fatal(e)
		}
		fromDB, e := readStore.RestoreSample(context.Background(), item.Frame, item.Block)
		if e != nil {
			t.Fatal(e)
		}
		if !bytes.Equal(full, fromDB) {
			t.Fatal("wire/SQLite reconstruction mismatch")
		}
		if n == 0 || len(item.Payload) < payloadMin {
			payloadMin = len(item.Payload)
		}
		if len(item.Payload) > payloadMax {
			payloadMax = len(item.Payload)
		}
		total += len(item.Payload)
		n++
	}
	if n != nFrames*30 {
		t.Fatalf("wire contained %d samples", n)
	}
	// The decoded variant of the stream carries the second model input.
	var decodedStream bytes.Buffer
	if err = readStore.ExportStream(context.Background(), &decodedStream, true); err != nil {
		t.Fatal(err)
	}
	decodedReader, err := NewStreamReader(bytes.NewReader(decodedStream.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	decodedCount := 0
	for {
		item, e := decodedReader.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			t.Fatal(e)
		}
		if len(item.Decoded) != DecodedBytes {
			t.Fatalf("sample %d carries %d decoded bytes", decodedCount, len(item.Decoded))
		}
		full, e := header.Restore(item.Payload)
		if e != nil {
			t.Fatal(e)
		}
		want, e := DecodeBlock(full)
		if e != nil {
			t.Fatal(e)
		}
		if !bytes.Equal(item.Decoded, want) {
			t.Fatalf("stream decoded mosaic differs from the Go decoder at %d", decodedCount)
		}
		decodedCount++
	}
	if decodedCount != nFrames*30 {
		t.Fatalf("decoded stream contained %d samples", decodedCount)
	}
	avg := float64(total) / float64(n)
	t.Logf("shared-header v1: %d frames, %d samples; prefix=%d suffix=%d; payload=%d..%d mean=%.2f; per-block lengths + frame indexes=%.0f bps (excluding session config/network); all SQLite/stream reconstructions match", nFrames, n, len(header.Prefix), len(header.Suffix), payloadMin, payloadMax, avg, ((avg+2)*30+8)*30*8)
	cfg := encoder.EncoderConfig()
	if filepath.Base(cfg.Executable) == "brainhole-jpegxs.exe" {
		for _, frame := range []int64{0, int64(nFrames - 1)} {
			full, e := readStore.RestoreSample(context.Background(), frame, 0)
			if e != nil {
				t.Fatal(e)
			}
			jxs := filepath.Join(dir, fmt.Sprintf("restored-%d.jxs", frame))
			raw := jxs + ".raw"
			if e = os.WriteFile(jxs, full, 0600); e != nil {
				t.Fatal(e)
			}
			if b, e := exec.Command(cfg.Executable, "decode", "--input", jxs, "--output", raw).CombinedOutput(); e != nil {
				t.Fatalf("decode reconstructed stream: %v %s", e, b)
			}
			info, e := os.Stat(raw)
			if e != nil || info.Size() != 640 {
				t.Fatalf("decoded geometry invalid: %v", e)
			}
			if nFrames == 1 {
				break
			}
		}
	}
}
