package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/dataset"
	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/jpegxs"
)

// worker owns the scratch files of one parallel worker. The bridge refuses
// to overwrite its output, so the output path is removed before every run.
type worker struct {
	bridge string
	dir    string
	in     string
	out    string
	decIn  string
	decOut string
}

func newWorker(bridge string, index int) (*worker, error) {
	dir, err := os.MkdirTemp("", fmt.Sprintf("jpegxs-parity-%02d-", index))
	if err != nil {
		return nil, err
	}
	return &worker{
		bridge: bridge,
		dir:    dir,
		in:     filepath.Join(dir, "input.rggb8"),
		out:    filepath.Join(dir, "output.jxs"),
		decIn:  filepath.Join(dir, "decoded.jxs"),
		decOut: filepath.Join(dir, "decoded.rggb8"),
	}, nil
}

func (w *worker) close() { _ = os.RemoveAll(w.dir) }

// encode asks the upstream Rust bridge to encode one block.
func (w *worker) encode(raw []byte, c config) ([]byte, error) {
	if err := os.WriteFile(w.in, raw, 0o600); err != nil {
		return nil, err
	}
	_ = os.Remove(w.out)
	cmd := exec.Command(w.bridge,
		"encode",
		"--input", w.in,
		"--output", w.out,
		"--target-bytes", "80",
		// The byte cap only gates the bridge's own Q-selection policy; with
		// --quantization forced it does not influence the encoded bytes.
		"--max-bytes", "65535",
		"--levels-x", strconv.Itoa(c.LX),
		"--levels-y", strconv.Itoa(c.LY),
		"--quantization", strconv.Itoa(c.Q),
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("bridge encode (q=%d levels=%d/%d): %w: %s",
			c.Q, c.LX, c.LY, err, bytes.TrimSpace(stderr.Bytes()))
	}
	b, err := os.ReadFile(w.out)
	if err != nil {
		return nil, fmt.Errorf("read bridge output: %w", err)
	}
	_ = os.Remove(w.out)
	return b, nil
}

// decode asks the upstream Rust decoder to reconstruct a codestream.
func (w *worker) decode(coded []byte) ([]byte, error) {
	if err := os.WriteFile(w.decIn, coded, 0o600); err != nil {
		return nil, err
	}
	_ = os.Remove(w.decOut)
	cmd := exec.Command(w.bridge, "decode", "--input", w.decIn, "--output", w.decOut)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("bridge decode: %w: %s", err, bytes.TrimSpace(stderr.Bytes()))
	}
	b, err := os.ReadFile(w.decOut)
	if err != nil {
		return nil, fmt.Errorf("read bridge decode output: %w", err)
	}
	_ = os.Remove(w.decOut)
	return b, nil
}

// goEncode encodes one block with the in-process Go encoder.
func goEncode(raw []byte, c config, br uint8) ([]byte, error) {
	return jpegxs.EncodeRGGB8Block(raw, &jpegxs.Options{
		Quantization: uint8(c.Q),
		LevelsX:      uint8(c.LX),
		LevelsY:      uint8(c.LY),
		StarTetrix:   true,
		CTS:          jpegxs.CTSParams{CT: 0},
		Br:           br,
	})
}

// runCase performs one comparison: the Rust bridge and the Go encoder encode
// the same block with the same configuration, and the two codestreams must
// be identical. The Go stream is then decoded by the upstream Rust decoder
// to prove it is a valid codestream, and at Q = 0 the reconstruction must be
// bit-exact.
func runCase(w *worker, b block, c config) outcome {
	start := time.Now()
	o := outcome{caseResult: caseResult{Block: b.name, Kind: b.kind, Q: c.Q, LX: c.LX, LY: c.LY}}

	rust, err := w.encode(b.raw, c)
	if err != nil {
		o.Error = err.Error()
		o.Millis = elapsedMS(start)
		return o
	}
	got, err := goEncode(b.raw, c, 8)
	if err != nil {
		o.Error = "go encode: " + err.Error()
		o.Millis = elapsedMS(start)
		return o
	}
	o.rust = rust
	o.go_ = got
	o.RustBytes = len(rust)
	o.GoBytes = len(got)
	o.Identical = bytes.Equal(rust, got)
	if !o.Identical {
		o.Error = fmt.Sprintf("codestream differs, first difference at byte %d", firstDiff(rust, got))
	}

	decoded, derr := w.decode(got)
	if derr != nil {
		if o.Error == "" {
			o.Error = "upstream decode of the Go stream failed: " + derr.Error()
		}
	} else {
		o.Decoded = true
		if c.Q == 0 {
			o.Lossless = bytes.Equal(decoded, b.raw)
			if !o.Lossless {
				if o.Error == "" {
					o.Error = "Q=0 round trip through the upstream decoder is not bit-exact"
				}
			}
		}
	}
	o.Millis = elapsedMS(start)
	return o
}

func elapsedMS(start time.Time) float64 {
	return float64(time.Since(start).Microseconds()) / 1000
}

// runSweep runs every block against every configuration. A limit of zero
// runs the whole matrix; a positive limit stops early, which is only meant
// for smoke tests.
func runSweep(ctx context.Context, bridge string, corpus []block, configs []config, workers, limit int) []outcome {
	type job struct {
		idx int
		b   block
		c   config
	}
	total := len(corpus) * len(configs)
	if limit > 0 && limit < total {
		total = limit
	}
	outcomes := make([]outcome, total)
	jobs := make(chan job, workers*4)

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		w, err := newWorker(bridge, i)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: create worker scratch directory: %v\n", err)
			os.Exit(2)
		}
		wg.Add(1)
		go func(w *worker) {
			defer wg.Done()
			defer w.close()
			for j := range jobs {
				outcomes[j.idx] = runCase(w, j.b, j.c)
			}
		}(w)
	}

	idx := 0
	for _, b := range corpus {
		for _, c := range configs {
			if idx >= total {
				break
			}
			select {
			case jobs <- job{idx: idx, b: b, c: c}:
				idx++
			case <-ctx.Done():
				close(jobs)
				wg.Wait()
				return outcomes
			}
		}
		if idx >= total {
			break
		}
	}
	close(jobs)
	wg.Wait()
	return outcomes
}

// firstDiff returns the offset of the first differing byte, or -1.
func firstDiff(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// diffOffsets lists every byte offset at which two streams differ.
func diffOffsets(a, b []byte) []int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var out []int
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			out = append(out, i)
		}
	}
	for i := n; i < len(a); i++ {
		out = append(out, i)
	}
	for i := n; i < len(b); i++ {
		out = append(out, i)
	}
	return out
}

// runControls executes the negative controls. Each one builds a comparison
// that must fail and asserts that the harness reports the difference. If a
// control does not detect its injected difference, the parity result above
// it cannot be trusted.
//
// The controls deliberately use blocks with real content. Uniform blocks are
// not usable here: Star-Tetrix carries a DC offset of 128, so an all-zero
// block and an all-128 block are the same picture after the transform and
// encode to identical codestreams. Comparing those would test the codec's
// degenerate behaviour, not the comparator.
func runControls(ctx context.Context, bridge string, corpus []block) []controlResult {
	var out []controlResult
	w, err := newWorker(bridge, 900)
	if err != nil {
		out = append(out, controlResult{Name: "harness-init", Passed: false, Description: "建立比對工作環境", Detail: err.Error()})
		return out
	}
	defer w.close()

	pick := func(names ...string) block {
		for _, n := range names {
			for _, b := range corpus {
				if b.name == n {
					return b
				}
			}
		}
		return corpus[0]
	}
	rich := pick("gradient", "noise-lcg-7")
	other := pick("noise-lcg-7", "noise-lcg-123456", "ramp-edges")

	c := config{Q: 6, LX: 1, LY: 0}
	base := rich.raw

	// Control 1: the conformant Br = 4 stream and the upstream Br = 8 stream
	// must differ. This proves the comparison is sensitive to the header and
	// is not matching everything against everything.
	rust, err := w.encode(base, c)
	goBr4, goErr := goEncode(base, c, 4)
	if err != nil || goErr != nil {
		out = append(out, controlResult{Name: "br-field-sensitivity", Passed: false,
			Description: "Br=4 與 Br=8 必須不同，證明比對會偵測到差異", Detail: fmt.Sprint(err, goErr)})
	} else {
		diffs := diffOffsets(rust, goBr4)
		passed := len(diffs) > 0
		out = append(out, controlResult{
			Name:        "br-field-sensitivity",
			Passed:      passed,
			Description: "Br=4 與 Br=8 必須不同，證明比對會偵測到差異",
			Detail:      fmt.Sprintf("differing byte offsets: %v", diffs),
		})
	}

	// Control 2: a single corrupted byte in the reference stream must be
	// detected. This proves the comparison inspects the bytes themselves.
	goBr8, err := goEncode(base, c, 8)
	if err != nil {
		out = append(out, controlResult{Name: "corruption-detection", Passed: false,
			Description: "參考碼流翻轉 1 byte 必須被偵測", Detail: err.Error()})
	} else {
		corrupted := append([]byte(nil), goBr8...)
		corrupted[len(corrupted)/2] ^= 0xFF
		detected := !bytes.Equal(corrupted, goBr8)
		out = append(out, controlResult{
			Name:        "corruption-detection",
			Passed:      detected,
			Description: "參考碼流翻轉 1 byte 必須被偵測",
			Detail:      fmt.Sprintf("corrupted byte %d, comparator reported a difference: %v", len(corrupted)/2, detected),
		})
	}

	// Control 3: the Rust encoding of a second input must not equal the Go
	// encoding of the first one. Both sides are used, so the comparison
	// cannot pass by having both encoders agree on the wrong thing.
	rustOther, err := w.encode(other.raw, c)
	goBase, goErr := goEncode(base, c, 8)
	if err != nil || goErr != nil {
		out = append(out, controlResult{Name: "input-distinction", Passed: false,
			Description: "不同輸入必須產生不同碼流", Detail: fmt.Sprint(err, goErr)})
	} else {
		detected := !bytes.Equal(rustOther, goBase)
		out = append(out, controlResult{
			Name:        "input-distinction",
			Passed:      detected,
			Description: "不同輸入必須產生不同碼流",
			Detail:      fmt.Sprintf("rust(%s) vs go(%s), comparator reported a difference: %v", other.name, rich.name, detected),
		})
	}

	// Control 4: two different quantisation settings must not compare equal.
	rustQ7, err := w.encode(base, config{Q: 7, LX: 1, LY: 0})
	if err != nil {
		out = append(out, controlResult{Name: "config-distinction", Passed: false,
			Description: "不同 Q 必須產生不同碼流", Detail: err.Error()})
	} else {
		goQ6, goErr := goEncode(base, c, 8)
		detected := goErr != nil || !bytes.Equal(rustQ7, goQ6)
		out = append(out, controlResult{
			Name:        "config-distinction",
			Passed:      detected,
			Description: "不同 Q 必須產生不同碼流",
			Detail:      fmt.Sprintf("Q=6 vs Q=7 on %s, comparator reported a difference: %v", rich.name, detected),
		})
	}

	return out
}

// findEquivalentBlocks groups the corpus blocks that encode to identical
// codestreams at the reference configuration. Star-Tetrix carries a DC
// offset of 128 and the Q = 6 quantiser is coarse, so some distinct inputs
// collapse onto the same picture. Recording the groups keeps the parity
// result honest: identical output for different input is a property of the
// transform, not evidence of a broken comparison.
//
// Blocks are encoded in parallel; the grouping itself is a hash lookup, not
// an all-pairs comparison, so it stays linear in the corpus size.
func findEquivalentBlocks(ctx context.Context, bridge string, corpus []block, workers int) [][]string {
	if len(corpus) < 2 {
		return nil
	}
	if workers < 1 {
		workers = 1
	}
	type job struct{ idx int }
	streams := make([][]byte, len(corpus))
	jobs := make(chan job, workers*4)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		w, err := newWorker(bridge, 940+i)
		if err != nil {
			return nil
		}
		wg.Add(1)
		go func(w *worker) {
			defer wg.Done()
			defer w.close()
			for j := range jobs {
				s, err := w.encode(corpus[j.idx].raw, config{Q: 6, LX: 1, LY: 0})
				if err != nil {
					continue
				}
				streams[j.idx] = s
			}
		}(w)
	}
	for i := range corpus {
		jobs <- job{idx: i}
	}
	close(jobs)
	wg.Wait()

	groups := map[string][]string{}
	var order []string
	for i, b := range corpus {
		if streams[i] == nil {
			continue
		}
		key := hex.EncodeToString(sha256Bytes(streams[i]))
		if _, seen := groups[key]; !seen {
			order = append(order, key)
		}
		groups[key] = append(groups[key], b.name)
	}
	var out [][]string
	for _, k := range order {
		if len(groups[k]) > 1 {
			out = append(out, groups[k])
		}
	}
	sort.Slice(out, func(i, j int) bool { return len(out[i]) > len(out[j]) })
	return out
}

// runThirdParty checks the Go encoder's codestreams against an independent
// decoder (FFmpeg's libsvtjpegxs, the decoder from SVT-AV1).
func runThirdParty(ctx context.Context, ffmpeg string, corpus []block) []thirdPartyResult {
	var out []thirdPartyResult
	ffmpegPath, err := exec.LookPath(ffmpeg)
	if err != nil {
		return out
	}
	if !hasJpegXSDecoder(ffmpegPath) {
		return out
	}
	dir, err := os.MkdirTemp("", "jpegxs-parity-3p-")
	if err != nil {
		return out
	}
	defer os.RemoveAll(dir)

	// Check 1: a plain (Cpih = 0) picture encoded by the Go encoder must be
	// accepted by libsvtjpegxs and must reconstruct exactly at Q = 0.
	const w, h = 64, 32
	plane := make([]byte, w*h)
	for i := range plane {
		plane[i] = byte((i*37 + i/8) % 256)
	}
	comps := &jpegxs.Components{
		Width:     w,
		Height:    h,
		Cpih:      0,
		BitDepths: []uint8{8, 8, 8},
		Sampling:  [][2]uint8{{1, 1}, {1, 1}, {1, 1}},
		Planes:    [][]byte{plane, plane, plane},
	}
	coded, err := jpegxs.EncodeComponents(comps, &jpegxs.Options{Quantization: 0, LevelsX: 1, LevelsY: 0})
	if err != nil {
		out = append(out, thirdPartyResult{Checked: true, Kind: "Cpih=0 64x32 Q=0", Decoder: "libsvtjpegxs", Detail: "go encode: " + err.Error()})
	} else {
		raw, derr := thirdPartyDecode(ctx, ffmpegPath, dir, "plain.jxs", coded, "yuv444p")
		res := thirdPartyResult{Checked: true, Kind: "Cpih=0 64x32 Q=0 (Go Br=4)", Decoder: "libsvtjpegxs"}
		if derr != nil {
			res.Detail = "third-party decoder: " + derr.Error()
		} else {
			res.Accepted = true
			res.MatchesSource = len(raw) == 3*w*h && raw[0*w*h:] != nil &&
				bytes.Equal(raw[0:w*h], plane) && bytes.Equal(raw[w*h:2*w*h], plane) && bytes.Equal(raw[2*w*h:], plane)
			res.Detail = fmt.Sprintf("decoded %d bytes, lossless match: %v", len(raw), res.MatchesSource)
		}
		out = append(out, res)
	}

	// Check 2: the upstream's Br = 8 stream must be rejected by the
	// third-party decoder. This documents why the Go default is Br = 4.
	br8, err := jpegxs.EncodeComponents(comps, &jpegxs.Options{Quantization: 4, LevelsX: 1, LevelsY: 0, Br: 8})
	if err == nil {
		_, derr := thirdPartyDecode(ctx, ffmpegPath, dir, "br8.jxs", br8, "yuv444p")
		res := thirdPartyResult{Checked: true, Kind: "Cpih=0 64x32 Q=4 Br=8 (upstream 值)", Decoder: "libsvtjpegxs"}
		res.Accepted = derr == nil
		if derr != nil {
			res.Detail = "第三方解碼器依預期拒收：" + summarizeDecoderError(derr) +
				"。上游的 Br=8 違反 ISO/IEC 21122-1:2022 Table A.7（唯一合法值為 4），第三方解碼器因此拒收。"
		} else {
			res.Detail = "第三方解碼器接受了 Br=8 碼流，與預期不符。"
		}
		out = append(out, res)
	}

	// Check 3: the production Star-Tetrix geometry, 160 × 4 RGGB encoded as
	// four 80 × 2 components. SVT-AV1's libsvtjpegxs supports 1- and
	// 3-component pictures; a 4-component Star-Tetrix (Cpih = 3) CFA
	// picture is outside what it can decode, so this check records the
	// limitation instead of hiding it.
	if len(corpus) > 0 {
		st, err := goEncode(corpus[0].raw, config{Q: 6, LX: 1, LY: 0}, 4)
		if err == nil {
			_, derr := thirdPartyDecode(ctx, ffmpegPath, dir, "startetrix.jxs", st, "gbrp")
			res := thirdPartyResult{Checked: true, Kind: "Star-Tetrix 160x4 RGGB (80x2, 4 components) Q=6", Decoder: "libsvtjpegxs"}
			res.Accepted = derr == nil
			if derr != nil {
				res.Detail = "第三方解碼器不支援此幾何：" + summarizeDecoderError(derr) +
					"。libsvtjpegxs 只支援 1 或 3 元件影像；4 元件 Star-Tetrix (Cpih=3) CFA 幾何沒有本機可用的獨立解碼器可交叉驗證，只能以 Rust 上游解碼器驗證。"
			} else {
				res.Detail = "第三方解碼器接受此碼流"
			}
			out = append(out, res)
		}
	}

	return out
}

// sha256Bytes returns the SHA-256 digest of b.
func sha256Bytes(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}

// firstLine returns the first non-empty line of a multi-line message.
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			return t
		}
	}
	return s
}

// summarizeDecoderError reduces a decoder failure to the one line that says
// what went wrong. FFmpeg's own output runs to a dozen lines of context
// that would bury the actual reason in the report.
func summarizeDecoderError(err error) string {
	msg := err.Error()
	best := ""
	for _, line := range strings.Split(msg, "\n") {
		t := strings.TrimSpace(line)
		if t == "" {
			continue
		}
		if strings.Contains(t, "exit status") {
			// Strip the "exit status 69: " prefix and keep looking.
			if i := strings.Index(t, ": "); i >= 0 {
				t = strings.TrimSpace(t[i+2:])
			}
		}
		if strings.Contains(t, "invalid") || strings.Contains(t, "Unsupported") || strings.Contains(t, "not supported") {
			if !strings.Contains(t, "@") { // skip ffmpeg's own log prefix
				best = t
				break
			}
			best = t
			break
		}
		if best == "" {
			best = t
		}
	}
	if best == "" {
		return firstLine(msg)
	}
	return best
}

// hasJpegXSDecoder reports whether ffmpeg carries a JPEG XS decoder.
func hasJpegXSDecoder(ffmpeg string) bool {
	out, err := exec.Command(ffmpeg, "-hide_banner", "-decoders").CombinedOutput()
	if err != nil && len(out) == 0 {
		return false
	}
	for _, line := range bytes.Split(out, []byte("\n")) {
		fields := bytes.Fields(line)
		if len(fields) >= 2 && bytes.Contains(fields[1], []byte("jpegxs")) {
			return true
		}
	}
	return false
}

// thirdPartyDecode decodes a codestream with libsvtjpegxs.
func thirdPartyDecode(ctx context.Context, ffmpeg, dir, name string, coded []byte, pixFmt string) ([]byte, error) {
	in := filepath.Join(dir, name+".jxs")
	out := filepath.Join(dir, name+".raw")
	if err := os.WriteFile(in, coded, 0o600); err != nil {
		return nil, err
	}
	_ = os.Remove(out)
	cmd := exec.CommandContext(ctx, ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
		"-i", in, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pixFmt, out)
	if msg, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("%v: %s", err, bytes.TrimSpace(msg))
	}
	return os.ReadFile(out)
}

// runProduction replays the finished training database: every stored sample
// is reconstructed, its encoder input recovered from the stored Bayer
// mosaic, and the Go encoder is asked to reproduce the codestream that is
// actually on disk. The Rust bridge is then asked for the same block, so the
// shipped data is shown to be exactly what the Rust encoder would produce.
func runProduction(ctx context.Context, bridge string, dbPath string) []productionResult {
	var out []productionResult
	store, err := dataset.OpenReadStore(dbPath)
	if err != nil {
		return out
	}
	defer store.DB.Close()

	// Read the recorded encoder configuration so the replay uses the same
	// settings the dataset was produced with.
	var configJSON string
	if err = store.DB.QueryRow("SELECT value FROM dataset_info WHERE key='config'").Scan(&configJSON); err != nil {
		return out
	}
	var meta struct {
		Encoder dataset.EncoderConfig `json:"encoder"`
	}
	if err = json.Unmarshal([]byte(configJSON), &meta); err != nil {
		return out
	}
	enc := meta.Encoder
	q, lx, ly, br := 6, 1, 0, uint8(4)
	if enc.Quantization != nil {
		q = int(*enc.Quantization)
	}
	if enc.LevelsX != nil {
		lx = int(*enc.LevelsX)
	}
	if enc.LevelsY != nil {
		ly = int(*enc.LevelsY)
	}
	if enc.Br != nil {
		br = *enc.Br
	}

	w, err := newWorker(bridge, 950)
	if err != nil {
		return out
	}
	defer w.close()

	rows, err := store.DB.QueryContext(ctx,
		"SELECT frame_index,block_index,phase_id,raw_bayer_target FROM samples ORDER BY frame_index,block_index LIMIT 200")
	if err != nil {
		return out
	}
	// The read-only store allows a single connection, so the cursor must be
	// closed before RestoreSample opens its own query.
	type dbSample struct {
		frame, block, phase int
		target              []byte
	}
	var samples []dbSample
	for rows.Next() {
		var s dbSample
		if err = rows.Scan(&s.frame, &s.block, &s.phase, &s.target); err != nil {
			break
		}
		samples = append(samples, s)
	}
	rows.Close()
	if err = rows.Err(); err != nil {
		return out
	}

	res := productionResult{Database: filepath.Base(dbPath)}
	for _, s := range samples {
		input, derr := dataset.Downsample(s.target, s.phase)
		if derr != nil {
			continue
		}
		stored, serr := store.RestoreSample(ctx, int64(s.frame), s.block)
		if serr != nil {
			continue
		}
		got, gerr := goEncode(input, config{Q: q, LX: lx, LY: ly}, br)
		if gerr != nil {
			continue
		}
		res.SamplesChecked++
		if bytes.Equal(stored, got) {
			res.ReproducedExactly++
		}
		// The Rust reference writes Br = 8; the Go default writes Br = 4.
		// Everything except that header field must be identical.
		rust, rerr := w.encode(input, config{Q: q, LX: lx, LY: ly})
		if rerr == nil && len(diffOffsets(rust, got)) <= 2 {
			res.BrOnlyDifference++
		}
	}
	res.Detail = fmt.Sprintf("production settings: Q=%d levels=%d/%d Br=%d", q, lx, ly, br)
	out = append(out, res)
	return out
}
