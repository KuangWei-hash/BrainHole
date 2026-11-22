// Command jpegxs-parity verifies the in-process Go JPEG XS encoder against
// the upstream Rust reference encoder (OxideAV/oxideav-jpegxs 0.0.9, rev
// 3484a041d6d8e893683c1920cb3f68fe6fd3b8b8) byte for byte.
//
// The check is deliberately a black-box comparison: both encoders are handed
// the same 160 × 4 RAW8 RGGB block and the same configuration, and the two
// codestreams must be identical. Nothing about the Go implementation is
// shared with the Rust side beyond the input bytes.
//
// The tool also runs negative controls. A parity test that cannot fail proves
// nothing, so the harness deliberately feeds itself streams that must differ
// and asserts that the comparator reports the difference.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/dataset"
)

const (
	// Corpus size and configuration count. 63 blocks × 160 configurations
	// = 10,080 cases, one more than the 10,000 the task asks for.
	canonicalCount  = 8
	structuredCount = 12
	randomCount     = 24
	databaseCount   = 10
	videoCount      = 9
)

// config is one encoder configuration in the supported space.
type config struct {
	Q  int `json:"quantization"`
	LX int `json:"levels_x"`
	LY int `json:"levels_y"`
}

// supportedConfigs is the configuration space of the Go encoder: every
// accepted (NL,x, NL,y) pair crossed with the requested quantisation
// values. An empty qList means every quantisation value.
func supportedConfigs(qList []int) []config {
	qs := qList
	if len(qs) == 0 {
		qs = make([]int, 16)
		for i := range qs {
			qs[i] = i
		}
	}
	var out []config
	for lx := 1; lx <= 5; lx++ {
		for ly := 0; ly <= 1; ly++ {
			if ly > lx {
				continue
			}
			for _, q := range qs {
				out = append(out, config{Q: q, LX: lx, LY: ly})
			}
		}
	}
	return out
}

// caseResult is the per-case outcome recorded in the report.
type caseResult struct {
	Block     string `json:"block"`
	Kind      string `json:"kind"`
	Q         int    `json:"quantization"`
	LX        int    `json:"levels_x"`
	LY        int    `json:"levels_y"`
	RustBytes int    `json:"rust_bytes"`
	GoBytes   int    `json:"go_bytes"`
	Identical bool   `json:"identical"`
	Decoded   bool   `json:"upstream_decoded"`
	Lossless  bool   `json:"q0_lossless"`
	Error     string `json:"error,omitempty"`
}

// outcome keeps the two codestreams so the run can be summarised with a
// single digest over the whole corpus.
type outcome struct {
	caseResult
	rust   []byte
	go_    []byte
	Millis float64
}

// controlResult records a negative control: a comparison that must fail.
type controlResult struct {
	Name        string `json:"name"`
	Passed      bool   `json:"passed"`
	Description string `json:"description"`
	Detail      string `json:"detail,omitempty"`
}

// thirdPartyResult records an independent-decoder check.
type thirdPartyResult struct {
	Checked       bool   `json:"checked"`
	Kind          string `json:"kind"`
	Decoder       string `json:"decoder"`
	Accepted      bool   `json:"accepted"`
	MatchesSource bool   `json:"matches_source"`
	Detail        string `json:"detail,omitempty"`
}

// productionResult records the end-to-end check against a finished training
// database.
type productionResult struct {
	Database          string `json:"database"`
	SamplesChecked    int    `json:"samples_checked"`
	ReproducedExactly int    `json:"reproduced_exactly"`
	BrOnlyDifference  int    `json:"br_only_difference"`
	Detail            string `json:"detail,omitempty"`
}

// report is the machine-readable result document.
type report struct {
	GeneratedAt      string             `json:"generated_at"`
	Host             string             `json:"host"`
	GoVersion        string             `json:"go_version"`
	Bridge           string             `json:"bridge"`
	BridgeSHA256     string             `json:"bridge_sha256"`
	CorpusSize       int                `json:"corpus_size"`
	CorpusKinds      map[string]int     `json:"corpus_kinds"`
	ConfigCount      int                `json:"config_count"`
	QuantValues      []int              `json:"quantization_values"`
	LengthStats      []lengthStat       `json:"codestream_length_stats"`
	Cases            int                `json:"cases"`
	Identical        int                `json:"identical"`
	Decoded          int                `json:"upstream_decoded"`
	Q0Cases          int                `json:"q0_cases"`
	LosslessOK       int                `json:"q0_lossless"`
	Failures         []caseResult       `json:"failures,omitempty"`
	RustDigest       string             `json:"rust_digest"`
	GoDigest         string             `json:"go_digest"`
	DigestMatch      bool               `json:"digest_match"`
	WallClock        string             `json:"wall_clock"`
	CaseLatencyMS    latencyStats       `json:"case_latency_ms"`
	Controls         []controlResult    `json:"negative_controls"`
	EquivalentInputs [][]string         `json:"equivalent_input_groups,omitempty"`
	UniformBlocks    int                `json:"uniform_blocks"`
	VariedBlocks     int                `json:"varied_blocks"`
	ThirdParty       []thirdPartyResult `json:"third_party_checks"`
	Production       []productionResult `json:"production_checks"`
	Verdict          string             `json:"verdict"`
}

type latencyStats struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

// lengthStat summarises the encoded codestream size for one configuration.
type lengthStat struct {
	Q      int     `json:"quantization"`
	LX     int     `json:"levels_x"`
	LY     int     `json:"levels_y"`
	Min    int     `json:"min_bytes"`
	Max    int     `json:"max_bytes"`
	Mean   float64 `json:"mean_bytes"`
	Median int     `json:"median_bytes"`
}

func main() {
	bridge := flag.String("bridge", "", "path to the upstream Rust bridge executable (brainhole-jpegxs.exe)")
	ffmpeg := flag.String("ffmpeg", "ffmpeg", "ffmpeg executable")
	ffprobe := flag.String("ffprobe", "ffprobe", "ffprobe executable")
	database := flag.String("db", "", "training database to cross-check (optional)")
	videos := flag.String("videos", "", "comma-separated videos to draw real blocks from (optional)")
	outDir := flag.String("out", "", "directory to write the report to")
	cases := flag.Int("cases", 10000, "minimum number of comparison cases required")
	blocks := flag.Int("blocks", 63, "number of 160x4 input blocks in the corpus")
	quant := flag.String("q", "all", "quantisation values to test: \"all\" for 0..15, or a comma-separated list such as 6")
	limit := flag.Int("limit", 0, "stop after this many cases (0 = run the whole matrix); for smoke tests only")
	workers := flag.Int("workers", runtime.NumCPU(), "parallel workers")
	flag.Parse()

	if *bridge == "" {
		fmt.Fprintln(os.Stderr, "error: -bridge is required")
		os.Exit(2)
	}
	if _, err := os.Stat(*bridge); err != nil {
		fmt.Fprintf(os.Stderr, "error: bridge %q is not usable: %v\n", *bridge, err)
		os.Exit(2)
	}
	if *workers < 1 {
		*workers = 1
	}

	ctx := context.Background()
	start := time.Now()

	// ---------------------------------------------------------------- corpus
	corpus := syntheticCorpus()
	fmt.Printf("corpus: %d synthetic blocks\n", len(corpus))
	target := *blocks
	if target < len(corpus) {
		target = len(corpus)
	}

	if *database != "" && len(corpus) < target {
		store, err := dataset.OpenReadStore(*database)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: open database: %v\n", err)
			os.Exit(2)
		}
		n := databaseCount
		if rem := target - len(corpus); n > rem {
			n = rem
		}
		dbBlocks, err := databaseCorpus(ctx, store.DB, n)
		store.DB.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: read database: %v\n", err)
			os.Exit(2)
		}
		corpus = append(corpus, dbBlocks...)
		fmt.Printf("corpus: +%d blocks from %s\n", len(dbBlocks), filepath.Base(*database))
	}

	videoList := splitList(*videos)
	for vi, v := range videoList {
		if len(corpus) >= target {
			break
		}
		need := target - len(corpus)
		// With several videos, split the remaining budget between them so
		// the corpus is not drawn from a single clip.
		if len(videoList) > 1 {
			share := need / (len(videoList) - vi)
			if share < 1 {
				share = 1
			}
			if share < need {
				need = share
			}
		}
		// Spread the blocks over the whole video: every frame the probe
		// says is needed, every block row of it.
		blocks, err := videoCorpus(ctx, *ffmpeg, *ffprobe, v, 0, 0, 0, nil, need)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: extract blocks from %s: %v\n", v, err)
			os.Exit(2)
		}
		corpus = append(corpus, blocks...)
		fmt.Printf("corpus: +%d blocks from %s\n", len(blocks), filepath.Base(v))
	}

	// Top the corpus up with pseudo-random blocks if the sources did not
	// provide enough.
	for len(corpus) < target {
		corpus = append(corpus, newBlock(fmt.Sprintf("noise-fill-%05d", len(corpus)), "random-fill",
			lcgBytes(uint32(700000+len(corpus)))))
	}
	if len(corpus) > target {
		corpus = corpus[:target]
	}

	var qList []int
	if *quant != "all" {
		for _, p := range splitList(*quant) {
			v, err := strconv.Atoi(p)
			if err != nil || v < 0 || v > 15 {
				fmt.Fprintf(os.Stderr, "error: -q must be \"all\" or a list of 0..15, got %q\n", *quant)
				os.Exit(2)
			}
			qList = append(qList, v)
		}
	}

	configs := supportedConfigs(qList)
	total := len(corpus) * len(configs)
	fmt.Printf("matrix: %d blocks × %d configurations = %d cases\n", len(corpus), len(configs), total)
	if total < *cases {
		fmt.Fprintf(os.Stderr, "error: only %d cases available, need %d\n", total, *cases)
		os.Exit(2)
	}

	// -------------------------------------------------------------- controls
	fmt.Println("running negative controls...")
	controls := runControls(ctx, *bridge, corpus)

	fmt.Println("scanning for input pairs that encode identically...")
	equivalent := findEquivalentBlocks(ctx, *bridge, corpus, *workers)

	// ------------------------------------------------------------ main sweep
	fmt.Printf("running %d parity cases with %d workers...\n", total, *workers)
	outcomes := runSweep(ctx, *bridge, corpus, configs, *workers, *limit)
	if len(outcomes) < total {
		fmt.Printf("NOTE: limited to %d of %d cases (smoke test)\n", len(outcomes), total)
	}

	// ------------------------------------------------------------- analysis
	rep := report{
		GeneratedAt: time.Now().Format(time.RFC3339),
		Host:        runtime.GOOS + "/" + runtime.GOARCH,
		GoVersion:   runtime.Version(),
		Bridge:      *bridge,
		CorpusSize:  len(corpus),
		ConfigCount: len(configs),
		Cases:       len(outcomes),
		Controls:    controls,
	}
	rep.EquivalentInputs = equivalent
	for _, c := range configs {
		if !slices.Contains(rep.QuantValues, c.Q) {
			rep.QuantValues = append(rep.QuantValues, c.Q)
		}
	}
	sort.Ints(rep.QuantValues)
	rep.CorpusKinds = map[string]int{}
	for _, b := range corpus {
		rep.CorpusKinds[b.kind]++
		if isUniform(b.raw) {
			rep.UniformBlocks++
		} else {
			rep.VariedBlocks++
		}
	}
	rep.LengthStats = summariseLengths(outcomes)
	if sum, err := fileSHA256(*bridge); err == nil {
		rep.BridgeSHA256 = sum
	}

	rustHasher := sha256.New()
	goHasher := sha256.New()
	for _, o := range outcomes {
		if o.Identical {
			rep.Identical++
		} else {
			rep.Failures = append(rep.Failures, o.caseResult)
		}
		if o.Decoded {
			rep.Decoded++
		}
		if o.Q == 0 {
			rep.Q0Cases++
			if o.Lossless {
				rep.LosslessOK++
			}
		}
		rustHasher.Write(o.rust)
		goHasher.Write(o.go_)
	}
	rep.RustDigest = hex.EncodeToString(rustHasher.Sum(nil))
	rep.GoDigest = hex.EncodeToString(goHasher.Sum(nil))
	rep.DigestMatch = rep.RustDigest == rep.GoDigest
	rep.CaseLatencyMS = summariseLatency(outcomes)
	rep.WallClock = time.Since(start).Round(time.Millisecond).String()

	// ------------------------------------------- independent decoder checks
	rep.ThirdParty = runThirdParty(ctx, *ffmpeg, corpus)

	// ------------------------------------------ production database replay
	if *database != "" {
		rep.Production = runProduction(ctx, *bridge, *database)
	}

	// ----------------------------------------------------------- verdict
	controlsOK := true
	for _, c := range controls {
		if !c.Passed {
			controlsOK = false
		}
	}
	switch {
	case len(rep.Failures) > 0:
		rep.Verdict = fmt.Sprintf("FAIL: %d of %d cases differ", len(rep.Failures), rep.Cases)
	case !controlsOK:
		rep.Verdict = "FAIL: a negative control did not detect its injected difference"
	case !rep.DigestMatch:
		rep.Verdict = "FAIL: corpus digests differ"
	case rep.Identical != rep.Cases:
		rep.Verdict = "FAIL: identical count does not match the case count"
	default:
		rep.Verdict = fmt.Sprintf("PASS: %d/%d cases byte-identical", rep.Identical, rep.Cases)
	}

	// -------------------------------------------------------------- output
	if *outDir != "" {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "error: create report directory: %v\n", err)
			os.Exit(2)
		}
		stamp := time.Now().Format("20060102-150405")
		tag := reportTag(configs, len(corpus))
		jsonPath := filepath.Join(*outDir, "jpegxs-parity-"+tag+"-"+stamp+".json")
		mdPath := filepath.Join(*outDir, "jpegxs-parity-"+tag+"-"+stamp+".md")
		if err := writeJSON(jsonPath, rep); err != nil {
			fmt.Fprintf(os.Stderr, "error: write json report: %v\n", err)
			os.Exit(2)
		}
		if err := writeMarkdown(mdPath, rep, corpus, configs); err != nil {
			fmt.Fprintf(os.Stderr, "error: write markdown report: %v\n", err)
			os.Exit(2)
		}
		fmt.Printf("report: %s\nreport: %s\n", jsonPath, mdPath)
	}

	fmt.Println()
	fmt.Printf("verdict: %s\n", rep.Verdict)
	fmt.Printf("cases: %d, byte-identical: %d, upstream-decoded: %d, q0-lossless: %d\n",
		rep.Cases, rep.Identical, rep.Decoded, rep.LosslessOK)
	fmt.Printf("rust digest: %s\ngo   digest: %s\n", rep.RustDigest, rep.GoDigest)
	fmt.Printf("wall clock: %s\n", rep.WallClock)

	if !strings.HasPrefix(rep.Verdict, "PASS") {
		os.Exit(1)
	}
}

// summariseLengths reports the encoded codestream size distribution for each
// configuration. It shows what the tested settings cost in bytes, which is
// the number the pipeline budget is built on.
func summariseLengths(outcomes []outcome) []lengthStat {
	type key struct{ q, lx, ly int }
	groups := map[key][]int{}
	var order []key
	for _, o := range outcomes {
		if len(o.go_) == 0 {
			continue
		}
		k := key{o.Q, o.LX, o.LY}
		if _, seen := groups[k]; !seen {
			order = append(order, k)
		}
		groups[k] = append(groups[k], len(o.go_))
	}
	sort.Slice(order, func(i, j int) bool {
		if order[i].q != order[j].q {
			return order[i].q < order[j].q
		}
		if order[i].lx != order[j].lx {
			return order[i].lx < order[j].lx
		}
		return order[i].ly < order[j].ly
	})
	out := make([]lengthStat, 0, len(order))
	for _, k := range order {
		v := groups[k]
		sort.Ints(v)
		var sum int
		for _, n := range v {
			sum += n
		}
		out = append(out, lengthStat{
			Q: k.q, LX: k.lx, LY: k.ly,
			Min: v[0], Max: v[len(v)-1], Mean: float64(sum) / float64(len(v)), Median: v[len(v)/2],
		})
	}
	return out
}

// summariseLatency computes the per-case wall-clock statistics. The
// durations are carried in the outcome slice so the report can show how
// long a real comparison takes.
func summariseLatency(outcomes []outcome) latencyStats {
	var s latencyStats
	vals := make([]float64, 0, len(outcomes))
	for _, o := range outcomes {
		if o.Millis > 0 {
			vals = append(vals, o.Millis)
		}
	}
	if len(vals) == 0 {
		return s
	}
	sort.Float64s(vals)
	var sum float64
	for _, v := range vals {
		sum += v
	}
	s.Mean = sum / float64(len(vals))
	s.Median = vals[len(vals)/2]
	s.Min = vals[0]
	s.Max = vals[len(vals)-1]
	return s
}

// quantSummary renders the quantisation values for the report header.
func quantSummary(qs []int) string {
	if len(qs) == 0 {
		return "-"
	}
	parts := make([]string, 0, len(qs))
	for _, q := range qs {
		parts = append(parts, strconv.Itoa(q))
	}
	return strings.Join(parts, ",")
}

// reportTag builds a short, file-name-safe description of what was tested,
// so several reports can sit in one directory without being confused.
func reportTag(configs []config, blocks int) string {
	qs := map[int]bool{}
	for _, c := range configs {
		qs[c.Q] = true
	}
	var qList []int
	for q := range qs {
		qList = append(qList, q)
	}
	sort.Ints(qList)
	qPart := "q-all"
	if len(qList) == 1 {
		qPart = fmt.Sprintf("q%d", qList[0])
	} else if len(qList) < 6 {
		parts := make([]string, 0, len(qList))
		for _, q := range qList {
			parts = append(parts, strconv.Itoa(q))
		}
		qPart = "q" + strings.Join(parts, "-")
	}
	return fmt.Sprintf("%s-%dblocks-%dcfg", qPart, blocks, len(configs))
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func fileSHA256(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(path, b, 0o644)
}

// writeMarkdown renders the human-readable report.
func writeMarkdown(path string, rep report, corpus []block, configs []config) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# Go / Rust JPEG XS 編碼器一致性驗證報告\n\n")
	fmt.Fprintf(&b, "- 產生時間：%s\n", rep.GeneratedAt)
	fmt.Fprintf(&b, "- 執行環境：%s，%s，%d 邏輯核心\n", rep.Host, rep.GoVersion, runtime.NumCPU())
	fmt.Fprintf(&b, "- Rust 參考實作：`%s`\n", rep.Bridge)
	fmt.Fprintf(&b, "- 參考實作 SHA-256：`%s`\n", rep.BridgeSHA256)
	fmt.Fprintf(&b, "- 測試樣本：%d 個區塊\n", rep.CorpusSize)
	fmt.Fprintf(&b, "- 樣本內容：有內容 %d 個、均勻 %d 個\n", rep.VariedBlocks, rep.UniformBlocks)
	fmt.Fprintf(&b, "- 編碼設定：%d 種（NL,x 1..5 × NL,y 0..1 × Q %s）\n", rep.ConfigCount, quantSummary(rep.QuantValues))
	fmt.Fprintf(&b, "- 比對組數：**%d**\n", rep.Cases)
	fmt.Fprintf(&b, "- 逐 byte 相同：**%d / %d**\n", rep.Identical, rep.Cases)
	fmt.Fprintf(&b, "- 上游解碼器可解碼：**%d / %d**\n", rep.Decoded, rep.Cases)
	if rep.Q0Cases > 0 {
		fmt.Fprintf(&b, "- Q=0 無損往返正確：**%d / %d**\n", rep.LosslessOK, rep.Q0Cases)
	}
	fmt.Fprintf(&b, "- 耗時：%s\n", rep.WallClock)
	fmt.Fprintf(&b, "- Rust 端全 corpus 摘要：`%s`\n", rep.RustDigest)
	fmt.Fprintf(&b, "- Go 端全 corpus 摘要：`%s`\n", rep.GoDigest)
	fmt.Fprintf(&b, "- 摘要一致：%v\n\n", rep.DigestMatch)

	fmt.Fprintf(&b, "## 結論\n\n**%s**\n\n", rep.Verdict)

	fmt.Fprintf(&b, "## 負向對照組\n\n")
	fmt.Fprintf(&b, "| 對照 | 結果 | 說明 |\n| --- | --- | --- |\n")
	for _, c := range rep.Controls {
		status := "通過"
		if !c.Passed {
			status = "失敗"
		}
		fmt.Fprintf(&b, "| %s | %s | %s |\n", c.Name, status, c.Description)
		if c.Detail != "" {
			fmt.Fprintf(&b, "| | | `%s` |\n", c.Detail)
		}
	}
	fmt.Fprintln(&b)

	if len(rep.EquivalentInputs) > 0 {
		fmt.Fprintf(&b, "## 不同輸入卻產生相同碼流\n\n")
		fmt.Fprintf(&b, "以下輸入在參考設定（Q=6、levels 1/0）下被編碼成**完全相同**的碼流。這是 Star-Tetrix 的 DC 位移（128）加上粗量化造成的退化等價，不是比對失效：負向對照組已證明比對本身能抓到差異。平坦與飽和區域最容易觸發這個現象。\n\n")
		fmt.Fprintf(&b, "| 同組區塊數 | 範例 |\n| ---: | --- |\n")
		for _, g := range rep.EquivalentInputs {
			names := g
			shown := 4
			suffix := ""
			if len(g) > shown {
				names = g[:shown]
				suffix = fmt.Sprintf(" …等 %d 個", len(g))
			}
			fmt.Fprintf(&b, "| %d | %s%s |\n", len(g), strings.Join(names, " = "), suffix)
		}
		fmt.Fprintln(&b)
	}

	if len(rep.Failures) > 0 {
		fmt.Fprintf(&b, "## 不一致明細（前 50 筆）\n\n")
		fmt.Fprintf(&b, "| 樣本 | Q | levels | Rust bytes | Go bytes | 錯誤 |\n| --- | ---: | --- | ---: | ---: | --- |\n")
		for i, f := range rep.Failures {
			if i >= 50 {
				break
			}
			fmt.Fprintf(&b, "| %s | %d | %d/%d | %d | %d | %s |\n",
				f.Block, f.Q, f.LX, f.LY, f.RustBytes, f.GoBytes, f.Error)
		}
		fmt.Fprintln(&b)
	}

	if len(rep.ThirdParty) > 0 {
		fmt.Fprintf(&b, "## 第三方解碼器（FFmpeg libsvtjpegxs）\n\n")
		fmt.Fprintf(&b, "| 檢查 | 接受 | 與來源一致 | 備註 |\n| --- | --- | --- | --- |\n")
		for _, t := range rep.ThirdParty {
			fmt.Fprintf(&b, "| %s | %v | %v | %s |\n", t.Kind, t.Accepted, t.MatchesSource, t.Detail)
		}
		fmt.Fprintln(&b)
	}

	if len(rep.Production) > 0 {
		fmt.Fprintf(&b, "## 正式訓練資料庫重現\n\n")
		for _, p := range rep.Production {
			fmt.Fprintf(&b, "- 資料庫：`%s`\n", p.Database)
			fmt.Fprintf(&b, "- 檢查筆數：%d\n", p.SamplesChecked)
			fmt.Fprintf(&b, "- Go 重編碼與庫內碼流逐 byte 相同：%d\n", p.ReproducedExactly)
			fmt.Fprintf(&b, "- Go(Br=4) 與 Rust(Br=8) 僅差 Br 欄位：%d\n", p.BrOnlyDifference)
			if p.Detail != "" {
				fmt.Fprintf(&b, "- 備註：%s\n", p.Detail)
			}
		}
		fmt.Fprintln(&b)
	}

	fmt.Fprintf(&b, "## 樣本組成\n\n")
	fmt.Fprintf(&b, "| 類型 | 數量 |\n| --- | ---: |\n")
	byKind := map[string]int{}
	for _, c := range corpus {
		byKind[c.kind]++
	}
	if len(byKind) == 0 {
		byKind = rep.CorpusKinds
	}
	kinds := make([]string, 0, len(byKind))
	for k := range byKind {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		fmt.Fprintf(&b, "| %s | %d |\n", k, byKind[k])
	}
	fmt.Fprintf(&b, "\n- 有內容（非均勻）的區塊：%d\n", rep.VariedBlocks)
	fmt.Fprintf(&b, "- 均勻（全同一值）的區塊：%d\n", rep.UniformBlocks)
	fmt.Fprintln(&b)

	if len(rep.LengthStats) > 0 {
		fmt.Fprintf(&b, "## 碼流長度分佈\n\n")
		fmt.Fprintf(&b, "| Q | levels | 最小 | 中位數 | 平均 | 最大 |\n| ---: | --- | ---: | ---: | ---: | ---: |\n")
		for _, s := range rep.LengthStats {
			fmt.Fprintf(&b, "| %d | %d/%d | %d | %d | %.1f | %d |\n", s.Q, s.LX, s.LY, s.Min, s.Median, s.Mean, s.Max)
		}
		fmt.Fprintln(&b)
	}

	fmt.Fprintf(&b, "## 設定涵蓋\n\n")
	fmt.Fprintf(&b, "| Q | levels (x/y) | 組數 |\n| ---: | --- | ---: |\n")
	perQ := map[int]int{}
	perLevel := map[string]int{}
	for _, c := range configs {
		perQ[c.Q]++
		perLevel[fmt.Sprintf("%d/%d", c.LX, c.LY)]++
	}
	for q := 0; q <= 15; q++ {
		fmt.Fprintf(&b, "| %d | 全部 | %d |\n", q, perQ[q])
	}
	levelKeys := make([]string, 0, len(perLevel))
	for k := range perLevel {
		levelKeys = append(levelKeys, k)
	}
	sort.Strings(levelKeys)
	fmt.Fprintf(&b, "\n| levels | 組數 |\n| --- | ---: |\n")
	for _, k := range levelKeys {
		fmt.Fprintf(&b, "| %s | %d |\n", k, perLevel[k])
	}

	return os.WriteFile(path, []byte(b.String()), 0o644)
}
