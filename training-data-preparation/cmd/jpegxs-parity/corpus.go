package main

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/dataset"
	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/jpegxs"
)

// block is one 160 × 4 RAW8 RGGB encoder input plus the provenance of how
// it was produced.
type block struct {
	name string
	kind string
	raw  []byte
}

func newBlock(name, kind string, raw []byte) block {
	return block{name: name, kind: kind, raw: raw}
}

// flatBlock returns a block whose every byte is v.
func flatBlock(v byte) []byte {
	b := make([]byte, jpegxs.RawBayerBlockWidth*jpegxs.RawBayerBlockHeight)
	for i := range b {
		b[i] = v
	}
	return b
}

// lcgBytes fills a block with a linear congruential sequence, the same
// generator the upstream bridge's own unit tests use.
func lcgBytes(seed uint32) []byte {
	b := make([]byte, jpegxs.RawBayerBlockWidth*jpegxs.RawBayerBlockHeight)
	s := seed
	for i := range b {
		s = s*1664525 + 1013904223
		b[i] = byte(s >> 24)
	}
	return b
}

// xorshiftBytes fills a block with a xorshift64* sequence. It is a different
// generator from lcgBytes on purpose: the two produce uncorrelated byte
// patterns, so a port that mishandles one of them would not be masked by
// the other.
//
// The seed is first run through a splitmix64 finaliser. Without that step
// consecutive seeds would collapse onto each other — the generator's state
// has to differ, not just its nominal seed value.
func xorshiftBytes(seed uint64) []byte {
	b := make([]byte, jpegxs.RawBayerBlockWidth*jpegxs.RawBayerBlockHeight)
	s := splitmix64(&seed)
	for i := range b {
		s ^= s >> 12
		s ^= s << 25
		s ^= s >> 27
		b[i] = byte((s * 2685821657736338717) >> 56)
	}
	return b
}

// splitmix64 mixes a seed so that nearby seeds produce unrelated states.
func splitmix64(seed *uint64) uint64 {
	*seed += 0x9E3779B97F4A7C15
	z := *seed
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// syntheticCorpus returns the deterministic, video-independent part of the
// corpus: the canonical upstream fixtures, structured edge cases and two
// families of pseudo-random noise.
func syntheticCorpus() []block {
	var out []block

	// The canonical fixtures, matching the upstream bridge's own unit-test
	// patterns so both sides see byte-identical inputs.
	out = append(out,
		newBlock("zeros", "canonical", flatBlock(0)),
		newBlock("flat128", "canonical", flatBlock(128)),
		newBlock("flat255", "canonical", flatBlock(255)),
	)
	gradient := make([]byte, jpegxs.RawBayerBlockWidth*jpegxs.RawBayerBlockHeight)
	for i := range gradient {
		gradient[i] = byte((i*7 + (i/160)*11) % 256)
	}
	out = append(out, newBlock("gradient", "canonical", gradient))
	out = append(out,
		newBlock("noise-lcg-1", "canonical", lcgBytes(1)),
		newBlock("noise-lcg-7", "canonical", lcgBytes(7)),
		newBlock("noise-lcg-123456", "canonical", lcgBytes(123456)),
	)
	ramp := make([]byte, jpegxs.RawBayerBlockWidth*jpegxs.RawBayerBlockHeight)
	for y := 0; y < jpegxs.RawBayerBlockHeight; y++ {
		for x := 0; x < jpegxs.RawBayerBlockWidth; x++ {
			ramp[y*jpegxs.RawBayerBlockWidth+x] = byte((x*3 + y*37) % 256)
		}
	}
	ramp[0] = 255
	ramp[jpegxs.RawBayerBlockWidth-1] = 255
	ramp[len(ramp)-1] = 0
	out = append(out, newBlock("ramp-edges", "canonical", ramp))

	// Structured edge cases: extremes, isolated impulses and patterns that
	// stress the wavelet's boundary handling and the Star-Tetrix
	// neighbour reflection.
	alt := make([]byte, len(ramp))
	for i := range alt {
		alt[i] = byte(i & 1)
	}
	out = append(out, newBlock("alternating-01", "structured", alt))

	checker := make([]byte, len(ramp))
	for y := 0; y < jpegxs.RawBayerBlockHeight; y++ {
		for x := 0; x < jpegxs.RawBayerBlockWidth; x++ {
			if ((x/8)+(y/2))%2 == 0 {
				checker[y*jpegxs.RawBayerBlockWidth+x] = 255
			}
		}
	}
	out = append(out, newBlock("checker-8x8", "structured", checker))

	impulse := flatBlock(128)
	impulse[0] = 255
	out = append(out, newBlock("impulse-first", "structured", impulse))

	impulse = flatBlock(128)
	impulse[len(impulse)-1] = 0
	out = append(out, newBlock("impulse-last", "structured", impulse))

	impulse = flatBlock(128)
	impulse[len(impulse)/2] = 255
	impulse[len(impulse)/2+1] = 0
	out = append(out, newBlock("impulse-middle", "structured", impulse))

	// A step edge down the middle of the block.
	step := make([]byte, len(ramp))
	for y := 0; y < jpegxs.RawBayerBlockHeight; y++ {
		for x := 0; x < jpegxs.RawBayerBlockWidth; x++ {
			if x < jpegxs.RawBayerBlockWidth/2 {
				step[y*jpegxs.RawBayerBlockWidth+x] = 20
			} else {
				step[y*jpegxs.RawBayerBlockWidth+x] = 230
			}
		}
	}
	out = append(out, newBlock("step-edge", "structured", step))

	// Every Bayer colour channel saturated differently.
	channel := make([]byte, len(ramp))
	for y := 0; y < jpegxs.RawBayerBlockHeight; y++ {
		for x := 0; x < jpegxs.RawBayerBlockWidth; x++ {
			c := (y%2)*2 + x%2
			channel[y*jpegxs.RawBayerBlockWidth+x] = []byte{255, 0, 128, 64}[c]
		}
	}
	out = append(out, newBlock("channel-saturated", "structured", channel))

	// Near-flat values around the Star-Tetrix DC offset of 128.
	nearFlat := make([]byte, len(ramp))
	for i := range nearFlat {
		nearFlat[i] = byte(126 + (i % 3))
	}
	out = append(out, newBlock("near-flat-126", "structured", nearFlat))

	// A single spike in a flat field: the sparsest possible payload.
	spike := flatBlock(100)
	spike[317] = 255
	out = append(out, newBlock("single-spike", "structured", spike))

	// High-frequency noise, which stresses the run coder and the packet
	// header escaping.
	highFreq := make([]byte, len(ramp))
	for i := range highFreq {
		highFreq[i] = byte((i*97 + (i&1)*128) % 256)
	}
	out = append(out, newBlock("high-frequency", "structured", highFreq))

	// Full-scale sawtooth across the whole block.
	saw := make([]byte, len(ramp))
	for i := range saw {
		saw[i] = byte(i % 256)
	}
	out = append(out, newBlock("sawtooth", "structured", saw))

	// Values that repeat in long runs, which the coder may compress into
	// run-length segments.
	runLength := make([]byte, len(ramp))
	for i := range runLength {
		runLength[i] = byte((i / 37) % 256)
	}
	out = append(out, newBlock("long-runs", "structured", runLength))

	// Pseudo-random noise, two generator families.
	for i := 0; i < 12; i++ {
		out = append(out, newBlock(fmt.Sprintf("noise-lcg-%d", 1000+i), "random-lcg", lcgBytes(uint32(1000+i))))
	}
	for i := 0; i < 12; i++ {
		out = append(out, newBlock(fmt.Sprintf("noise-xorshift-%d", 2000+i), "random-xorshift", xorshiftBytes(uint64(2000+i))))
	}

	return out
}

// probeFrameCount returns the number of video frames in a file, or 0 when
// it cannot be determined. It is used to spread the sampled frames evenly
// over the whole video instead of clustering them at the beginning.
func probeFrameCount(ctx context.Context, ffprobe, path string) int {
	if ffprobe == "" {
		return 0
	}
	cmd := exec.CommandContext(ctx, ffprobe, "-v", "error", "-select_streams", "v:0",
		"-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path)
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// videoCorpus extracts real blocks from a video file using the production
// decoder settings: the same ffmpeg scaling, the same Bayer simulation and
// the same 16-phase downsampling as the training pipeline.
//
// Frames are spread evenly over the whole video: the frame count is probed
// first and the step is chosen so the sampled frames cover the entire clip
// rather than its opening seconds. A nil rows list means every block row
// of a frame.
//
// The decoder may still be producing frames when enough blocks have been
// collected; it is stopped before waiting for it, because a child blocked on
// a full pipe would deadlock the wait.
func videoCorpus(ctx context.Context, ffmpeg, ffprobe, path string, startFrame, frameCount, step int, rows []int, want int) ([]block, error) {
	if rows == nil {
		rows = make([]int, dataset.BlockCount)
		for i := range rows {
			rows[i] = i
		}
	}
	// Spread the samples over the whole video when the length is known.
	if step <= 0 {
		total := probeFrameCount(ctx, ffprobe, path)
		needed := (want + len(rows) - 1) / len(rows)
		if total > 0 && needed > 0 {
			step = total / needed
			if step < 1 {
				step = 1
			}
		} else {
			step = 1
		}
	}
	args := []string{
		"-hide_banner", "-loglevel", "error", "-nostdin",
		"-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
		"-vf", "scale=640:480:flags=bilinear", "-fps_mode", "passthrough",
		"-pix_fmt", "rgb24",
	}
	if frameCount > 0 {
		args = append(args, "-frames:v", fmt.Sprint(startFrame+frameCount))
	}
	args = append(args, "-f", "rawvideo", "pipe:1")
	cmd := exec.CommandContext(ctx, ffmpeg, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr []byte
	cmd.Stderr = &writerFunc{&stderr}
	if err = cmd.Start(); err != nil {
		return nil, err
	}
	frame := make([]byte, dataset.RGBFrameBytes)
	var out []block
	for f := 0; ; f++ {
		if len(out) >= want {
			break
		}
		if _, err = io.ReadFull(stdout, frame); err != nil {
			break
		}
		if f < startFrame || (f-startFrame)%step != 0 {
			continue
		}
		phase := dataset.Phase(int64(f))
		for _, row := range rows {
			if len(out) >= want {
				break
			}
			if row < 0 || row >= dataset.BlockCount {
				continue
			}
			rgbBlock := frame[row*dataset.RGBBlockBytes : (row+1)*dataset.RGBBlockBytes]
			raw, e := dataset.Bayer(rgbBlock)
			if e != nil {
				break
			}
			low, e := dataset.Downsample(raw, phase)
			if e != nil {
				break
			}
			name := fmt.Sprintf("%s-f%03d-p%02d-r%02d", baseName(path), f, phase, row)
			out = append(out, newBlock(name, "video", low))
		}
	}
	// The decoder may still be producing frames. Stop it before waiting:
	// a child blocked on a full pipe would otherwise deadlock the wait.
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
	if len(out) == 0 {
		return nil, fmt.Errorf("no blocks extracted from %s: %s", path, bytes.TrimSpace(stderr))
	}
	return out, nil
}

// databaseCorpus pulls real blocks out of a finished training database. The
// database stores the pre-downsample 640 × 16 Bayer mosaic plus the phase,
// so the exact 160 × 4 encoder input can be reconstructed and compared
// against what the production run actually encoded.
//
// Samples are spread across the whole video instead of taking the first
// rows: the opening frames are a fade-in from black and would contribute
// only uniform blocks.
func databaseCorpus(ctx context.Context, db *sql.DB, want int) ([]block, error) {
	// A diagonal spread: one frame every 89 frames, and a different block
	// row in each of them. The opening frames of a video are a fade-in from
	// black, so taking the first rows would contribute only uniform blocks.
	rows, err := db.QueryContext(ctx,
		"SELECT frame_index,block_index,phase_id,raw_bayer_target FROM samples "+
			"WHERE frame_index % 89 = 40 AND block_index = (frame_index / 89) % 30 "+
			"ORDER BY frame_index")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []block
	for rows.Next() {
		var frame, blockIndex, phase int
		var target []byte
		if err = rows.Scan(&frame, &blockIndex, &phase, &target); err != nil {
			return nil, err
		}
		low, e := dataset.Downsample(target, phase)
		if e != nil {
			return nil, e
		}
		out = append(out, newBlock(fmt.Sprintf("db-f%04d-b%02d-p%02d", frame, blockIndex, phase), "production-db", low))
		if len(out) >= want {
			break
		}
	}
	return out, rows.Err()
}

// isUniform reports whether every byte of a block has the same value. A
// uniform block compresses to a handful of bytes and exercises very little
// of the encoder, so the corpus keeps track of how many of them there are.
func isUniform(raw []byte) bool {
	if len(raw) == 0 {
		return true
	}
	for _, b := range raw[1:] {
		if b != raw[0] {
			return false
		}
	}
	return true
}

// writerFunc collects a command's stderr into a byte slice.
type writerFunc struct{ b *[]byte }

func (w *writerFunc) Write(p []byte) (int, error) {
	*w.b = append(*w.b, p...)
	return len(p), nil
}

func baseName(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			return path[i+1:]
		}
	}
	return path
}
