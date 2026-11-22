package dataset

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
)

type SourceFrame struct {
	TimestampUS int64
	EndUS       int64
}

// Probe keeps presentation timestamps in display order; it does not estimate them from FPS.
func Probe(ctx context.Context, ffprobe, input string) ([]SourceFrame, error) {
	cmd := exec.CommandContext(ctx, ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,duration_time,pkt_duration_time", "-of", "json", input)
	var out bytes.Buffer
	var log limitedLog
	cmd.Stdout = &out
	cmd.Stderr = &log
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffprobe: %w: %s", err, log.String())
	}
	var data struct {
		Frames []struct {
			PTS            string `json:"best_effort_timestamp_time"`
			Duration       string `json:"duration_time"`
			PacketDuration string `json:"pkt_duration_time"`
		} `json:"frames"`
	}
	if err := json.Unmarshal(out.Bytes(), &data); err != nil {
		return nil, err
	}
	if len(data.Frames) == 0 {
		return nil, fmt.Errorf("no video frames")
	}
	frames := make([]SourceFrame, len(data.Frames))
	var lastDuration int64
	for i, f := range data.Frames {
		pts, err := strconv.ParseFloat(f.PTS, 64)
		if err != nil {
			return nil, fmt.Errorf("frame %d has no usable presentation timestamp", i)
		}
		frames[i].TimestampUS = int64(pts*1e6 + 0.5)
		if i > 0 && frames[i].TimestampUS <= frames[i-1].TimestampUS {
			return nil, fmt.Errorf("non-increasing timestamps at frame %d", i)
		}
		duration := f.Duration
		if duration == "" {
			duration = f.PacketDuration
		}
		d, _ := strconv.ParseFloat(duration, 64)
		if i == len(frames)-1 {
			lastDuration = int64(d*1e6 + 0.5)
		}
	}
	if lastDuration <= 0 && len(frames) > 1 {
		lastDuration = frames[len(frames)-1].TimestampUS - frames[len(frames)-2].TimestampUS
	}
	if lastDuration <= 0 {
		lastDuration = 1_000_000 / FPS
	}
	for i := range frames {
		if i+1 < len(frames) {
			frames[i].EndUS = frames[i+1].TimestampUS
		} else {
			frames[i].EndUS = frames[i].TimestampUS + lastDuration
		}
	}
	return frames, nil
}

// SourceAt chooses the nearest source PTS on the offline 30 FPS output timeline.
// This tolerates timestamp rounding (e.g. Matroska millisecond time bases).
func SourceAt(frames []SourceFrame, output int64) (int, bool) {
	if len(frames) == 0 || output < 0 {
		return 0, false
	}
	t := frames[0].TimestampUS + output*1_000_000/FPS
	if t >= frames[len(frames)-1].EndUS {
		return 0, false
	}
	lo, hi := 0, len(frames)
	for lo < hi {
		mid := (lo + hi) / 2
		if frames[mid].TimestampUS <= t {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	index := lo - 1
	if lo < len(frames) && frames[lo].TimestampUS-t < t-frames[index].TimestampUS {
		index = lo
	}
	return index, true
}
