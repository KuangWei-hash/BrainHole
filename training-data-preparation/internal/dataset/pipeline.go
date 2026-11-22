package dataset

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type Options struct {
	Input, Output, FFmpeg, FFprobe, Split, Group string
	Resume                                       bool
	MaxFrames                                    int64
}

func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func Run(ctx context.Context, o Options, encoder Encoder) error {
	if o.Split != "train" && o.Split != "validation" && o.Split != "test" {
		return fmt.Errorf("split must be train, validation or test")
	}
	input, err := filepath.Abs(o.Input)
	if err != nil {
		return err
	}
	output, err := filepath.Abs(o.Output)
	if err != nil {
		return err
	}
	if strings.EqualFold(input, output) {
		return fmt.Errorf("input and output must differ")
	}
	frames, err := Probe(ctx, o.FFprobe, input)
	if err != nil {
		return err
	}
	hash, err := HashFile(input)
	if err != nil {
		return err
	}
	if o.Group == "" {
		o.Group = hash
	}
	version, err := exec.CommandContext(ctx, o.FFmpeg, "-version").Output()
	if err != nil {
		return fmt.Errorf("ffmpeg: %w", err)
	}
	meta := struct {
		Schema                                 int `json:"schema_version"`
		Source, Hash, Split, Group             string
		FFmpeg                                 string
		Width, Height, FPS, Phases, MaxEncoded int
		Resize, Bayer, Timing, Target          string
		Encoder                                EncoderConfig
	}{SchemaVersion, input, hash, o.Split, o.Group, strings.SplitN(string(version), "\n", 2)[0], Width, Height, FPS, 16, MaxEncodedBytes, "bilinear stretch; ffmpeg default color conversion", "RGGB direct RGB8 channel selection, no linearization", "nearest source PTS on offline output n/30 timeline; ties choose earlier", TargetDescription, encoder.EncoderConfig()}
	b, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	store, next, err := OpenStore(output, string(b), o.Resume)
	if err != nil {
		return err
	}
	defer store.DB.Close()
	log.Printf("source=%d frames, resume from output frame=%d, split=%s", len(frames), next, o.Split)
	decodeCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	cmd := exec.CommandContext(decodeCtx, o.FFmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", input, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "scale=640:480:flags=bilinear", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	var diagnostic limitedLog
	cmd.Stderr = &diagnostic
	if err = cmd.Start(); err != nil {
		return err
	}
	waited := false
	defer func() {
		if !waited {
			cancel()
			_ = stdout.Close()
			_ = cmd.Wait()
		}
	}()
	currentSource := -1
	rgb := make([]byte, RGBFrameBytes)
	for outputFrame := next; ; outputFrame++ {
		if o.MaxFrames > 0 && outputFrame >= o.MaxFrames {
			log.Printf("stopped at configured total-frame limit %d", o.MaxFrames)
			return nil
		}
		source, ok := SourceAt(frames, outputFrame)
		if !ok {
			break
		}
		for currentSource < source {
			if _, err = io.ReadFull(stdout, rgb); err != nil {
				return fmt.Errorf("decode source frame %d: %w; %s", currentSource+1, err, diagnostic.String())
			}
			currentSource++
		}
		phase := Phase(outputFrame)
		rows := make([]Sample, 0, BlockCount)
		for block := 0; block < BlockCount; block++ {
			rgbBlock := rgb[block*RGBBlockBytes : (block+1)*RGBBlockBytes]
			raw, e := Bayer(rgbBlock)
			if e != nil {
				return e
			}
			low, e := Downsample(raw, phase)
			if e != nil {
				return e
			}
			encoded, e := encoder.Encode(ctx, low)
			if e != nil {
				if logErr := store.LogError(outputFrame, block, e); logErr != nil {
					return fmt.Errorf("encode: %v; recording error: %w", e, logErr)
				}
				return fmt.Errorf("output frame %d block %d phase %d: %w", outputFrame, block, phase, e)
			}
			// The decoded mosaic is the future second model input; it is
			// produced by the in-process Go decoder from the very codestream
			// that was just encoded.
			decoded, e := DecodeBlock(encoded)
			if e != nil {
				if logErr := store.LogError(outputFrame, block, e); logErr != nil {
					return fmt.Errorf("decode: %v; recording error: %w", e, logErr)
				}
				return fmt.Errorf("output frame %d block %d phase %d: %w", outputFrame, block, phase, e)
			}
			rows = append(rows, Sample{Frame: outputFrame, SourceFrame: source, TimestampUS: frames[source].TimestampUS, Block: block, Phase: phase, Encoded: encoded, RawBayer: raw, Decoded: decoded})
		}
		if err = store.WriteFrame(ctx, rows); err != nil {
			if e := store.LogError(outputFrame, -1, err); e != nil {
				return fmt.Errorf("write frame: %v; log: %w", err, e)
			}
			return fmt.Errorf("output frame %d: %w", outputFrame, err)
		}
		log.Printf("frame=%d phase=%d source=%d: committed 30 samples", outputFrame, phase, source)
	}
	// Consume remaining source frames (e.g. frames dropped at the end of a high-FPS clip).
	for currentSource+1 < len(frames) {
		if _, err = io.ReadFull(stdout, rgb); err != nil {
			return fmt.Errorf("decode trailing frame: %w", err)
		}
		currentSource++
	}
	var extra [1]byte
	if n, e := stdout.Read(extra[:]); n != 0 || e != io.EOF {
		return fmt.Errorf("decoder/probe frame count mismatch")
	}
	err = cmd.Wait()
	waited = true
	if err != nil {
		return fmt.Errorf("ffmpeg: %w: %s", err, diagnostic.String())
	}
	_, err = store.DB.Exec("INSERT OR REPLACE INTO dataset_info(key,value) VALUES('status','complete')")
	if err == nil {
		log.Printf("dataset complete: %s", output)
	}
	return err
}
