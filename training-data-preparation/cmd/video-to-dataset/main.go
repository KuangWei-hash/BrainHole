package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/dataset"
)

func main() {
	var o dataset.Options
	var config string
	flag.StringVar(&o.Input, "input", "", "source video path (required)")
	flag.StringVar(&o.Output, "output", "", "SQLite output path (required)")
	flag.StringVar(&config, "encoder-config", "", "JPEG XS RGGB encoder JSON configuration (required)")
	flag.StringVar(&o.FFmpeg, "ffmpeg", "ffmpeg", "FFmpeg executable")
	flag.StringVar(&o.FFprobe, "ffprobe", "ffprobe", "FFprobe executable")
	flag.StringVar(&o.Split, "split", "train", "train, validation or test; split whole source/group consistently")
	flag.StringVar(&o.Group, "group", "", "independent scene/group ID (default: source SHA-256)")
	flag.BoolVar(&o.Resume, "resume", false, "resume an existing database only when source/settings match")
	flag.Int64Var(&o.MaxFrames, "max-frames", 0, "total output-frame limit for trials; 0 processes all frames")
	flag.Usage = func() {
		fmt.Fprintln(flag.CommandLine.Output(), "Video -> 640x480 RGB8 / 30 FPS -> 16-phase RGGB -> external JPEG XS -> SQLite\n\nAn actual RGGB JPEG XS encoder is required; no mock encoding is used.")
		flag.PrintDefaults()
	}
	flag.Parse()
	if o.Input == "" || o.Output == "" || config == "" || o.MaxFrames < 0 || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}
	encoder, err := dataset.LoadEncoder(config)
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	if err = dataset.Run(ctx, o, encoder); err != nil {
		log.Fatal(err)
	}
}
