package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/dataset"
)

func main() {
	input := flag.String("input", "", "completed schema v2 SQLite with RGB8 targets")
	output := flag.String("output", "", "new schema v3 SQLite with RAW8 RGGB targets")
	flag.Parse()
	if *input == "" || *output == "" || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	result, err := dataset.ConvertRGBTargets(ctx, *input, *output)
	if err != nil {
		log.Fatal(err)
	}
	info, err := os.Stat(*output)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("converted %d frames / %d samples to RAW8 RGGB targets; output=%d bytes; source retained", result.Frames, result.Samples, info.Size())
}
