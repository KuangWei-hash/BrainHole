package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/KuangWei-hash/BrainHole/training-data-preparation/internal/dataset"
)

func main() {
	input := flag.String("input", "", "SQLite schema v2/v3/v4 dataset")
	output := flag.String("output", "", "new output file (never overwrites)")
	mode := flag.String("mode", "stream", "stream (.bhxs) or codestream (.jxs)")
	frame := flag.Int64("frame", 0, "frame index for codestream export")
	block := flag.Int("block", 0, "block index for codestream export")
	withDecoded := flag.Bool("with-decoded", true, "include the 640-byte decoded low-resolution mosaic of every block in the stream (schema v4 only)")
	flag.Parse()
	if *input == "" || *output == "" || flag.NArg() != 0 || (*mode != "stream" && *mode != "codestream") || *frame < 0 || *block < 0 || *block >= 30 {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*input, *output, *mode, *frame, *block, *withDecoded); err != nil {
		log.Fatal(err)
	}
}

func run(input, output, mode string, frame int64, block int, withDecoded bool) error {
	s, err := dataset.OpenReadStore(input)
	if err != nil {
		return err
	}
	defer s.DB.Close()
	f, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	if mode == "stream" {
		err = s.ExportStream(context.Background(), f, withDecoded)
	} else {
		var full []byte
		full, err = s.RestoreSample(context.Background(), frame, block)
		if err == nil {
			_, err = f.Write(full)
		}
	}
	if err != nil {
		return fmt.Errorf("export incomplete; partial output at %s: %w", output, err)
	}
	return f.Close()
}
