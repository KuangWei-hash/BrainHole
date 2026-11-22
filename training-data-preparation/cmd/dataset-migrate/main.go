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
	input := flag.String("input", "", "source SQLite schema v3 dataset (read only)")
	output := flag.String("output", "", "new output file (never overwrites)")
	flag.Parse()
	if *input == "" || *output == "" || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*input, *output); err != nil {
		log.Fatal(err)
	}
}

func run(input, output string) error {
	result, err := dataset.MigrateToV4(context.Background(), input, output)
	if err != nil {
		return err
	}
	fmt.Printf("migrated %d frames / %d samples into schema v4: %s\n", result.Frames, result.Samples, output)
	return nil
}
