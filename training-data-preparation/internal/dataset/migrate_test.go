package dataset

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"path/filepath"
	"strings"
	"testing"
)

// legacyV3Schema returns the schema v3 samples table: the current one
// without the decoded column.
func legacyV3Schema(t *testing.T) string {
	t.Helper()
	legacy := strings.ReplaceAll(schema, " decoded BLOB NOT NULL CHECK(length(decoded)=640),\n", "")
	if strings.Contains(legacy, "decoded") {
		t.Fatal("legacy schema still carries the decoded column")
	}
	return legacy
}

// writeV3Fixture creates a completed schema v3 dataset with one frame.
func writeV3Fixture(t *testing.T, path string) {
	t.Helper()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err = db.Exec(legacyV3Schema(t) + "PRAGMA user_version=3;"); err != nil {
		t.Fatal(err)
	}
	meta := `{"schema_version":3,"Encoder":{"identity":"fixture-codec"},"Target":"RAW8 Bayer RGGB row-major [16,640], 10240 bytes; shared-header-v1 payload padded to 640","Split":"train"}`
	if _, err = db.Exec("INSERT INTO dataset_info(key,value) VALUES('config',?),('status','complete')", meta); err != nil {
		t.Fatal(err)
	}
	h, payload, err := SplitCodestream(testCodestream(), "fixture-codec")
	if err != nil {
		t.Fatal(err)
	}
	definition, _ := json.Marshal(h)
	if _, err = db.Exec("INSERT INTO shared_header(singleton,id,definition) VALUES(1,?,?)", h.ID, string(definition)); err != nil {
		t.Fatal(err)
	}
	for block := 0; block < BlockCount; block++ {
		target := make([]byte, RawBlockBytes)
		for i := range target {
			target[i] = byte(block + i)
		}
		if _, err = db.Exec("INSERT INTO samples(frame_index,source_frame_index,source_timestamp_us,block_index,phase_id,header_id,jpegxs_payload,valid_length,codestream_length,raw_bayer_target) VALUES(0,0,1234,?,1,?,?,?,?,?)",
			block, h.ID, payload, len(payload), len(testCodestream()), target); err != nil {
			t.Fatal(err)
		}
	}
}

// TestMigrateToV4AddsDecodedColumn checks that the migration fills the new
// decoded column from the stored payload and leaves everything else alone.
func TestMigrateToV4AddsDecodedColumn(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	input, output := filepath.Join(dir, "v3.sqlite"), filepath.Join(dir, "v4.sqlite")
	writeV3Fixture(t, input)
	before, err := HashFile(input)
	if err != nil {
		t.Fatal(err)
	}
	result, err := MigrateToV4(ctx, input, output)
	if err != nil {
		t.Fatal(err)
	}
	if result.Frames != 1 || result.Samples != BlockCount {
		t.Fatalf("migration result %+v", result)
	}
	after, err := HashFile(input)
	if err != nil || before != after {
		t.Fatal("source database changed", err)
	}
	destination, err := OpenReadStore(output)
	if err != nil {
		t.Fatal(err)
	}
	defer destination.DB.Close()
	if destination.Version != 4 {
		t.Fatalf("migrated database is schema v%d", destination.Version)
	}
	source, err := OpenReadStore(input)
	if err != nil {
		t.Fatal(err)
	}
	defer source.DB.Close()
	wantDecoded := testDecoded()
	for block := 0; block < BlockCount; block++ {
		decoded, err := destination.RestoreDecoded(ctx, 0, block)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(decoded, wantDecoded) {
			t.Fatalf("block %d: decoded mosaic mismatch", block)
		}
		var payload, target []byte
		var size int
		if err = destination.DB.QueryRow("SELECT jpegxs_payload,codestream_length,raw_bayer_target FROM samples WHERE frame_index=0 AND block_index=?", block).Scan(&payload, &size, &target); err != nil {
			t.Fatal(err)
		}
		var oldPayload, oldTarget []byte
		if err = source.DB.QueryRow("SELECT jpegxs_payload,raw_bayer_target FROM samples WHERE frame_index=0 AND block_index=?", block).Scan(&oldPayload, &oldTarget); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(payload, oldPayload) {
			t.Fatalf("block %d: payload changed", block)
		}
		if !bytes.Equal(target, oldTarget) {
			t.Fatalf("block %d: RAW8 target changed", block)
		}
	}
	// The provenance metadata records the migration.
	var stored string
	if err = destination.DB.QueryRow("SELECT value FROM dataset_info WHERE key='config'").Scan(&stored); err != nil {
		t.Fatal(err)
	}
	var meta map[string]any
	if err = json.Unmarshal([]byte(stored), &meta); err != nil {
		t.Fatal(err)
	}
	if meta["schema_version"].(float64) != 4 || meta["Migration"] == nil {
		t.Fatalf("migration metadata missing: %v", meta)
	}
	// A second migration must not overwrite an existing file.
	if _, err = MigrateToV4(ctx, input, output); err == nil {
		t.Fatal("overwrote an existing migration output")
	}
	// A v4 database is not a migration source.
	if _, err = MigrateToV4(ctx, output, filepath.Join(dir, "again.sqlite")); err == nil {
		t.Fatal("migrated a schema v4 database")
	}
}

// TestExportStreamCarriesDecodedMosaic checks the BHXS v2 stream: the
// decoded mosaic rides along with the payload and survives the round trip.
func TestExportStreamCarriesDecodedMosaic(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "v4.sqlite")
	store, _, err := OpenStore(path, `{"Encoder":{"identity":"fixture-codec"}}`, false)
	if err != nil {
		t.Fatal(err)
	}
	if err = store.WriteFrame(ctx, testRows(0)); err != nil {
		t.Fatal(err)
	}
	_ = store.DB.Close()

	read, err := OpenReadStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer read.DB.Close()

	var withDecoded, withoutDecoded bytes.Buffer
	if err = read.ExportStream(ctx, &withDecoded, true); err != nil {
		t.Fatal(err)
	}
	if err = read.ExportStream(ctx, &withoutDecoded, false); err != nil {
		t.Fatal(err)
	}
	if withDecoded.Len() <= withoutDecoded.Len() {
		t.Fatalf("the decoded stream is not larger: %d vs %d", withDecoded.Len(), withoutDecoded.Len())
	}
	// The decoded mosaic costs exactly 640 bytes per block; the stream
	// configuration also grows by the include_decoded flag.
	overhead := withDecoded.Len() - withoutDecoded.Len()
	if overhead < BlockCount*DecodedBytes || overhead > BlockCount*DecodedBytes+64 {
		t.Fatalf("unexpected decoded overhead: %d bytes", overhead)
	}

	reader, err := NewStreamReader(bytes.NewReader(withDecoded.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	want := testDecoded()
	count := 0
	for {
		item, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if item.Block != count || item.Frame != 0 {
			t.Fatalf("sample order %d/%d", item.Frame, item.Block)
		}
		if !bytes.Equal(item.Decoded, want) {
			t.Fatalf("block %d: stream decoded mosaic mismatch", item.Block)
		}
		count++
	}
	if count != BlockCount {
		t.Fatalf("stream carried %d samples", count)
	}

	// The payload-only stream stays readable and carries no mosaic.
	plain, err := NewStreamReader(bytes.NewReader(withoutDecoded.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	item, err := plain.Next()
	if err != nil {
		t.Fatal(err)
	}
	if item.Decoded != nil {
		t.Fatal("payload-only stream carries a decoded mosaic")
	}
}
