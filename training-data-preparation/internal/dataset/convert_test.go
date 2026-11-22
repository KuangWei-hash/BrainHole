package dataset

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func TestConvertRGBTargetsPreservesInputsAndSource(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	input, output := filepath.Join(dir, "rgb-v2.sqlite"), filepath.Join(dir, "raw-v3.sqlite")
	db, err := sql.Open("sqlite", input)
	if err != nil {
		t.Fatal(err)
	}
	legacy := strings.ReplaceAll(strings.ReplaceAll(schema, "raw_bayer_target", "rgb_target"), "=10240", "=30720")
	// The v2 schema predates the decoded column; drop it so the legacy
	// fixture really is a v2 table.
	legacy = strings.ReplaceAll(legacy, " decoded BLOB NOT NULL CHECK(length(decoded)=640),\n", "")
	if strings.Contains(legacy, "decoded") {
		t.Fatal("legacy schema still carries the decoded column")
	}
	if _, err = db.Exec(legacy + "PRAGMA user_version=2;"); err != nil {
		t.Fatal(err)
	}
	meta := `{"schema_version":2,"Encoder":{"identity":"fixture-codec"},"Target":"RGB8 HWC [16,640,3]","Split":"train"}`
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
		rgb := make([]byte, RGBBlockBytes)
		for i := 0; i < len(rgb); i += 3 {
			rgb[i] = byte(11 + block)
			rgb[i+1] = byte(77 + block)
			rgb[i+2] = byte(201 + block)
		}
		_, err = db.Exec("INSERT INTO samples(frame_index,source_frame_index,source_timestamp_us,block_index,phase_id,header_id,jpegxs_payload,valid_length,codestream_length,rgb_target) VALUES(0,0,0,?,1,?,?,?,?,?)", block, h.ID, payload, len(payload), len(testCodestream()), rgb)
		if err != nil {
			t.Fatal(err)
		}
	}
	_ = db.Close()
	before, err := HashFile(input)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ConvertRGBTargets(ctx, input, output)
	if err != nil {
		t.Fatal(err)
	}
	if result.Frames != 1 || result.Samples != 30 {
		t.Fatal(result)
	}
	after, err := HashFile(input)
	if err != nil || before != after {
		t.Fatal("source database changed", err)
	}
	source, err := OpenReadStore(input)
	if err != nil {
		t.Fatal(err)
	}
	defer source.DB.Close()
	destination, err := OpenReadStore(output)
	if err != nil {
		t.Fatal(err)
	}
	defer destination.DB.Close()
	var version int
	_ = destination.DB.QueryRow("PRAGMA user_version").Scan(&version)
	if version != SchemaVersion {
		t.Fatal(version)
	}
	var columns int
	if err = destination.DB.QueryRow("SELECT COUNT(*) FROM pragma_table_info('samples') WHERE name='rgb_target'").Scan(&columns); err != nil {
		t.Fatal(err)
	}
	if columns != 0 {
		t.Fatal("RGB target column retained")
	}
	for block := 0; block < BlockCount; block++ {
		var raw []byte
		if err = destination.DB.QueryRow("SELECT raw_bayer_target FROM samples WHERE frame_index=0 AND block_index=?", block).Scan(&raw); err != nil {
			t.Fatal(err)
		}
		if len(raw) != RawBlockBytes {
			t.Fatal("wrong RAW target length")
		}
		for y := 0; y < BlockHeight; y++ {
			for x := 0; x < Width; x++ {
				want := []byte{byte(11 + block), byte(77 + block), byte(201 + block)}[y%2+x%2]
				if raw[y*Width+x] != want {
					t.Fatalf("wrong RGGB target at block %d (%d,%d)", block, x, y)
				}
			}
		}
		full, e := destination.RestoreSample(ctx, 0, block)
		if e != nil || !bytes.Equal(full, testCodestream()) {
			t.Fatal("encoded inputs changed", e)
		}
	}
	var oldStream, newStream bytes.Buffer
	if err = source.ExportStream(ctx, &oldStream, false); err != nil {
		t.Fatal(err)
	}
	if err = destination.ExportStream(ctx, &newStream, false); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(oldStream.Bytes(), newStream.Bytes()) {
		t.Fatal("target conversion changed the BHXS stream")
	}
	var stored string
	_ = destination.DB.QueryRow("SELECT value FROM dataset_info WHERE key='config'").Scan(&stored)
	var converted map[string]any
	if err = json.Unmarshal([]byte(stored), &converted); err != nil {
		t.Fatal(err)
	}
	if converted["Target"] != TargetDescription || converted["Conversion"] == nil {
		t.Fatal("target/provenance metadata missing")
	}
	if _, _, err = OpenStore(input, meta, true); err == nil {
		t.Fatal("silently resumed RGB v2 as RAW v3")
	}
	if _, err = ConvertRGBTargets(ctx, input, output); err == nil {
		t.Fatal("overwrote an existing conversion")
	}
}
