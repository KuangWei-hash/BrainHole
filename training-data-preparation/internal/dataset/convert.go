package dataset

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

type ConversionResult struct{ Frames, Samples int64 }

// ConvertRGBTargets writes a new v4 database from a completed v2 dataset.
// Encoded inputs are preserved; no video decoding or JPEG XS re-encoding occurs.
//
// The v4 schema also stores the decoded low-resolution mosaic of every
// sample; it is produced by decoding the preserved payload with the
// in-process Go JPEG XS decoder.
func ConvertRGBTargets(ctx context.Context, input, output string) (result ConversionResult, err error) {
	absInput, err := filepath.Abs(input)
	if err != nil {
		return result, err
	}
	absOutput, err := filepath.Abs(output)
	if err != nil {
		return result, err
	}
	if strings.EqualFold(absInput, absOutput) {
		return result, fmt.Errorf("conversion requires a different output file")
	}
	source, err := OpenReadStore(absInput)
	if err != nil {
		return result, err
	}
	defer source.DB.Close()
	var version int
	if err = source.DB.QueryRowContext(ctx, "PRAGMA user_version").Scan(&version); err != nil {
		return result, err
	}
	if version != 2 {
		return result, fmt.Errorf("conversion requires RGB-target schema v2; got %d", version)
	}
	var status, metadata string
	if err = source.DB.QueryRowContext(ctx, "SELECT value FROM dataset_info WHERE key='status'").Scan(&status); err != nil {
		return result, err
	}
	if status != "complete" {
		return result, fmt.Errorf("source dataset must be complete")
	}
	if err = source.DB.QueryRowContext(ctx, "SELECT value FROM dataset_info WHERE key='config'").Scan(&metadata); err != nil {
		return result, err
	}
	header, err := source.Header()
	if err != nil {
		return result, err
	}
	var meta map[string]any
	if err = json.Unmarshal([]byte(metadata), &meta); err != nil {
		return result, err
	}
	sourceHash, err := HashFile(absInput)
	if err != nil {
		return result, err
	}
	meta["schema_version"] = SchemaVersion
	meta["Target"] = TargetDescription
	meta["Conversion"] = map[string]any{"source_database": absInput, "source_database_sha256": sourceHash, "source_schema_version": 2, "method": "RGB8 to full-resolution RAW8 RGGB by channel selection; preserve encoded inputs; decode every preserved payload with the in-process Go decoder to fill the schema v4 decoded column"}
	b, err := json.Marshal(meta)
	if err != nil {
		return result, err
	}
	destination, _, err := OpenStore(absOutput, string(b), false)
	if err != nil {
		return result, err
	}
	defer destination.DB.Close()
	rows, err := source.DB.QueryContext(ctx, `SELECT frame_index,source_frame_index,source_timestamp_us,block_index,phase_id,header_id,jpegxs_payload,codestream_length,rgb_target FROM samples ORDER BY frame_index,block_index`)
	if err != nil {
		return result, err
	}
	defer rows.Close()
	batch := make([]Sample, 0, BlockCount)
	for rows.Next() {
		var r Sample
		var id string
		var payload, rgb []byte
		var size int
		if err = rows.Scan(&r.Frame, &r.SourceFrame, &r.TimestampUS, &r.Block, &r.Phase, &id, &payload, &size, &rgb); err != nil {
			return result, err
		}
		if id != header.ID {
			return result, fmt.Errorf("source shared header mismatch")
		}
		if r.Frame != result.Frames || r.Block != len(batch) {
			return result, fmt.Errorf("source frame/block sequence is not complete")
		}
		r.Encoded, err = header.Restore(payload)
		if err != nil {
			return result, err
		}
		if len(r.Encoded) != size {
			return result, fmt.Errorf("source codestream size mismatch")
		}
		r.RawBayer, err = Bayer(rgb)
		if err != nil {
			return result, err
		}
		r.Decoded, err = DecodeBlock(r.Encoded)
		if err != nil {
			return result, fmt.Errorf("frame %d block %d: decode: %w", r.Frame, r.Block, err)
		}
		batch = append(batch, r)
		if len(batch) == BlockCount {
			if err = destination.WriteFrame(ctx, batch); err != nil {
				return result, err
			}
			result.Frames++
			result.Samples += BlockCount
			batch = batch[:0]
		}
	}
	if err = rows.Err(); err != nil {
		return result, err
	}
	if len(batch) != 0 || result.Frames == 0 {
		return result, fmt.Errorf("source ends with an incomplete or empty dataset")
	}
	_, err = destination.DB.ExecContext(ctx, "INSERT OR REPLACE INTO dataset_info(key,value) VALUES('status','complete')")
	return result, err
}
