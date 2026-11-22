package dataset

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

// MigrateToV4 upgrades a schema v3 dataset to schema v4.
//
// Schema v4 adds the decoded low-resolution mosaic to every sample: the
// 160x4 RGGB block the stored JPEG XS payload reconstructs, which becomes
// the second model input. The migration decodes every payload with the
// in-process Go JPEG XS decoder — the same decoder that reproduces the
// upstream Rust decoder byte for byte — so the new column needs no
// external toolchain and no re-encoding.
//
// The encoded payloads, the shared header, the RAW8 targets, the timing
// and the phase assignment are copied unchanged. The source database is
// never modified; the result is written to a new file.
func MigrateToV4(ctx context.Context, input, output string) (result ConversionResult, err error) {
	absInput, err := filepath.Abs(input)
	if err != nil {
		return result, err
	}
	absOutput, err := filepath.Abs(output)
	if err != nil {
		return result, err
	}
	if strings.EqualFold(absInput, absOutput) {
		return result, fmt.Errorf("migration requires a different output file")
	}
	source, err := OpenReadStore(absInput)
	if err != nil {
		return result, err
	}
	defer source.DB.Close()
	if source.Version != 3 {
		return result, fmt.Errorf("migration requires schema v3; got v%d", source.Version)
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
	meta["Migration"] = map[string]any{
		"source_database":          absInput,
		"source_database_sha256":   sourceHash,
		"source_schema_version":    3,
		"method":                   "decode every stored JPEG XS payload with the in-process Go decoder (oxideav-jpegxs rev 3484a04 parity); encoded inputs, shared header, RAW8 targets, timing and phases preserved",
		"decoded_column":           "decoded",
		"decoded_bytes_per_sample": DecodedBytes,
		"decoded_description":      "low-resolution decoded mosaic, 160x4 RGGB RAW8 row-major, decoded from the stored payload",
	}
	b, err := json.Marshal(meta)
	if err != nil {
		return result, err
	}
	destination, _, err := OpenStore(absOutput, string(b), false)
	if err != nil {
		return result, err
	}
	defer destination.DB.Close()

	rows, err := source.DB.QueryContext(ctx, `SELECT frame_index,source_frame_index,source_timestamp_us,block_index,phase_id,header_id,jpegxs_payload,codestream_length,raw_bayer_target FROM samples ORDER BY frame_index,block_index`)
	if err != nil {
		return result, err
	}
	defer rows.Close()
	batch := make([]Sample, 0, BlockCount)
	for rows.Next() {
		var r Sample
		var id string
		var payload, target []byte
		var size int
		if err = rows.Scan(&r.Frame, &r.SourceFrame, &r.TimestampUS, &r.Block, &r.Phase, &id, &payload, &size, &target); err != nil {
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
		if len(target) != RawBlockBytes {
			return result, fmt.Errorf("source RAW8 target is %d bytes, want %d", len(target), RawBlockBytes)
		}
		r.RawBayer = target
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
