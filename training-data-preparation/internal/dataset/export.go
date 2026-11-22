package dataset

import (
	"context"
	"database/sql"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strings"
)

// OpenReadStore opens an existing dataset for reading. Schema v2, v3 and
// v4 databases are accepted; only v4 carries the decoded mosaic.
func OpenReadStore(path string) (*Store, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	uriPath := filepath.ToSlash(abs)
	if !strings.HasPrefix(uriPath, "/") {
		uriPath = "/" + uriPath
	}
	u := url.URL{Scheme: "file", Path: uriPath, RawQuery: "mode=ro"}
	db, err := sql.Open("sqlite", u.String())
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	var version int
	if err = db.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		db.Close()
		return nil, err
	}
	if version != 2 && version != 3 && version != SchemaVersion {
		db.Close()
		return nil, fmt.Errorf("expected SQLite schema v2, v3 or v%d, got %d", SchemaVersion, version)
	}
	s := &Store{DB: db, Version: version}
	h, err := s.Header()
	if err != nil {
		db.Close()
		return nil, err
	}
	s.EncoderIdentity = h.EncoderIdentity
	return s, nil
}

// ExportStream writes the dataset as a BHXS stream.
//
// With includeDecoded set the stream carries the low-resolution decoded
// mosaic of every block next to its payload — the second model input —
// which requires a schema v4 source. Without it the stream is the
// bandwidth-lean payload-only format.
func (s *Store) ExportStream(ctx context.Context, w io.Writer, includeDecoded bool) error {
	h, err := s.Header()
	if err != nil {
		return err
	}
	if includeDecoded && s.Version < 4 {
		return fmt.Errorf("schema v%d has no decoded mosaic column; migrate to schema v4 first", s.Version)
	}
	writer, err := NewStreamWriterV2(w, h, includeDecoded)
	if err != nil {
		return err
	}
	query := "SELECT frame_index,block_index,header_id,jpegxs_payload,codestream_length FROM samples ORDER BY frame_index,block_index"
	if includeDecoded {
		query = "SELECT frame_index,block_index,header_id,jpegxs_payload,codestream_length,decoded FROM samples ORDER BY frame_index,block_index"
	}
	rows, err := s.DB.QueryContext(ctx, query)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var frame int64
		var block, size int
		var id string
		var payload []byte
		var decoded []byte
		if includeDecoded {
			if err = rows.Scan(&frame, &block, &id, &payload, &size, &decoded); err != nil {
				return err
			}
		} else if err = rows.Scan(&frame, &block, &id, &payload, &size); err != nil {
			return err
		}
		if id != h.ID || size != len(h.Prefix)+len(payload)+len(h.Suffix) {
			return fmt.Errorf("invalid sample/header reference or size")
		}
		if includeDecoded {
			if len(decoded) != DecodedBytes {
				return fmt.Errorf("sample %d/%d decoded mosaic is %d bytes, want %d", frame, block, len(decoded), DecodedBytes)
			}
			if err = writer.WriteBlockDecoded(frame, block, payload, decoded); err != nil {
				return err
			}
			continue
		}
		if err = writer.WriteBlock(frame, block, payload); err != nil {
			return err
		}
	}
	if err = rows.Err(); err != nil {
		return err
	}
	return writer.Finish()
}
