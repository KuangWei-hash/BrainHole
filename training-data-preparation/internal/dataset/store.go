package dataset

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	_ "modernc.org/sqlite"
)

const schema = `
CREATE TABLE IF NOT EXISTS dataset_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shared_header (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 id TEXT NOT NULL UNIQUE,
 definition TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
 frame_index INTEGER NOT NULL CHECK(frame_index>=0),
 source_frame_index INTEGER NOT NULL CHECK(source_frame_index>=0),
 source_timestamp_us INTEGER NOT NULL,
 block_index INTEGER NOT NULL CHECK(block_index BETWEEN 0 AND 29),
 phase_id INTEGER NOT NULL CHECK(phase_id BETWEEN 1 AND 16),
 header_id TEXT NOT NULL REFERENCES shared_header(id),
 jpegxs_payload BLOB NOT NULL,
 valid_length INTEGER NOT NULL CHECK(valid_length BETWEEN 1 AND 640 AND valid_length=length(jpegxs_payload)),
 codestream_length INTEGER NOT NULL CHECK(codestream_length BETWEEN 1 AND 640 AND codestream_length>valid_length),
 raw_bayer_target BLOB NOT NULL CHECK(length(raw_bayer_target)=10240),
 decoded BLOB NOT NULL CHECK(length(decoded)=640),
 PRIMARY KEY(frame_index,block_index),
 CHECK(phase_id=(frame_index % 16)+1)
);
CREATE TABLE IF NOT EXISTS errors (
 id INTEGER PRIMARY KEY, frame_index INTEGER NOT NULL, block_index INTEGER NOT NULL,
 message TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);`

type Store struct {
	DB              *sql.DB
	EncoderIdentity string
	// Version is the SQLite user_version the database was opened with.
	// It selects which optional columns the samples table carries.
	Version int
}

type Sample struct {
	Frame             int64
	SourceFrame       int
	TimestampUS       int64
	Block, Phase      int
	Encoded, RawBayer []byte
	// Decoded is the low-resolution mosaic the encoded codestream
	// reconstructs — the future second model input. Schema v4 requires
	// it to be exactly DecodedBytes long.
	Decoded []byte
}

func OpenStore(path, metadata string, resume bool) (*Store, int64, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, 0, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	created := err == nil
	if err == nil {
		_ = f.Close()
	} else if !os.IsExist(err) || !resume {
		return nil, 0, fmt.Errorf("output exists or cannot be created (use -resume for matching datasets): %w", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, 0, err
	}
	db.SetMaxOpenConns(1)
	fail := func(err error) (*Store, int64, error) { _ = db.Close(); return nil, 0, err }
	var version int
	if err = db.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		return fail(err)
	}
	if !created && version != SchemaVersion {
		return fail(fmt.Errorf("SQLite schema version %d cannot be resumed; schema v%d requires a new or explicitly migrated database", version, SchemaVersion))
	}
	if _, err = db.Exec("PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;" + schema + fmt.Sprintf("PRAGMA user_version=%d;", SchemaVersion)); err != nil {
		return fail(err)
	}
	var prior string
	err = db.QueryRow("SELECT value FROM dataset_info WHERE key='config'").Scan(&prior)
	if err == sql.ErrNoRows {
		var count int
		if err = db.QueryRow("SELECT COUNT(*) FROM samples").Scan(&count); err != nil {
			return fail(err)
		}
		if count != 0 {
			return fail(fmt.Errorf("database has samples without configuration"))
		}
		_, err = db.Exec("INSERT INTO dataset_info(key,value) VALUES('config',?)", metadata)
	} else if err == nil && prior != metadata {
		return fail(fmt.Errorf("resume configuration/source mismatch; use a new output database"))
	}
	if err != nil {
		return fail(err)
	}
	var frames, maxFrame int64
	if err = db.QueryRow("SELECT COUNT(DISTINCT frame_index), COALESCE(MAX(frame_index),-1) FROM samples").Scan(&frames, &maxFrame); err != nil {
		return fail(err)
	}
	var bad int
	if err = db.QueryRow("SELECT COUNT(*) FROM (SELECT frame_index FROM samples GROUP BY frame_index HAVING COUNT(*) != 30)").Scan(&bad); err != nil {
		return fail(err)
	}
	if bad != 0 || frames != maxFrame+1 {
		return fail(fmt.Errorf("database contains incomplete or nonconsecutive frames"))
	}
	var meta struct{ Encoder EncoderConfig }
	if err = json.Unmarshal([]byte(metadata), &meta); err != nil {
		return fail(fmt.Errorf("metadata JSON: %w", err))
	}
	s := &Store{DB: db, EncoderIdentity: meta.Encoder.Identity, Version: SchemaVersion}
	if frames > 0 {
		h, e := s.Header()
		if e != nil {
			return fail(e)
		}
		if h.EncoderIdentity != s.EncoderIdentity {
			return fail(fmt.Errorf("shared header/metadata encoder identity mismatch"))
		}
	}
	return s, maxFrame + 1, nil
}

func (s *Store) WriteFrame(ctx context.Context, rows []Sample) error {
	if len(rows) != BlockCount {
		return fmt.Errorf("frame requires 30 samples")
	}
	for i, r := range rows {
		if r.Block != i || r.Frame != rows[0].Frame || r.Phase != Phase(r.Frame) {
			return fmt.Errorf("invalid frame pairing")
		}
	}
	var header SharedHeader
	payloads := make([][]byte, len(rows))
	for i, r := range rows {
		h, p, err := SplitCodestream(r.Encoded, s.EncoderIdentity)
		if err != nil {
			return fmt.Errorf("block %d: %w", i, err)
		}
		if i == 0 {
			header = h
		} else if header.ID != h.ID {
			return fmt.Errorf("shared header changed at block %d; start a new dataset/configuration", i)
		}
		payloads[i] = p
	}
	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	definition, err := json.Marshal(header)
	if err != nil {
		return err
	}
	var existingID, existingDefinition string
	err = tx.QueryRowContext(ctx, "SELECT id,definition FROM shared_header WHERE singleton=1").Scan(&existingID, &existingDefinition)
	if err == sql.ErrNoRows {
		_, err = tx.ExecContext(ctx, "INSERT INTO shared_header(singleton,id,definition) VALUES(1,?,?)", header.ID, string(definition))
	} else if err == nil {
		var existing SharedHeader
		if e := json.Unmarshal([]byte(existingDefinition), &existing); e != nil {
			return e
		}
		if e := existing.Validate(); e != nil {
			return e
		}
		if existingID != header.ID || existing.ID != header.ID {
			return fmt.Errorf("shared header changed across frames/resume; use a new dataset/configuration")
		}
	}
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx, `INSERT INTO samples(frame_index,source_frame_index,source_timestamp_us,block_index,phase_id,header_id,jpegxs_payload,valid_length,codestream_length,raw_bayer_target,decoded) VALUES(?,?,?,?,?,?,?,?,?,?,?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for i, r := range rows {
		if len(r.Decoded) != DecodedBytes {
			return fmt.Errorf("block %d: decoded mosaic must be %d bytes, got %d", i, DecodedBytes, len(r.Decoded))
		}
		if _, err = stmt.ExecContext(ctx, r.Frame, r.SourceFrame, r.TimestampUS, r.Block, r.Phase, header.ID, payloads[i], len(payloads[i]), len(r.Encoded), r.RawBayer, r.Decoded); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *Store) Header() (SharedHeader, error) {
	var h SharedHeader
	var id, definition string
	if err := s.DB.QueryRow("SELECT id,definition FROM shared_header WHERE singleton=1").Scan(&id, &definition); err != nil {
		return h, err
	}
	if err := json.Unmarshal([]byte(definition), &h); err != nil {
		return h, err
	}
	if err := h.Validate(); err != nil {
		return h, err
	}
	if id != h.ID {
		return h, fmt.Errorf("stored header ID mismatch")
	}
	return h, nil
}

// RestoreSample is for inspection/decoder interoperability; AI inputs use payload only.
func (s *Store) RestoreSample(ctx context.Context, frame int64, block int) ([]byte, error) {
	h, err := s.Header()
	if err != nil {
		return nil, err
	}
	var payload []byte
	var id string
	var size int
	if err = s.DB.QueryRowContext(ctx, "SELECT header_id,jpegxs_payload,codestream_length FROM samples WHERE frame_index=? AND block_index=?", frame, block).Scan(&id, &payload, &size); err != nil {
		return nil, err
	}
	if id != h.ID {
		return nil, fmt.Errorf("sample/header ID mismatch")
	}
	full, err := h.Restore(payload)
	if err != nil {
		return nil, err
	}
	if len(full) != size {
		return nil, fmt.Errorf("sample codestream length mismatch")
	}
	return full, nil
}

// RestoreDecoded returns the low-resolution decoded mosaic stored for one
// sample — the future second model input. Schema v4 stores it next to the
// encoded payload.
func (s *Store) RestoreDecoded(ctx context.Context, frame int64, block int) ([]byte, error) {
	var decoded []byte
	if err := s.DB.QueryRowContext(ctx, "SELECT decoded FROM samples WHERE frame_index=? AND block_index=?", frame, block).Scan(&decoded); err != nil {
		return nil, err
	}
	if len(decoded) != DecodedBytes {
		return nil, fmt.Errorf("stored decoded mosaic is %d bytes, want %d", len(decoded), DecodedBytes)
	}
	return decoded, nil
}

func (s *Store) LogError(frame int64, block int, err error) error {
	_, e := s.DB.Exec("INSERT INTO errors(frame_index,block_index,message) VALUES(?,?,?)", frame, block, err.Error())
	return e
}
