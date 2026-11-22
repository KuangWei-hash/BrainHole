package dataset

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"io"
	"path/filepath"
	"testing"
)

func TestPackingMarkerParsingLengthModesAndTampering(t *testing.T) {
	original := testCodestream()
	h, payload, err := SplitCodestream(original, "codec Q8")
	if err != nil {
		t.Fatal(err)
	}
	if len(h.Prefix) != 100 || len(h.Suffix) != 2 || len(payload) != 44 {
		t.Fatal("unexpected fixture layout")
	}
	restored, err := h.Restore(payload)
	if err != nil || !bytes.Equal(restored, original) {
		t.Fatal("roundtrip mismatch", err)
	}
	for n := 0; n < len(original); n++ {
		if _, _, err := SplitCodestream(original[:n], "codec Q8"); err == nil {
			t.Fatalf("accepted truncation %d", n)
		}
	}
	declared := append([]byte(nil), original...)
	binary.BigEndian.PutUint32(declared[h.LengthOffset:h.LengthOffset+4], uint32(len(declared)))
	h2, p2, err := SplitCodestream(declared, "codec Q8")
	if err != nil {
		t.Fatal(err)
	}
	if !h2.PatchLength || h.ID == h2.ID {
		t.Fatal("length signaling not identified")
	}
	again, _ := h2.Restore(p2)
	if !bytes.Equal(again, declared) {
		t.Fatal("nonzero Lcod was not restored")
	}
	// Same syntax with a different coded size shares the normalized header.
	larger := append(append(append([]byte(nil), declared[:len(declared)-2]...), 0), 0xff, 0x11)
	binary.BigEndian.PutUint32(larger[h.LengthOffset:h.LengthOffset+4], uint32(len(larger)))
	h3, p3, err := SplitCodestream(larger, "codec Q8")
	if err != nil {
		t.Fatal(err)
	}
	if h3.ID != h2.ID {
		t.Fatal("per-block Lcod prevented header sharing")
	}
	again, _ = h3.Restore(p3)
	if !bytes.Equal(again, larger) {
		t.Fatal("dynamic Lcod reconstruction mismatch")
	}
	broken := append([]byte(nil), declared...)
	binary.BigEndian.PutUint32(broken[h.LengthOffset:h.LengthOffset+4], 1)
	if _, _, err := SplitCodestream(broken, "codec Q8"); err == nil {
		t.Fatal("accepted invalid Lcod")
	}
	h.Prefix[5] ^= 1
	if _, err = h.Restore(payload); err == nil {
		t.Fatal("accepted header corruption")
	}
	wrongCFA := append([]byte(nil), original...)
	crg := bytes.Index(wrongCFA, []byte{0xff, 0x19})
	wrongCFA[crg+4] = 1
	if _, _, err := SplitCodestream(wrongCFA, "codec Q8"); err == nil {
		t.Fatal("accepted non-RGGB registration")
	}
}

func TestStoreSharedHeaderRollbackDriftAndLegacy(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "shared.sqlite")
	s, _, err := OpenStore(path, `{"schema":2}`, false)
	if err != nil {
		t.Fatal(err)
	}
	defer s.DB.Close()
	bad := testRows(0)
	bad[20].RawBayer = []byte{1}
	if err = s.WriteFrame(ctx, bad); err == nil {
		t.Fatal("accepted bad RGB")
	}
	var count int
	_ = s.DB.QueryRow("SELECT COUNT(*) FROM shared_header").Scan(&count)
	if count != 0 {
		t.Fatal("header survived failed frame transaction")
	}
	if err = s.WriteFrame(ctx, testRows(0)); err != nil {
		t.Fatal(err)
	}
	full, err := s.RestoreSample(ctx, 0, 12)
	if err != nil || !bytes.Equal(full, testCodestream()) {
		t.Fatal("stored reconstruction mismatch", err)
	}
	changed := testRows(1)
	for i := range changed {
		wgt := bytes.Index(changed[i].Encoded, []byte{0xff, 0x14})
		changed[i].Encoded[wgt+4] = 1
	}
	if err = s.WriteFrame(ctx, changed); err == nil {
		t.Fatal("accepted shared header drift")
	}
	_ = s.DB.QueryRow("SELECT COUNT(*) FROM samples").Scan(&count)
	if count != 30 {
		t.Fatal("header drift partially committed")
	}
	legacyPath := filepath.Join(t.TempDir(), "legacy.sqlite")
	db, err := sql.Open("sqlite", legacyPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = db.Exec("CREATE TABLE samples(jpegxs_encoded BLOB)"); err != nil {
		t.Fatal(err)
	}
	_ = db.Close()
	if _, _, err = OpenStore(legacyPath, `{"schema":2}`, true); err == nil {
		t.Fatal("accepted legacy schema")
	}
	db, err = sql.Open("sqlite", legacyPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	_ = db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE name='shared_header'").Scan(&count)
	if count != 0 {
		t.Fatal("legacy database was modified")
	}
}

func TestStreamFramesPhasesTruncationAndBounds(t *testing.T) {
	h, p, err := SplitCodestream(testCodestream(), "codec Q8")
	if err != nil {
		t.Fatal(err)
	}
	var data bytes.Buffer
	w, err := NewStreamWriter(&data, h)
	if err != nil {
		t.Fatal(err)
	}
	for frame := int64(0); frame < 17; frame++ {
		for block := 0; block < BlockCount; block++ {
			if err = w.WriteBlock(frame, block, p); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err = w.Finish(); err != nil {
		t.Fatal(err)
	}
	r, err := NewStreamReader(bytes.NewReader(data.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 17*BlockCount; i++ {
		item, err := r.Next()
		if err != nil {
			t.Fatal(err)
		}
		if item.Frame != int64(i/30) || item.Block != i%30 || item.Phase != Phase(item.Frame) || !bytes.Equal(item.Payload, p) {
			t.Fatal("frame/block/phase mismatch")
		}
	}
	if _, err = r.Next(); err != io.EOF {
		t.Fatal("expected clean EOF", err)
	}
	truncated, err := NewStreamReader(bytes.NewReader(data.Bytes()[:data.Len()-1]))
	if err != nil {
		t.Fatal(err)
	}
	for {
		_, err = truncated.Next()
		if err != nil {
			break
		}
	}
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal("truncation not reported", err)
	}
	var first bytes.Buffer
	writer, _ := NewStreamWriter(&first, h)
	preambleLen := first.Len()
	if err = writer.WriteBlock(0, 1, p); err == nil {
		t.Fatal("accepted skipped block")
	}
	if err = writer.WriteBlock(0, 0, p); err != nil {
		t.Fatal(err)
	}
	if err = writer.Finish(); err == nil {
		t.Fatal("accepted incomplete frame")
	}
	broken := append([]byte(nil), first.Bytes()...)
	binary.BigEndian.PutUint16(broken[preambleLen+8:preambleLen+10], 641)
	reader, err := NewStreamReader(bytes.NewReader(broken))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = reader.Next(); err == nil {
		t.Fatal("accepted oversized payload")
	}
	broken = append([]byte(nil), first.Bytes()...)
	broken[4] = 99
	if _, err = NewStreamReader(bytes.NewReader(broken)); err == nil {
		t.Fatal("accepted unknown stream version")
	}
}

func TestStreamDeliversFirstBlockBeforeFrameComplete(t *testing.T) {
	h, p, err := SplitCodestream(testCodestream(), "codec Q8")
	if err != nil {
		t.Fatal(err)
	}
	pr, pw := io.Pipe()
	defer pr.Close()
	release := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		w, e := NewStreamWriter(pw, h)
		if e == nil {
			e = w.WriteBlock(0, 0, p)
		}
		if e == nil {
			<-release
			for block := 1; block < BlockCount; block++ {
				if e = w.WriteBlock(0, block, p); e != nil {
					break
				}
			}
		}
		if e == nil {
			e = w.Finish()
		}
		_ = pw.CloseWithError(e)
		done <- e
	}()
	r, err := NewStreamReader(pr)
	if err != nil {
		close(release)
		t.Fatal(err)
	}
	item, err := r.Next()
	close(release)
	if err != nil || item.Block != 0 {
		t.Fatal("first block unavailable", err)
	}
	for i := 1; i < BlockCount; i++ {
		if _, err = r.Next(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = r.Next(); err != io.EOF {
		t.Fatal(err)
	}
	if err = <-done; err != nil {
		t.Fatal(err)
	}
}

func TestThirtyFPSTimestampRoundingDoesNotDuplicateFrames(t *testing.T) {
	frames := []SourceFrame{{0, 33000}, {33000, 67000}, {67000, 100000}, {100000, 133333}}
	for i := int64(0); i < 4; i++ {
		source, ok := SourceAt(frames, i)
		if !ok || source != int(i) {
			t.Fatalf("30 FPS output %d mapped to %d", i, source)
		}
	}
}
