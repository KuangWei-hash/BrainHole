package dataset

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
)

const PackingVersion = 1

// SharedHeader describes how to rebuild the complete, independently decodable
// JPEG XS codestream. ID binds the reconstruction rules and encoder identity.
type SharedHeader struct {
	Version         int    `json:"version"`
	ID              string `json:"id"`
	EncoderIdentity string `json:"encoder_identity"`
	Prefix          []byte `json:"prefix"`
	Suffix          []byte `json:"suffix"`
	LengthOffset    int    `json:"length_offset"`
	PatchLength     bool   `json:"patch_length"`
}

func (h SharedHeader) digest() string {
	b := []byte{'B', 'H', 'X', 'H'}
	b = binary.BigEndian.AppendUint16(b, uint16(h.Version))
	b = binary.BigEndian.AppendUint16(b, uint16(len(h.EncoderIdentity)))
	b = append(b, []byte(h.EncoderIdentity)...)
	b = binary.BigEndian.AppendUint16(b, uint16(len(h.Prefix)))
	b = append(b, h.Prefix...)
	b = binary.BigEndian.AppendUint16(b, uint16(len(h.Suffix)))
	b = append(b, h.Suffix...)
	b = binary.BigEndian.AppendUint16(b, uint16(h.LengthOffset))
	if h.PatchLength {
		b = append(b, 1)
	} else {
		b = append(b, 0)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func (h SharedHeader) Validate() error {
	if h.Version != PackingVersion || len(h.EncoderIdentity) > 4096 || h.ID != h.digest() {
		return fmt.Errorf("shared header version or SHA-256 mismatch")
	}
	if len(h.Prefix) < 8 || len(h.Prefix)+len(h.Suffix) >= MaxEncodedBytes || !bytes.Equal(h.Suffix, []byte{0xff, 0x11}) || h.LengthOffset < 0 || h.LengthOffset+4 > len(h.Prefix) {
		return fmt.Errorf("invalid shared header bounds")
	}
	if !bytes.Equal(h.Prefix[h.LengthOffset:h.LengthOffset+4], make([]byte, 4)) {
		return fmt.Errorf("template Lcod must be normalized to zero")
	}
	return nil
}

// SplitCodestream parses actual marker lengths, rather than dropping a fixed
// byte count. v1 intentionally supports only our RGGB single-slice layout.
func SplitCodestream(full []byte, identity string) (SharedHeader, []byte, error) {
	bad := func(s string) (SharedHeader, []byte, error) {
		return SharedHeader{}, nil, fmt.Errorf("shared-header packing: %s", s)
	}
	if len(identity) > 4096 {
		return bad("encoder identity exceeds 4096 UTF-8 bytes")
	}
	if len(full) < 8 || len(full) > MaxEncodedBytes || !bytes.Equal(full[:2], []byte{0xff, 0x10}) || !bytes.Equal(full[len(full)-2:], []byte{0xff, 0x11}) {
		return bad("invalid complete codestream bounds or SOC/EOC")
	}
	return splitHeader(full, identity, map[byte]bool{})
}

func splitHeader(full []byte, identity string, seen map[byte]bool) (SharedHeader, []byte, error) {
	bad := func(s string) (SharedHeader, []byte, error) {
		return SharedHeader{}, nil, fmt.Errorf("shared-header packing: %s", s)
	}
	offset, lengthOffset := 2, -1
	var patch bool
	for {
		if offset+4 > len(full)-2 || full[offset] != 0xff {
			return bad("truncated header marker")
		}
		marker := full[offset+1]
		size := int(binary.BigEndian.Uint16(full[offset+2 : offset+4]))
		end := offset + 2 + size
		if size < 2 || end > len(full)-2 {
			return bad("invalid marker segment size")
		}
		if marker == 0x20 {
			if size != 4 || binary.BigEndian.Uint16(full[offset+4:end]) != 0 {
				return bad("only slice zero with standard SLH is supported")
			}
			for _, required := range []byte{0x50, 0x12, 0x13, 0x14, 0x18, 0x19} {
				if !seen[required] {
					return bad("missing required Bayer header")
				}
			}
			if end >= len(full)-2 {
				return bad("empty coded payload")
			}
			prefix := append([]byte(nil), full[:end]...)
			clear(prefix[lengthOffset : lengthOffset+4])
			h := SharedHeader{Version: PackingVersion, EncoderIdentity: identity, Prefix: prefix, Suffix: []byte{0xff, 0x11}, LengthOffset: lengthOffset, PatchLength: patch}
			h.ID = h.digest()
			payload := append([]byte(nil), full[end:len(full)-2]...)
			rebuilt, err := h.Restore(payload)
			if err != nil || !bytes.Equal(rebuilt, full) {
				return bad("lossless reassembly check failed")
			}
			return h, payload, nil
		}
		if seen[marker] && marker != 0x15 {
			return bad("duplicate header marker")
		}
		if offset == 2 && marker != 0x50 {
			return bad("CAP must follow SOC")
		}
		switch marker {
		case 0x50, 0x14, 0x15, 0x18:
		case 0x12:
			if size != 26 {
				return bad("unsupported PIH length")
			}
			if binary.BigEndian.Uint16(full[offset+12:offset+14]) != 80 || binary.BigEndian.Uint16(full[offset+14:offset+16]) != 2 || full[offset+20] != 4 || full[offset+25]&15 != 3 {
				return bad("expected 80x2 four-component Star-Tetrix RGGB grid")
			}
			if binary.BigEndian.Uint16(full[offset+16:offset+18]) != 0 {
				return bad("precinct columns unsupported")
			}
			ly := full[offset+26] & 15
			if ly > 1 {
				return bad("unsupported vertical levels")
			}
			precinctRows := (2 + (1 << ly) - 1) / (1 << ly)
			hsl := int(binary.BigEndian.Uint16(full[offset+18 : offset+20]))
			if hsl < precinctRows {
				return bad("multiple slices unsupported")
			}
			lengthOffset = offset + 4
			declared := binary.BigEndian.Uint32(full[lengthOffset : lengthOffset+4])
			if declared != 0 && declared != uint32(len(full)) {
				return bad("Lcod does not match actual length")
			}
			patch = declared != 0
		case 0x13:
			if size != 10 || !bytes.Equal(full[offset+4:end], []byte{8, 0x11, 8, 0x11, 8, 0x11, 8, 0x11}) {
				return bad("expected four full-rate 8-bit component planes")
			}
		case 0x19:
			canonical := []byte{0, 0, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x80, 0, 0x80, 0, 0x80, 0}
			if size != 18 || !bytes.Equal(full[offset+4:end], canonical) {
				return bad("expected canonical RGGB component registration")
			}
		default:
			return bad("unsupported header marker")
		}
		seen[marker] = true
		offset = end
	}
}

func (h SharedHeader) Restore(payload []byte) ([]byte, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	total := len(h.Prefix) + len(payload) + len(h.Suffix)
	if len(payload) < 1 || total > MaxEncodedBytes {
		return nil, fmt.Errorf("payload or reconstructed codestream exceeds bounds")
	}
	full := make([]byte, 0, total)
	full = append(full, h.Prefix...)
	full = append(full, payload...)
	full = append(full, h.Suffix...)
	if h.PatchLength {
		binary.BigEndian.PutUint32(full[h.LengthOffset:h.LengthOffset+4], uint32(total))
	}
	return full, nil
}
