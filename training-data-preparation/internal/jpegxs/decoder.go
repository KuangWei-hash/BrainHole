package jpegxs

import (
	"fmt"
)

// JPEG XS decoding — ISO/IEC 21122-1:2022.
//
// Port of the decoder side of the upstream Rust implementation
// (oxideav-jpegxs 0.0.9, rev 3484a041d6d8e893683c1920cb3f68fe6fd3b8b8):
// the codestream markers (Annex A), the slice walk (Annex B), the entropy
// decoder (Annex C), the inverse dequantiser (Annex D), the inverse 5/3
// wavelet (Annex E), the inverse colour transform (Annex F) and the output
// scaling (Annex G).
//
// Supported subset — the same restriction the Go encoder carries:
//
//   - 8-bit components
//   - Cpih ∈ {0, 1, 3}
//   - NL,x = 1, NL,y = 0 (single-level horizontal DWT)
//   - one slice, one precinct column (Cw = 0)
//   - no CWD (decomposition suppression), no NLT, no refinement
//
// Anything outside the subset is reported as an error rather than decoded
// on a best-effort basis, matching the encoder's contract.

// decodeComponents decodes a bare JPEG XS codestream into its component
// planes.
func decodeComponents(buf []byte) (*Components, error) {
	cs, err := parseCodestream(buf)
	if err != nil {
		return nil, err
	}
	return decodePicture(cs)
}

// DecodeRGGB8Block decodes a codestream produced by EncodeRGGB8Block back
// into its 160 × 4 RAW8 RGGB mosaic.
//
// The result is the lossy reconstruction: at Q = 6 the decoded mosaic
// differs from the original block, which is exactly what the second model
// input needs.
func DecodeRGGB8Block(coded []byte) ([]byte, error) {
	comps, err := decodeComponents(coded)
	if err != nil {
		return nil, err
	}
	if comps.Cpih != 3 || len(comps.Planes) != 4 {
		return nil, fmt.Errorf("jpegxs decode RGGB8: expected a Star-Tetrix picture with 4 components, got Cpih=%d with %d planes",
			comps.Cpih, len(comps.Planes))
	}
	planeW := int(comps.Width)
	planeH := int(comps.Height)
	out := make([]byte, planeW*2*planeH*2)
	for y := 0; y < planeH; y++ {
		for x := 0; x < planeW; x++ {
			base := (2 * y) * (2 * planeW)
			out[base+2*x] = comps.Planes[0][y*planeW+x]
			out[base+2*x+1] = comps.Planes[1][y*planeW+x]
			base = (2*y + 1) * (2 * planeW)
			out[base+2*x] = comps.Planes[2][y*planeW+x]
			out[base+2*x+1] = comps.Planes[3][y*planeW+x]
		}
	}
	return out, nil
}

// pictureHeader is the parsed PIH marker (Annex A.4.4).
type pictureHeader struct {
	lcod uint32
	ppih uint16
	plev uint16
	wf   uint16
	hf   uint16
	cw   uint16
	hsl  uint16
	nc   uint8
	ng   uint8
	ss   uint8
	bw   uint8
	fq   uint8
	br   uint8
	fslc uint8
	ppoc uint8
	cpih uint8
	nlx  uint8
	nly  uint8
	lh   uint8
	rl   uint8
	qpih uint8
	fs   uint8
	rm   uint8
}

// component is one entry of the CDT marker (Annex A.4.5).
type component struct {
	bitDepth uint8
	sx       uint8
	sy       uint8
}

// ctsMarker is the parsed CTS marker (Annex A.4.8).
type ctsMarker struct {
	cf uint8
	e1 uint8
	e2 uint8
}

// codestream is the parsed header set plus the entropy-coded slice data.
type codestream struct {
	pih    pictureHeader
	cdt    []component
	wgt    []byte
	cts    *ctsMarker
	crg    []crgEntry
	slices []sliceData
	hasCRG bool
}

// sliceData is one slice: its header index and its entropy-coded bytes.
type sliceData struct {
	// yslh is the slice index (Annex A.4.12, Table A.25).
	yslh uint16
	data []byte
}

// crgEntry is one component registration (Annex A.4.9).
type crgEntry struct {
	x uint16
	y uint16
}

// parseCodestream parses the bare codestream: SOC, the marker segments and
// the entropy-coded slice data.
func parseCodestream(buf []byte) (*codestream, error) {
	cs := &codestream{}
	if len(buf) < 2 || buf[0] != 0xff || buf[1] != 0x10 {
		return nil, fmt.Errorf("jpegxs decode: stream does not start with SOC (ff 10)")
	}
	pos := 2
	readMarker := func() (byte, error) {
		if pos+1 >= len(buf) {
			return 0, fmt.Errorf("jpegxs decode: truncated marker at offset %d", pos)
		}
		if buf[pos] != 0xff {
			return 0, fmt.Errorf("jpegxs decode: expected a marker at offset %d, got 0x%02x", pos, buf[pos])
		}
		return buf[pos+1], nil
	}
	for {
		code, err := readMarker()
		if err != nil {
			return nil, err
		}
		pos += 2
		switch code {
		case 0x11: // EOC
			return cs, nil
		case 0x50: // CAP
			body, next, err := readSegment(buf, pos)
			if err != nil {
				return nil, err
			}
			pos = next
			_ = body
		case 0x12: // PIH
			body, next, err := readSegment(buf, pos)
			if err != nil {
				return nil, err
			}
			pos = next
			if err := cs.parsePIH(body); err != nil {
				return nil, err
			}
		case 0x13: // CDT
			body, next, err := readSegment(buf, pos)
			if err != nil {
				return nil, err
			}
			pos = next
			if err := cs.parseCDT(body); err != nil {
				return nil, err
			}
		case 0x14: // WGT
			body, next, err := readSegment(buf, pos)
			if err != nil {
				return nil, err
			}
			pos = next
			cs.wgt = body
		case 0x18: // CTS
			body, next, err := readSegment(buf, pos)
			if err != nil {
				return nil, err
			}
			pos = next
			if len(body) != 2 {
				return nil, fmt.Errorf("jpegxs decode: CTS body must be 2 bytes, got %d", len(body))
			}
			reserved := (body[0] >> 4) & 0x0f
			if reserved != 0 {
				return nil, fmt.Errorf("jpegxs decode: CTS reserved nibble must be 0 per A.4.8, got %d", reserved)
			}
			cf := body[0] & 0x0f
			if cf != 0 && cf != 3 {
				return nil, fmt.Errorf("jpegxs decode: CTS Cf=%d reserved for ISO/IEC use (Table A.20)", cf)
			}
			e1 := (body[1] >> 4) & 0x0f
			e2 := body[1] & 0x0f
			if e1 > 3 || e2 > 3 {
				return nil, fmt.Errorf("jpegxs decode: CTS e1/e2 exceed the Table A.19 cap of 3")
			}
			cs.cts = &ctsMarker{cf: cf, e1: e1, e2: e2}
		case 0x19: // CRG
			body, next, err := readSegment(buf, pos)
			if err != nil {
				return nil, err
			}
			pos = next
			if len(body) != 4*int(cs.pih.nc) {
				return nil, fmt.Errorf("jpegxs decode: CRG body must be 4*Nc = %d bytes, got %d", 4*int(cs.pih.nc), len(body))
			}
			for c := 0; c < int(cs.pih.nc); c++ {
				off := 4 * c
				cs.crg = append(cs.crg, crgEntry{
					x: uint16(body[off])<<8 | uint16(body[off+1]),
					y: uint16(body[off+2])<<8 | uint16(body[off+3]),
				})
			}
			cs.hasCRG = true
		case 0x20: // SLH — the slice header; the entropy data follows.
			body, next, err := readSegment(buf, pos)
			if err != nil {
				return nil, err
			}
			if len(body) != 2 {
				return nil, fmt.Errorf("jpegxs decode: SLH body must be 2 bytes, got %d", len(body))
			}
			yslh := uint16(body[0])<<8 | uint16(body[1])
			pos = next
			// Entropy bytes may contain ff20/ff11. Walk the declared precinct
			// lengths instead of mistaking those bytes for SLH/EOC markers.
			p := &cs.pih
			if p.nlx != 1 || p.nly != 0 || p.cw != 0 {
				return nil, fmt.Errorf("jpegxs decode: slice framing requires NL,x=1 NL,y=0 Cw=0")
			}
			startRow := int(yslh) * int(p.hsl)
			if p.hsl == 0 || startRow >= int(p.hf) {
				return nil, fmt.Errorf("jpegxs decode: invalid slice index or height")
			}
			rows := min(int(p.hsl), int(p.hf)-startRow)
			// Table C.1: 24-bit Lprc, Q/R bytes, two D bits per band;
			// the supported NL=1/0 subset has two bands per component.
			headerBytes := (40 + 4*int(p.nc) + 7) / 8
			end := pos
			for row := 0; row < rows; row++ {
				if end+headerBytes > len(buf)-2 {
					return nil, fmt.Errorf("jpegxs decode: truncated precinct header in slice %d", yslh)
				}
				length := int(buf[end])<<16 | int(buf[end+1])<<8 | int(buf[end+2])
				if length < 1 || length > (1<<20)-1 || length > len(buf)-2-end-headerBytes {
					return nil, fmt.Errorf("jpegxs decode: invalid precinct length in slice %d", yslh)
				}
				end += headerBytes + length
			}
			cs.slices = append(cs.slices, sliceData{yslh: yslh, data: buf[pos:end]})
			pos = end
		default:
			return nil, fmt.Errorf("jpegxs decode: unsupported marker 0xff%02x at offset %d", code, pos-2)
		}
	}
}

// readSegment reads a marker segment body and returns it with the offset
// just past the segment.
func readSegment(buf []byte, pos int) ([]byte, int, error) {
	if pos+2 > len(buf) {
		return nil, 0, fmt.Errorf("jpegxs decode: truncated marker segment length at offset %d", pos)
	}
	length := int(buf[pos])<<8 | int(buf[pos+1])
	if length < 2 {
		return nil, 0, fmt.Errorf("jpegxs decode: marker segment length %d is invalid", length)
	}
	bodyLen := length - 2
	if pos+2+bodyLen > len(buf) {
		return nil, 0, fmt.Errorf("jpegxs decode: marker segment body of %d bytes exceeds the buffer", bodyLen)
	}
	return buf[pos+2 : pos+2+bodyLen], pos + 2 + bodyLen, nil
}

// parsePIH parses the picture header body (24 bytes).
func (cs *codestream) parsePIH(body []byte) error {
	if len(body) != 24 {
		return fmt.Errorf("jpegxs decode: PIH body must be 24 bytes, got %d", len(body))
	}
	be16 := func(b []byte) uint16 { return uint16(b[0])<<8 | uint16(b[1]) }
	be32 := func(b []byte) uint32 {
		return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
	}
	cs.pih = pictureHeader{
		lcod: be32(body[0:4]),
		ppih: be16(body[4:6]),
		plev: be16(body[6:8]),
		wf:   be16(body[8:10]),
		hf:   be16(body[10:12]),
		cw:   be16(body[12:14]),
		hsl:  be16(body[14:16]),
		nc:   body[16],
		ng:   body[17],
		ss:   body[18],
		bw:   body[19],
	}
	cs.pih.fq = body[20] >> 4
	cs.pih.br = body[20] & 0x0f
	cs.pih.fslc = (body[21] >> 7) & 0x01
	cs.pih.ppoc = (body[21] >> 4) & 0x07
	cs.pih.cpih = body[21] & 0x0f
	cs.pih.nlx = body[22] >> 4
	cs.pih.nly = body[22] & 0x0f
	cs.pih.lh = (body[23] >> 7) & 0x01
	cs.pih.rl = (body[23] >> 6) & 0x01
	cs.pih.qpih = (body[23] >> 4) & 0x03
	cs.pih.fs = (body[23] >> 2) & 0x03
	cs.pih.rm = body[23] & 0x03

	p := &cs.pih
	if p.nc == 0 || p.nc > 8 {
		return fmt.Errorf("jpegxs decode: PIH Nc must be 1..=8, got %d", p.nc)
	}
	if p.wf == 0 || p.hf == 0 {
		return fmt.Errorf("jpegxs decode: PIH dimensions must be non-zero, got %dx%d", p.wf, p.hf)
	}
	switch p.cpih {
	case 0, 1, 3:
	default:
		return fmt.Errorf("jpegxs decode: PIH Cpih = %d is reserved (Table A.9 defines only 0, 1, 3)", p.cpih)
	}
	if p.qpih > 1 {
		return fmt.Errorf("jpegxs decode: PIH Qpih = %d is reserved (Table A.10)", p.qpih)
	}
	if p.fs > 1 {
		return fmt.Errorf("jpegxs decode: PIH Fs = %d is reserved (Table A.11)", p.fs)
	}
	if p.rm > 1 {
		return fmt.Errorf("jpegxs decode: PIH Rm = %d is reserved (Table A.12)", p.rm)
	}
	return nil
}

// parseCDT parses the component table body.
func (cs *codestream) parseCDT(body []byte) error {
	nc := int(cs.pih.nc)
	if len(body) != nc*2 {
		return fmt.Errorf("jpegxs decode: CDT body must be 2*Nc = %d bytes, got %d", nc*2, len(body))
	}
	cs.cdt = make([]component, nc)
	for i := 0; i < nc; i++ {
		bd := body[i*2]
		pack := body[i*2+1]
		sx := pack >> 4
		sy := pack & 0x0f
		if bd < 8 || bd > 16 {
			return fmt.Errorf("jpegxs decode: CDT component %d bit depth %d out of range 8..=16", i, bd)
		}
		if sx == 0 || sy == 0 {
			return fmt.Errorf("jpegxs decode: CDT component %d sampling factors %d/%d must be >= 1", i, sx, sy)
		}
		cs.cdt[i] = component{bitDepth: bd, sx: sx, sy: sy}
	}
	return nil
}
