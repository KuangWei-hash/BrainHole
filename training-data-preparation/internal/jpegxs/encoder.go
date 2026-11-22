// Package jpegxs implements a JPEG XS (ISO/IEC 21122-1) encoder in Go.
//
// It is a port of the encoder half of the OxideAV/oxideav-jpegxs Rust
// crate (rev 3484a041d6d8e893683c1920cb3f68fe6fd3b8b8, crate 0.0.9),
// restricted to the subset the BrainHole training-data pipeline needs:
//
//   - Cpih = 0 / 1 / 3 (no colour transform, RCT, Star-Tetrix)
//   - 8-bit components, uniform bit depth
//   - NL,x = 1..5, NL,y = 0..1
//   - Q[p] = 0..15, Qpih = 0 (deadzone inverse quantiser)
//   - a single slice, a single precinct column, no CWD suppression,
//     no run mode, no refinement, no NLT, no profile shaping
//
// The goal is byte-for-byte parity with the upstream Rust bridge, so the
// encoded training data stays interchangeable with data produced by the
// Rust encoder. Anything outside the supported subset is rejected with
// an error rather than silently encoded differently.
package jpegxs

import (
	"fmt"
)

// Options configures the encoder.
type Options struct {
	// Quantization is the picture-level Q[p] (0..15). 0 is lossless.
	Quantization uint8
	// LevelsX is NL,x (1..5).
	LevelsX uint8
	// LevelsY is NL,y (0..1, and never greater than LevelsX).
	LevelsY uint8
	// StarTetrix enables the Star-Tetrix transform (Cpih = 3).
	StarTetrix bool
	// CTS holds the CTS / CRG parameters used with StarTetrix.
	CTS CTSParams
	// Br is the raw bitplane-count width (Table A.7).
	//
	// ISO/IEC 21122-1:2022 Table A.7 lists exactly one conformant value
	// for Br — 4 — alongside Ng = 4 and Ss = 8. The upstream Rust encoder
	// writes Br = 8, which a conforming third-party decoder rejects
	// (libsvtjpegxs: "Picture header invalid Br expected 4, read=8"), so
	// the default here is 4.
	//
	// A value of 0 selects the default. Br = 8 is accepted for
	// byte-for-byte parity with the upstream encoder, at the cost of
	// interoperability.
	Br uint8
}

// CTSParams holds the CTS marker fields for the Star-Tetrix transform.
type CTSParams struct {
	// E1 and E2 are the chroma-weighting exponents (0..3).
	E1, E2 uint8
	// CF is the CTS extent (0 = full transform, 3 = in-line).
	CF uint8
	// CT is the CFA pattern type (Table F.9: 0 = RGGB / BGGR,
	// 1 = GRBG / GBRG).
	CT uint8
}

// Components describes a planar picture.
type Components struct {
	// Width and Height are the picture dimensions in samples.
	Width, Height uint32
	// Cpih selects the colour transform (0 = none, 1 = RCT, 3 = Star-Tetrix).
	Cpih uint8
	// BitDepths holds B[i] per component; all entries must be equal.
	BitDepths []uint8
	// Sampling holds the (sx, sy) sampling factors per component.
	Sampling [][2]uint8
	// Planes holds the component samples, row-major, one byte per sample
	// for 8-bit components.
	Planes [][]byte
}

// defaultBr is the raw bitplane-count width. ISO/IEC 21122-1:2022
// Table A.7 lists exactly one conformant value for Br — 4 — next to
// Ng = 4 and Ss = 8.
const defaultBr = 4

// encodeConfig is the internal encoder configuration.
type encodeConfig struct {
	width    uint16
	height   uint16
	nc       uint8
	bitDepth uint8
	bw       uint8
	lh       uint8
	ng       uint8
	ss       uint8
	br       uint8
	nlx      uint8
	nly      uint8
	cpih     uint8
	qpih     uint8
	fs       uint8
	fq       uint8
	q        uint8
	sx       []uint8
	sy       []uint8
	ctsE1    uint8
	ctsE2    uint8
	ctsCF    uint8
	stCT     uint8
	rp       uint8
	cw       uint16
	sd       uint8
	hsl      uint16
	rm       uint8

	bandGains      []uint8
	bandPriorities []uint8
}

// longPacketHeaders reports whether the packet header is long: Table C.3
// makes it long unless Wf × Nc < 32752 and Lh == 0.
func (c *encodeConfig) longPacketHeaders() bool {
	return c.lh != 0 || uint32(c.width)*uint32(c.nc) >= 32752
}

// nBeta returns the number of wavelet filter types Nβ (Annex B.3).
func nBeta(nlx, nly uint8) uint32 {
	mn := uint32(min(nlx, nly))
	mx := uint32(max(nlx, nly))
	return 2*mn + mx + 1
}

// EncodeComponents encodes a planar picture into a bare JPEG XS
// codestream.
func EncodeComponents(c *Components, opts *Options) ([]byte, error) {
	if opts == nil {
		opts = &Options{}
	}
	if c == nil {
		return nil, fmt.Errorf("jpegxs encoder: nil components")
	}
	if len(c.Planes) == 0 {
		return nil, fmt.Errorf("jpegxs encoder: no component planes")
	}
	if len(c.Planes) > 8 {
		return nil, fmt.Errorf("jpegxs encoder: Nc must be 1..=8 (Annex A.4.3), got %d", len(c.Planes))
	}
	if len(c.BitDepths) != len(c.Planes) {
		return nil, fmt.Errorf("jpegxs encoder: bit depth count %d != component count %d", len(c.BitDepths), len(c.Planes))
	}
	if len(c.Sampling) != len(c.Planes) {
		return nil, fmt.Errorf("jpegxs encoder: sampling count %d != component count %d", len(c.Sampling), len(c.Planes))
	}
	bd := c.BitDepths[0]
	for i, b := range c.BitDepths {
		if b != bd {
			return nil, fmt.Errorf("jpegxs encoder: mixed component bit depths (B[0]=%d, B[%d]=%d)", bd, i, b)
		}
	}
	if bd != 8 {
		return nil, fmt.Errorf("jpegxs encoder: only 8-bit components are supported, got B[0]=%d", bd)
	}
	if c.Width < 2 || c.Height < 2 {
		return nil, fmt.Errorf("jpegxs encoder: picture dimensions must be >= 2, got %dx%d", c.Width, c.Height)
	}
	if c.Width > 65535 || c.Height > 65535 {
		return nil, fmt.Errorf("jpegxs encoder: picture dimensions must fit in u16, got %dx%d", c.Width, c.Height)
	}
	nc := uint8(len(c.Planes))
	switch c.Cpih {
	case 0:
	case 1:
		if nc < 3 {
			return nil, fmt.Errorf("jpegxs encoder: Cpih=1 (RCT) requires Nc>=3 per Annex F.2, got %d", nc)
		}
	case 3:
		if nc < 4 {
			return nil, fmt.Errorf("jpegxs encoder: Cpih=3 (Star-Tetrix) requires Nc>=4 per Annex F.2, got %d", nc)
		}
	default:
		return nil, fmt.Errorf("jpegxs encoder: Cpih must be 0, 1 or 3, got %d", c.Cpih)
	}
	if opts.Quantization > 15 {
		return nil, fmt.Errorf("jpegxs encoder: quantization %d outside 0..=15", opts.Quantization)
	}
	nlx, nly := opts.LevelsX, opts.LevelsY
	if nlx < 1 || nlx > 5 {
		return nil, fmt.Errorf("jpegxs encoder: NL,x must be 1..=5, got %d", nlx)
	}
	if nly > 1 {
		return nil, fmt.Errorf("jpegxs encoder: NL,y must be 0..=1, got %d", nly)
	}
	if nly > nlx {
		return nil, fmt.Errorf("jpegxs encoder: NL,y=%d must not exceed NL,x=%d", nly, nlx)
	}
	if opts.StarTetrix != (c.Cpih == 3) {
		return nil, fmt.Errorf("jpegxs encoder: StarTetrix option (%v) disagrees with Cpih=%d",
			opts.StarTetrix, c.Cpih)
	}
	// Br — the raw bitplane-count width. Table A.7 lists a single
	// conformant value (4); 8 is accepted for upstream byte parity.
	br := opts.Br
	if br == 0 {
		br = defaultBr
	}
	if br != 4 && br != 8 {
		return nil, fmt.Errorf("jpegxs encoder: Br must be 4 (the only conformant value, Table A.7) or 8, got %d", br)
	}

	sx := make([]uint8, nc)
	sy := make([]uint8, nc)
	for i, s := range c.Sampling {
		sx[i], sy[i] = s[0], s[1]
		if sx[i] == 0 || sy[i] == 0 {
			return nil, fmt.Errorf("jpegxs encoder: component %d has zero sampling factor", i)
		}
	}

	cfg := &encodeConfig{
		width:    uint16(c.Width),
		height:   uint16(c.Height),
		nc:       nc,
		bitDepth: bd,
		bw:       bd, // no NLT, no high precision → Table A.8 (Bw, Fq) = (B[0], 0)
		lh:       0,
		ng:       4,
		ss:       8,
		br:       br,
		nlx:      nlx,
		nly:      nly,
		cpih:     c.Cpih,
		qpih:     0, // deadzone inverse quantiser (Annex D.2)
		fs:       0, // signs jointly with data (Table C.8)
		fq:       0, // integer-transform path
		q:        opts.Quantization,
		sx:       sx,
		sy:       sy,
		ctsE1:    opts.CTS.E1,
		ctsE2:    opts.CTS.E2,
		ctsCF:    opts.CTS.CF,
		stCT:     opts.CTS.CT,
		rp:       0,
		cw:       0,
		sd:       0,
		hsl:      0,
		rm:       0,
	}
	cfg.bandGains = buildBandGainsSD(nc, cfg.sd, nlx, nly, sy)
	cfg.bandPriorities = buildBandPrioritiesSD(nc, cfg.sd, nlx, nly, sy)

	for i, p := range c.Planes {
		wc := (int(c.Width) + int(sx[i]) - 1) / int(sx[i])
		hc := (int(c.Height) + int(sy[i]) - 1) / int(sy[i])
		if len(p) != wc*hc {
			return nil, fmt.Errorf("jpegxs encoder: plane %d size %d != Wc*Hc %d (Wc=%d, Hc=%d)",
				i, len(p), wc*hc, wc, hc)
		}
	}

	out, err := buildCodestream(cfg, c.Planes)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// buildCodestream assembles the codestream: main header, slice, EOC.
//
// A sub-packet that outgrows the short packet header (possible below the
// Wf × Nc ≥ 32752 switch when a component is not vertically decomposed,
// Annex C.3 NOTE 2) forces long headers and a re-run.
func buildCodestream(cfg *encodeConfig, planes [][]byte) ([]byte, error) {
	build := func(cfg *encodeConfig) ([]byte, error) {
		out := make([]byte, 0, 1024)
		if err := writeMainHeader(&out, cfg); err != nil {
			return nil, err
		}
		if err := writeSlice(&out, cfg, planes); err != nil {
			return nil, err
		}
		out = append(out, 0xff, 0x11) // EOC
		return out, nil
	}
	out, err := build(cfg)
	if err != nil && cfg.lh == 0 && isShortHeaderOverflow(err) {
		cfg2 := *cfg
		cfg2.lh = 1
		return build(&cfg2)
	}
	return out, err
}

// isShortHeaderOverflow reports whether err is the short-packet-header
// overflow the driver recognises.
func isShortHeaderOverflow(err error) bool {
	return err != nil && len(err.Error()) > 0 &&
		containsStr(err.Error(), shortHeaderOverflow)
}

// containsStr is a tiny substring helper kept local to avoid pulling in
// strings for a single call site.
func containsStr(haystack, needle string) bool {
	n := len(needle)
	if n == 0 {
		return true
	}
	for i := 0; i+n <= len(haystack); i++ {
		if haystack[i:i+n] == needle {
			return true
		}
	}
	return false
}

// buildCapBytes builds the CAP marker cap[] byte array — the bytes after
// Lcap — for the tools this configuration uses, per ISO/IEC 21122-1:2022
// Table A.5 (MSB-first bit numbering; bit 0 is intentionally unused).
//
//	bit 1 — Star-Tetrix transform + CTS marker (Cpih == 3)
//	bit 2 — quadratic non-linear transform
//	bit 3 — extended non-linear transform
//	bit 4 — a component with sy[i] > 1 is present
//	bit 5 — component-dependent wavelet decomposition (CWD, Sd > 0)
//	bit 6 — lossless decoding (q == 0)
//	bit 8 — packet-based raw-mode switch (Rl = 1)
func buildCapBytes(cfg *encodeConfig) []byte {
	var bits [16]bool
	bits[1] = cfg.cpih == 3
	bits[4] = false // no vertical subsampling is supported
	bits[5] = cfg.sd > 0
	bits[6] = cfg.q == 0
	bits[8] = true

	var b [2]byte
	for i, set := range bits {
		if set {
			b[i/8] |= 1 << (7 - (i % 8))
		}
	}
	// Shrink Lcap so the last cap byte is not all zero (§A.4.3). Byte 1 is
	// always non-zero (bit 8), so both bytes are always emitted.
	n := len(b)
	for n > 0 && b[n-1] == 0 {
		n--
	}
	return b[:n]
}

// writeMainHeader writes SOC, CAP, PIH, CDT, WGT and — for Cpih = 3 —
// CTS and CRG.
func writeMainHeader(out *[]byte, cfg *encodeConfig) error {
	// SOC.
	*out = append(*out, 0xff, 0x10)

	// CAP — the capabilities marker declares the optional decoder tools
	// required to decode this codestream (§A.4.3, Table A.5).
	*out = append(*out, 0xff, 0x50)
	capBytes := buildCapBytes(cfg)
	lcap := uint16(2 + len(capBytes))
	*out = append(*out, byte(lcap>>8), byte(lcap))
	*out = append(*out, capBytes...)

	// PIH — Lpih = 26, body = 24 bytes.
	*out = append(*out, 0xff, 0x12)
	*out = append(*out, 0, 26)
	writePIHBody(out, cfg)

	// CDT — Lcdt = 2 + 2*Nc, body = 2*Nc bytes.
	*out = append(*out, 0xff, 0x13)
	lcdt := uint16(2 + 2*int(cfg.nc))
	*out = append(*out, byte(lcdt>>8), byte(lcdt))
	for i := 0; i < int(cfg.nc); i++ {
		*out = append(*out, cfg.bitDepth)
		*out = append(*out, (cfg.sx[i]&0x0f)<<4|(cfg.sy[i]&0x0f))
	}

	// WGT — one (G[b], P[b]) pair per existing band.
	nExisting := countExistingBands(cfg)
	*out = append(*out, 0xff, 0x14)
	lwgt := uint16(2 + 2*int(nExisting))
	*out = append(*out, byte(lwgt>>8), byte(lwgt))
	for k := 0; k < int(nExisting); k++ {
		var g, p uint8
		if k < len(cfg.bandGains) {
			g = cfg.bandGains[k]
		}
		if k < len(cfg.bandPriorities) {
			p = cfg.bandPriorities[k]
		}
		*out = append(*out, g, p)
	}

	// CTS + CRG — required when Cpih=3 (Star-Tetrix) per A.4.8 / A.4.9.
	if cfg.cpih == 3 {
		// CTS — Lcts = 4, body = 2 bytes (Reserved/Cf, e1/e2).
		*out = append(*out, 0xff, 0x18)
		*out = append(*out, 0, 4)
		*out = append(*out, cfg.ctsCF&0x0f)
		*out = append(*out, (cfg.ctsE1&0x0f)<<4|(cfg.ctsE2&0x0f))

		// CRG — Lcrg = 2 + 4*Nc, body = 4*Nc bytes.
		*out = append(*out, 0xff, 0x19)
		lcrg := uint16(2 + 4*int(cfg.nc))
		*out = append(*out, byte(lcrg>>8), byte(lcrg))
		// Canonical CRG entries mapping back to the chosen Ct via
		// Table F.9 (RGGB for Ct=0, GRBG for Ct=1).
		var cfa [4][2]uint16
		if cfg.stCT == 0 {
			cfa = [4][2]uint16{{0, 0}, {32768, 0}, {0, 32768}, {32768, 32768}}
		} else {
			cfa = [4][2]uint16{{32768, 0}, {0, 0}, {32768, 32768}, {0, 32768}}
		}
		for c := 0; c < int(cfg.nc); c++ {
			xc, yc := cfa[c][0], cfa[c][1]
			*out = append(*out, byte(xc>>8), byte(xc), byte(yc>>8), byte(yc))
		}
	}
	return nil
}

// writePIHBody writes the 24-byte picture header body.
func writePIHBody(out *[]byte, cfg *encodeConfig) {
	put16 := func(v uint16) { *out = append(*out, byte(v>>8), byte(v)) }
	put32 := func(v uint32) {
		*out = append(*out, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
	}

	put32(0) // Lcod
	put16(0) // Ppih
	put16(0) // Plev
	put16(cfg.width)
	put16(cfg.height)
	put16(cfg.cw) // Cw (0 = full-width precincts)
	put16(effectiveHsl(cfg))
	*out = append(*out, cfg.nc)
	*out = append(*out, cfg.ng)
	*out = append(*out, cfg.ss)
	*out = append(*out, cfg.bw)
	// Fq:Br
	*out = append(*out, (cfg.fq&0x0f)<<4|(cfg.br&0x0f))
	// Fslc:Ppoc:Cpih
	*out = append(*out, cfg.cpih&0x0f)
	// NL,x:NL,y
	*out = append(*out, (cfg.nlx&0x0f)<<4|(cfg.nly&0x0f))
	// Lh (1) : Rl (1) : Qpih (2) : Fs (2) : Rm (2).
	*out = append(*out,
		(cfg.lh&1)<<7|
			1<<6|
			(cfg.qpih&0x03)<<4|
			(cfg.fs&0x03)<<2|
			(cfg.rm&0x03))
}

// effectiveHsl resolves the Hsl field (slice height in precinct rows).
// cfg.hsl == 0 means a single slice covering the whole picture, which on
// the wire is Hsl = Np,y.
func effectiveHsl(cfg *encodeConfig) uint16 {
	hpPow := uint32(1) << cfg.nly
	npY := (uint32(cfg.height) + hpPow - 1) / hpPow
	if cfg.hsl == 0 {
		return uint16(npY)
	}
	return cfg.hsl
}

// countExistingBands counts the bands that exist in the WGT marker.
func countExistingBands(cfg *encodeConfig) uint32 {
	nbetaPic := nBeta(cfg.nlx, cfg.nly)
	nDecomposed := int(cfg.nc - cfg.sd)
	n := uint32(0)
	for i := 0; i < nDecomposed; i++ {
		for beta := uint32(0); beta < nbetaPic; beta++ {
			if _, ok := pictureBetaToLocalBeta(beta, cfg.nlx, cfg.nly, cfg.sy[i]); ok {
				n++
			}
		}
	}
	n += uint32(cfg.sd)
	return n
}

// pictureBetaToLocalBeta maps a picture-level filter index to the
// component's chroma-local filter index. It reports false when the band
// does not exist for that component (Annex B.4, bx[β,i] = 0).
func pictureBetaToLocalBeta(betaPic uint32, nlx, nlyPic, syI uint8) (uint32, bool) {
	if syI == 0 {
		return 0, false
	}
	var log2Sy uint8
	switch syI {
	case 1:
		log2Sy = 0
	case 2:
		log2Sy = 1
	case 4:
		log2Sy = 2
	default:
		return 0, false
	}
	nlyI := nlyPic
	if log2Sy > nlyI {
		nlyI = 0
	} else {
		nlyI -= log2Sy
	}
	if log2Sy == 0 {
		// 4:4:4 / 4:2:2 — identity mapping.
		return betaPic, true
	}
	// Decode the picture-β to (dx, dy_pic, tx, ty) under the picture-level
	// NL,y.
	dxPic, dyPic, tx, ty := betaLevels(betaPic, nlx, nlyPic)
	nlyPicU := uint32(nlyPic)
	nlyIU := uint32(nlyI)
	nlxU := uint32(nlx)
	// Annex B.4 existence check: bx[β,i] = 0 iff
	// 2^max(NL,y − dy)·τy[β] mod sy[i] != 0.
	pow := uint32(1)
	if dyPic <= nlyPicU {
		pow = 1 << (nlyPicU - dyPic)
	}
	l0Image := uint32(0)
	if ty {
		l0Image = pow
	}
	if l0Image%uint32(syI) != 0 {
		return 0, false
	}
	// LL — both picture and chroma have β=0 as LL.
	if !ty && !tx {
		return 0, true
	}
	if !ty {
		// The picture's pure-horizontal HL (τy=0, τx=1) at the picture's
		// level dy_pic = NL,y. The matching chroma band is chroma's HL at
		// the same dx in chroma's own enumeration.
		beta1Chroma := nlxU - nlyIU + 1
		if dxPic > nlyIU {
			return nlxU + 1 - dxPic, true
		}
		triple := nlyIU - dxPic
		return beta1Chroma + 3*triple, true
	}
	// τy = 1 (LH or HH). The band lives in the proxy zone of both picture
	// and chroma.
	if dyPic > nlyIU {
		return 0, false
	}
	beta1Chroma := nlxU - nlyIU + 1
	triple := nlyIU - dyPic
	within := uint32(1) // LH=1
	if tx {
		within = 2 // HH=2
	}
	return beta1Chroma + 3*triple + within, true
}

// betaLevels decodes a picture-level filter index to (dx, dy, τx, τy).
func betaLevels(beta uint32, nlx, nly uint8) (dx, dy uint32, tx, ty bool) {
	key := betaToBandKey(beta, nlx, nly)
	return key.dx, key.dy, key.tauX, key.tauY
}

// buildBandGainsSD builds the per-band gain values in WGT emission order
// (picture-level β, then component i). For the standard 5/3 wavelet the
// gain of each band is the number of high-pass axes.
func buildBandGainsSD(nc, sd, nlx, nly uint8, sy []uint8) []uint8 {
	nbetaPic := nBeta(nlx, nly)
	nDecomposed := int(nc - sd)
	var gains []uint8
	for beta := uint32(0); beta < nbetaPic; beta++ {
		for i := 0; i < nDecomposed; i++ {
			nlyI := effectiveNLY(nly, sy[i])
			localBeta, ok := pictureBetaToLocalBeta(beta, nlx, nly, sy[i])
			if !ok {
				continue
			}
			key := betaToBandKey(localBeta, nlx, nlyI)
			gain := uint8(0)
			if key.tauX {
				gain++
			}
			if key.tauY {
				gain++
			}
			gains = append(gains, gain)
		}
	}
	return gains
}

// buildBandPrioritiesSD builds the per-band priority values P[b] for the
// WGT marker (Annex A.4.11 Table A.24), in the same emission order as
// buildBandGainsSD.
func buildBandPrioritiesSD(nc, sd, nlx, nly uint8, sy []uint8) []uint8 {
	nbetaPic := nBeta(nlx, nly)
	nDecomposed := uint32(nc - sd)
	var prios []uint8
	for beta := uint32(0); beta < nbetaPic; beta++ {
		for i := 0; i < int(nDecomposed); i++ {
			if _, ok := pictureBetaToLocalBeta(beta, nlx, nly, sy[i]); !ok {
				continue
			}
			b := nDecomposed*beta + uint32(i)
			prios = append(prios, uint8(min(b, 255)))
		}
	}
	return prios
}

// effectiveNLY returns a component's effective vertical decomposition
// depth N′L,y[i] = NL,y − log2(sy[i]) (Annex B.2).
func effectiveNLY(nly, sy uint8) uint8 {
	var log2Sy uint8
	switch sy {
	case 1:
		log2Sy = 0
	case 2:
		log2Sy = 1
	case 4:
		log2Sy = 2
	default:
		log2Sy = 0
	}
	if log2Sy > nly {
		return 0
	}
	return nly - log2Sy
}
