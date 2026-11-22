package jpegxs

import (
	"fmt"
)

// Slice and precinct emission — ISO/IEC 21122-1:2022 Annex B (band
// layout) and Annex C (precincts and packets).
//
// Port of the encoder-side half of the upstream Rust implementation
// (oxideav-jpegxs `src/encoder.rs`).

// writeSlice writes the picture's slices: the per-component colour
// transform, the forward DWT, and the per-precinct emission.
func writeSlice(out *[]byte, cfg *encodeConfig, planesU8 [][]byte) error {
	w := int(cfg.width)
	h := int(cfg.height)
	nc := int(cfg.nc)
	dcBias := int32(1) << (cfg.bw - 1)

	// 1) Move the input samples into the wavelet domain. The forward map
	// is Ω = (sample << ζ) − dc_bias, the exact inverse of the Annex G.2
	// (Table G.2) output scaling R = (Ω + dc_bias + half) >> ζ.
	zeta := int32(cfg.bw) - int32(cfg.bitDepth)
	if zeta < 0 {
		zeta = 0
	}
	compPlanes := make([][]int32, nc)
	for i, p := range planesU8 {
		q := make([]int32, len(p))
		for j, v := range p {
			q[j] = (int32(v) << zeta) - dcBias
		}
		compPlanes[i] = q
	}

	// 2) Per-component colour transform.
	switch cfg.cpih {
	case 1:
		if err := forwardRCT(compPlanes, w, h); err != nil {
			return err
		}
	case 3:
		if err := forwardStarTetrix(compPlanes, w, h, cfg.ctsE1, cfg.ctsE2, cfg.stCT, cfg.ctsCF); err != nil {
			return err
		}
	}

	nlx := cfg.nlx
	nly := cfg.nly

	// Compute Cs / Np,x per Annex B.5.
	maxSx := uint32(1)
	for _, v := range cfg.sx {
		if uint32(v) > maxSx {
			maxSx = uint32(v)
		}
	}
	var cs uint32
	if cfg.cw == 0 {
		cs = uint32(w)
	} else {
		cs = 8 * uint32(cfg.cw) * maxSx * (1 << nlx)
	}
	npX := int((uint32(w) + cs - 1) / cs)

	// Route everything with nlx > 1 or nly >= 1 through the cascade path.
	// Cw > 0 (Np,x > 1) also forces the cascade path because per-precinct
	// DWT does not commute with a multi-precinct-per-row layout.
	multiLevel := nlx > 1 || nly >= 1 || npX > 1 || cfg.sd > 0
	hpPow := 1 << nly
	npY := (h + hpPow - 1) / hpPow

	// Slice grouping per Annex B.10: precinct rows are partitioned into
	// ⌈Np,y / Hsl⌉ slices of hslRows rows each. cfg.hsl == 0 is the
	// single-slice mode.
	hslRows := npY
	if cfg.hsl != 0 {
		hslRows = int(cfg.hsl)
	}

	// SLH writer: Lslh = 4, body = Yslh (u16). Annex A.4.12 Table A.25.
	writeSLH := func(out *[]byte, t uint16) {
		*out = append(*out, 0xff, 0x20)
		*out = append(*out, 0, 4)
		*out = append(*out, byte(t>>8), byte(t))
	}

	if multiLevel {
		return writeSliceCascade(out, cfg, compPlanes, npY, npX, hslRows, writeSLH)
	}

	// NL=1/1 streaming per-precinct path.
	t := uint16(0)
	pyStart := 0
	for pyStart < npY {
		pyEnd := min(pyStart+hslRows, npY)
		writeSLH(out, t)
		for py := pyStart; py < pyEnd; py++ {
			y0 := py * hpPow
			y1 := min(y0+hpPow, h)
			hpReal := y1 - y0
			pbytes, err := encodePrecinctSingleLevel(cfg, compPlanes, y0, y1, hpReal)
			if err != nil {
				return err
			}
			*out = append(*out, pbytes...)
		}
		pyStart = pyEnd
		t++
	}
	return nil
}

// forwardRCT applies the reversible colour transform (Cpih = 1) — Annex
// F.4, Table F.3.
func forwardRCT(planes [][]int32, wf, hf int) error {
	if len(planes) < 3 {
		return fmt.Errorf("jpegxs Cpih=1 (forward RCT) requires at least 3 component planes, got %d", len(planes))
	}
	want := wf * hf
	for i := 0; i < 3; i++ {
		if len(planes[i]) != want {
			return fmt.Errorf("jpegxs forward RCT: component %d has %d samples, expected %d", i, len(planes[i]), want)
		}
	}
	for x := 0; x < want; x++ {
		r := planes[0][x]
		g := planes[1][x]
		b := planes[2][x]
		planes[0][x] = (r + 2*g + b) >> 2
		planes[1][x] = b - g
		planes[2][x] = r - g
	}
	return nil
}

// compBands holds the per-component sub-bands of the single-level path.
type compBands struct {
	nlyI           uint8
	ll             []int32
	hl             []int32
	lh             []int32
	hh             []int32
	llW            int
	hlW            int
	llHPerPrecinct int
	lhHPerPrecinct int
	picLLH         int
	picLHH         int
}

// encodePrecinctSingleLevel encodes one precinct of the NL,x = 1 /
// NL,y = 0 streaming path.
//
// One single-column precinct row is a complete horizontal transform unit,
// so per-precinct equals picture-level here.
func encodePrecinctSingleLevel(cfg *encodeConfig, compPlanes [][]int32, y0, y1, hpReal int) ([]byte, error) {
	if cfg.nly != 0 {
		return nil, fmt.Errorf("jpegxs encoder: streaming single-level path requires NL,y = 0")
	}
	w := int(cfg.width)
	hFull := int(cfg.height)
	nc := int(cfg.nc)
	hpPow := 1 << cfg.nly

	compBandList := make([]*compBands, 0, nc)
	for i := 0; i < nc; i++ {
		plane := compPlanes[i]
		sxI := int(cfg.sx[i])
		syI := int(cfg.sy[i])
		wc := (w + sxI - 1) / sxI
		hc := (hFull + syI - 1) / syI
		hpI := hpPow / syI
		nlyI := effectiveNLY(cfg.nly, cfg.sy[i])

		// Per-precinct strip rows for this component.
		y0i := y0 / syI
		y1i := min((y1+syI-1)/syI, hc)
		hpRealI := max(y1i-y0i, 0)

		strip := make([]int32, 0, wc*hpI)
		for y := y0i; y < y1i; y++ {
			strip = append(strip, plane[y*wc:(y+1)*wc]...)
		}
		// Pad with whole-sample symmetric reflection up to hpI rows.
		for len(strip) < wc*hpI {
			targetRow := len(strip) / wc
			var srcRow int
			if hpRealI >= 2 {
				mirrored := 2*hpRealI - targetRow - 2
				srcRow = min(mirrored, hpRealI-1)
			}
			rowStart := srcRow * wc
			for x := 0; x < wc; x++ {
				var val int32
				if hpRealI != 0 {
					val = strip[rowStart+x]
				}
				strip = append(strip, val)
			}
		}

		llW := (wc + 1) / 2
		hlW := wc / 2
		llHPerPrecinct := (hpI + 1) / 2
		lhHPerPrecinct := hpI / 2
		// Picture-level LL band height: nlyI == 0 → no vertical split, so
		// the LL rows equal the component rows (no /2).
		picLLH := hc
		picLHH := 0
		if nlyI != 0 {
			picLLH = (hc + 1) / 2
			picLHH = hc / 2
		}

		cb := &compBands{
			nlyI:           nlyI,
			llW:            llW,
			hlW:            hlW,
			llHPerPrecinct: llHPerPrecinct,
			lhHPerPrecinct: lhHPerPrecinct,
			picLLH:         picLLH,
			picLHH:         picLHH,
		}
		if nlyI == 0 {
			// 1-D horizontal-only DWT: one row, two bands (LL, HL).
			ll := make([]int32, llW)
			hl := make([]int32, hlW)
			if err := forwardHorizontal1D(strip, ll, hl); err != nil {
				return nil, err
			}
			fqDownscaleBand(ll, cfg.fq)
			fqDownscaleBand(hl, cfg.fq)
			cb.ll = ll
			cb.hl = hl
		} else {
			// 2-D DWT.
			ll := make([]int32, llW*llHPerPrecinct)
			hl := make([]int32, hlW*llHPerPrecinct)
			lh := make([]int32, llW*lhHPerPrecinct)
			hh := make([]int32, hlW*lhHPerPrecinct)
			if err := forward2D(wc, hpI, strip, ll, hl, lh, hh); err != nil {
				return nil, err
			}
			fqDownscaleBand(ll, cfg.fq)
			fqDownscaleBand(hl, cfg.fq)
			fqDownscaleBand(lh, cfg.fq)
			fqDownscaleBand(hh, cfg.fq)
			cb.ll = ll
			cb.hl = hl
			cb.lh = lh
			cb.hh = hh
		}
		compBandList = append(compBandList, cb)
	}

	// Per-component, per-precinct line counts.
	py := y0 / hpPow
	linesLLRealPerComp := make([]int, nc)
	linesLHRealPerComp := make([]int, nc)
	for i, cb := range compBandList {
		rowOffset := py
		if rowOffset >= cb.picLLH {
			linesLLRealPerComp[i] = 0
		} else {
			linesLLRealPerComp[i] = min(cb.llHPerPrecinct, cb.picLLH-rowOffset)
		}
		if cb.nlyI == 0 {
			linesLHRealPerComp[i] = 0
		} else if rowOffset >= cb.picLHH {
			linesLHRealPerComp[i] = 0
		} else {
			linesLHRealPerComp[i] = min(cb.lhHPerPrecinct, cb.picLHH-rowOffset)
		}
	}

	// Precinct-header band-existence bookkeeping: β=0 / 1 always exist for
	// every component; β=2 / 3 only for components with nlyI >= 1.
	nExisting := 0
	for _, cb := range compBandList {
		nExisting += 2 // LL + HL always exist
		if cb.nlyI >= 1 {
			nExisting += 2 // LH + HH
		}
	}
	headerBits := 24 + 8 + 8 + 2*nExisting
	headerBytes := (headerBits + 7) / 8
	precinctBytes := make([]byte, headerBytes)

	var entropy []byte

	// Per-band truncation positions. Single-level: β=0→G=0, β=1→G=1,
	// β=2→G=1, β=3→G=2, using the gain-weighted
	// T[p,b] = clamp(Q−G−r, 0, 15) of Annex C.6.2 Table C.10.
	ncU32 := uint32(cfg.nc)
	tForBand := func(gain uint8, beta uint32, compI int) uint8 {
		bandIndex := ncU32*beta + uint32(compI)
		refine := int32(0)
		if bandIndex < uint32(cfg.rp) {
			refine = 1
		}
		v := int32(cfg.q) - int32(gain) - refine
		if v < 0 {
			v = 0
		}
		if v > 15 {
			v = 15
		}
		return uint8(v)
	}

	// First packet (Annex B.7 Table B.4): the leading filter types of all
	// components are grouped jointly. With NL,y = 0 the first packet holds
	// β = 0 (LL) and β = 1 (HL) of every component.
	pictureNly0 := cfg.nly == 0
	var firstEntries []*perBandEntry
	for i, cb := range compBandList {
		if linesLLRealPerComp[i] == 0 {
			continue
		}
		// β=0 (LL) gain = 0.
		firstEntries = append(firstEntries, &perBandEntry{
			wpb:  uint32(cb.llW),
			line: cb.ll[:cb.llW],
			t:    tForBand(0, 0, i),
		})
	}
	// β=1 (HL) joins the first packet for every component when the
	// picture-level NL,y = 0, appended after all LL entries to preserve
	// β-major order.
	if pictureNly0 {
		for i, cb := range compBandList {
			if linesLLRealPerComp[i] == 0 {
				continue
			}
			firstEntries = append(firstEntries, &perBandEntry{
				wpb:  uint32(cb.hlW),
				line: cb.hl[:cb.hlW],
				t:    tForBand(1, 1, i),
			})
		}
	}
	if len(firstEntries) != 0 {
		if err := emitPacket(&entropy, cfg, firstEntries); err != nil {
			return nil, err
		}
	}

	// Proxy levels: β=1 (HL, G=1), β=2 (LH, G=1), β=3 (HH, G=2). All
	// existing components of a (β, line) are coded jointly in one packet.
	for betaIdx := 1; betaIdx <= 3; betaIdx++ {
		// At NL,y = 0 every component's HL band was already grouped into
		// the first packet.
		if betaIdx == 1 && pictureNly0 {
			continue
		}
		var entries []*perBandEntry
		for i, cb := range compBandList {
			if betaIdx >= 2 && cb.nlyI == 0 {
				continue
			}
			linesReal := linesLHRealPerComp[i]
			if betaIdx == 1 {
				linesReal = linesLLRealPerComp[i]
			}
			if linesReal == 0 {
				continue
			}
			var bandBuf []int32
			var wpb int
			switch betaIdx {
			case 1:
				bandBuf, wpb = cb.hl, cb.hlW
			case 2:
				bandBuf, wpb = cb.lh, cb.llW
			default:
				bandBuf, wpb = cb.hh, cb.hlW
			}
			// Gain per sub-band type: HL/LH = 1, HH = 2.
			gain := uint8(1)
			if betaIdx > 2 {
				gain = 2
			}
			entries = append(entries, &perBandEntry{
				wpb:  uint32(wpb),
				line: bandBuf[:wpb],
				t:    tForBand(gain, uint32(betaIdx), i),
			})
		}
		if len(entries) != 0 {
			if err := emitPacket(&entropy, cfg, entries); err != nil {
				return nil, err
			}
		}
	}

	lprc := len(entropy)
	if lprc == 0 {
		return nil, fmt.Errorf("jpegxs encoder: empty precinct (Lprc must be >= 1)")
	}
	if lprc > (1<<20)-1 {
		return nil, fmt.Errorf("jpegxs encoder: precinct exceeds 2^20-1 bytes (Lprc = %d)", lprc)
	}
	precinctBytes[0] = byte((lprc >> 16) & 0xff)
	precinctBytes[1] = byte((lprc >> 8) & 0xff)
	precinctBytes[2] = byte(lprc & 0xff)
	precinctBytes[3] = min(cfg.q, 31)
	precinctBytes[4] = cfg.rp // R[p] — precinct refinement (Annex C.2 Table C.1)
	// D[p,b] bits at offset 5+ stay 0.
	precinctBytes = append(precinctBytes, entropy...)
	return precinctBytes, nil
}
