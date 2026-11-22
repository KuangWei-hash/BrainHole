package jpegxs

import "fmt"

// Picture decoding — the slice walk (Annex B), the per-precinct packet
// loop and the synthesis of the component samples.
//
// The supported subset is the one the Go encoder produces: a single slice,
// a single precinct column (Cw = 0) and a single horizontal DWT level
// (NL,x = 1, NL,y = 0).

// ceilDiv divides a by b rounding up.
func ceilDiv(a, b int) int {
	if b == 0 {
		return 0
	}
	return (a + b - 1) / b
}

// cfaPatternType derives the CFA pattern type Ct from the CRG marker
// (Annex F.5, Table F.9).
func cfaPatternType(crg []crgEntry) (uint8, bool) {
	if len(crg) < 4 {
		return 0, false
	}
	quantise := func(v uint16) uint8 {
		if v == 0 {
			return 0
		}
		return 1
	}
	var q [4][2]uint8
	for i := 0; i < 4; i++ {
		q[i] = [2]uint8{quantise(crg[i].x), quantise(crg[i].y)}
	}
	rgGB := [4][2]uint8{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	bggr := [4][2]uint8{{1, 1}, {1, 0}, {0, 1}, {0, 0}}
	grbg := [4][2]uint8{{1, 0}, {0, 0}, {1, 1}, {0, 1}}
	gbrg := [4][2]uint8{{0, 1}, {0, 0}, {1, 1}, {1, 0}}
	if q == rgGB || q == bggr {
		return 0, true
	}
	if q == grbg || q == gbrg {
		return 1, true
	}
	return 0, false
}

// decodePicture decodes the parsed codestream into component planes.
func decodePicture(cs *codestream) (*Components, error) {
	p := &cs.pih
	nc := int(p.nc)

	// --- Subset validation -------------------------------------------------
	if p.nlx != 1 || p.nly != 0 {
		return nil, fmt.Errorf("jpegxs decode: NL,x=%d NL,y=%d is outside the supported subset (NL,x=1, NL,y=0)", p.nlx, p.nly)
	}
	if p.cw != 0 {
		return nil, fmt.Errorf("jpegxs decode: Cw=%d is outside the supported subset (Cw=0, a single precinct column)", p.cw)
	}
	if p.hsl == 0 {
		return nil, fmt.Errorf("jpegxs decode: Hsl must be at least 1")
	}
	if p.qpih > 1 {
		return nil, fmt.Errorf("jpegxs decode: Qpih=%d reserved for ISO/IEC use (Table A.10)", p.qpih)
	}
	if p.fq == 0 {
		if p.bw != cs.cdt[0].bitDepth {
			return nil, fmt.Errorf("jpegxs decode: (Bw=%d, Fq=0) must match B[0]=%d (Table A.8)", p.bw, cs.cdt[0].bitDepth)
		}
	} else if !(p.bw == 18 && p.fq == 6) && !(p.bw == 20 && p.fq == 8) {
		return nil, fmt.Errorf("jpegxs decode: (Bw=%d, Fq=%d) is not a valid Table A.8 combination", p.bw, p.fq)
	}
	if len(cs.cdt) != nc {
		return nil, fmt.Errorf("jpegxs decode: CDT has %d components but PIH says Nc=%d", len(cs.cdt), nc)
	}

	ct := uint8(0)
	if p.cpih == 3 {
		if cs.cts == nil {
			return nil, fmt.Errorf("jpegxs decode: Cpih=3 requires the CTS marker (A.4.8)")
		}
		if !cs.hasCRG {
			return nil, fmt.Errorf("jpegxs decode: Cpih=3 requires the CRG marker (A.4.9)")
		}
		var ok bool
		ct, ok = cfaPatternType(cs.crg)
		if !ok {
			return nil, fmt.Errorf("jpegxs decode: CRG entries do not match a Table F.9 CFA pattern (RGGB/BGGR/GRBG/GBRG)")
		}
		if nc < 4 {
			return nil, fmt.Errorf("jpegxs decode: Star-Tetrix requires Nc>=4 per Annex F.2, got %d", nc)
		}
		for i := 0; i < 4; i++ {
			if cs.cdt[i].sx != 1 || cs.cdt[i].sy != 1 {
				return nil, fmt.Errorf("jpegxs decode: Star-Tetrix requires sx[i]=sy[i]=1 for i<4, got component %d sx=%d sy=%d (Annex A.4.3 / F.2)",
					i, cs.cdt[i].sx, cs.cdt[i].sy)
			}
		}
	}

	// --- Band geometry (Annex B) -------------------------------------------
	//
	// NL,x = 1 and NL,y = 0 give Nβ = 2 filter types: β = 0 is the LL band
	// and β = 1 the horizontal-only HL band. The bitstream's flat band id
	// is b = (Nc − Sd) × β + i, so with Sd = 0 the band ids run 0..2·Nc−1
	// in β-major order — which is also the order of the WGT marker's
	// (G[b], P[b]) pairs and of the precinct header's D[p,b] entries.
	const nBeta = 2
	bands := make([]bandGeometry, nc*nBeta)
	for beta := 0; beta < nBeta; beta++ {
		for i := 0; i < nc; i++ {
			b := nc*beta + i
			wc := ceilDiv(int(p.wf), int(cs.cdt[i].sx))
			var wpb uint32
			if beta == 0 {
				wpb = uint32((wc + 1) / 2)
			} else {
				wpb = uint32(wc / 2)
			}
			var gain, priority uint8
			if 2*b+1 < len(cs.wgt) {
				gain = cs.wgt[2*b]
				priority = cs.wgt[2*b+1]
			}
			bands[b] = bandGeometry{
				wpb:      wpb,
				gain:     gain,
				priority: priority,
				l0:       0,
				l1:       1,
				exists:   true,
			}
		}
	}
	geom := precinctGeometry{
		bands:             bands,
		ng:                p.ng,
		ss:                p.ss,
		br:                p.br,
		fs:                p.fs,
		rm:                p.rm,
		rl:                p.rl,
		lh:                p.lh,
		shortPacketHeader: uint32(p.wf)*uint32(p.nc) < 32752,
	}

	// The packet layout of Table B.4: with NL,y = 0 the first packet
	// carries β1 = max(NL,x,NL,y) − min(NL,x,NL,y) + 1 = 2 filter types,
	// i.e. every band of every component, all on line 0.
	layout := packetLayout{}
	for beta := 0; beta < nBeta; beta++ {
		for i := 0; i < nc; i++ {
			layout.entries = append(layout.entries, packetEntry{band: uint16(nc*beta + i), line: 0})
		}
	}

	// --- Component sample buffers ------------------------------------------
	compW := make([]int, nc)
	compH := make([]int, nc)
	samples := make([][]int32, nc)
	for i := 0; i < nc; i++ {
		compW[i] = ceilDiv(int(p.wf), int(cs.cdt[i].sx))
		compH[i] = ceilDiv(int(p.hf), int(cs.cdt[i].sy))
		samples[i] = make([]int32, compW[i]*compH[i])
	}

	// --- Precinct loop ------------------------------------------------------
	//
	// Hp = 2^NL,y = 1, so each precinct covers one picture row and the
	// picture has Hf precinct rows in a single column. Hsl is the slice
	// height in precinct rows (Annex B.10): the slice count is
	// ⌈Np,y / Hsl⌉ and slice t owns the rows [t·Hsl, (t+1)·Hsl).
	npY := int(p.hf)
	hsl := int(p.hsl)
	nSlices := (npY + hsl - 1) / hsl
	if len(cs.slices) != nSlices {
		return nil, fmt.Errorf("jpegxs decode: picture declares %d slices (Np,y=%d, Hsl=%d) but the codestream carries %d",
			nSlices, npY, hsl, len(cs.slices))
	}

	for si, sl := range cs.slices {
		if int(sl.yslh) != si {
			return nil, fmt.Errorf("jpegxs decode: slice %d carries Yslh=%d; the slices must appear in order", si, sl.yslh)
		}
		pyStart := si * hsl
		pyEnd := pyStart + hsl
		if pyEnd > npY {
			pyEnd = npY
		}
		pos := 0
		for py := pyStart; py < pyEnd; py++ {
			if pos >= len(sl.data) {
				return nil, fmt.Errorf("jpegxs decode: slice %d ended after %d of %d precincts", si, py-pyStart, pyEnd-pyStart)
			}
			ph, err := parsePrecinctHeader(sl.data[pos:], &geom)
			if err != nil {
				return nil, err
			}
			pos += ph.headerBytes

			pk, err := parsePacketHeader(sl.data[pos:], &geom)
			if err != nil {
				return nil, err
			}
			pos += pk.headerBytes

			state := precinctState{}
			consumed, err := decodePacketBodyInto(sl.data[pos:], &geom, ph, pk, &layout, &state, nil)
			if err != nil {
				return nil, err
			}
			pos += consumed

			// Dequantise (Annex D).
			trunc := precinctTruncation(&geom, ph)
			dequant := make([][]int32, len(bands))
			for b := range bands {
				if !bands[b].exists {
					continue
				}
				coef := state.coefficients[b]
				out := make([]int32, int(coef.wpb)*int(coef.numLines))
				inverseQuantize(p.qpih, &coef, trunc[b], p.ng, out)
				dequant[b] = out
			}

			// Synthesise the component rows (Annex E).
			for i := 0; i < nc; i++ {
				ll := dequant[i]
				hl := dequant[nc+i]
				wc := compW[i]
				row := make([]int32, wc)
				synthLine(ll, hl, row)
				targetRow := py
				if targetRow >= compH[i] {
					continue
				}
				copy(samples[i][targetRow*wc:], row)
			}
		}
	}

	// --- Guard band ---------------------------------------------------------
	const sampleGuard = 1 << 26
	for i := range samples {
		for j := range samples[i] {
			if samples[i][j] > sampleGuard {
				samples[i][j] = sampleGuard
			} else if samples[i][j] < -sampleGuard {
				samples[i][j] = -sampleGuard
			}
		}
	}

	// --- Inverse colour transform (Annex F) ---------------------------------
	switch p.cpih {
	case 1:
		return nil, fmt.Errorf("jpegxs decode: Cpih=1 (RCT) is not implemented yet")
	case 3:
		var planes [4][]int32
		for i := 0; i < 4; i++ {
			planes[i] = samples[i]
		}
		if err := inverseStarTetrix(&planes, int(p.wf), int(p.hf), cs.cts.e1, cs.cts.e2, ct, cs.cts.cf); err != nil {
			return nil, err
		}
		for i := 0; i < 4; i++ {
			samples[i] = planes[i]
		}
	}

	// --- Output scaling (Annex G) ------------------------------------------
	outPlanes := make([][]byte, nc)
	for i := 0; i < nc; i++ {
		b, err := applyOutputScaling(samples[i], p.bw, cs.cdt[i].bitDepth)
		if err != nil {
			return nil, err
		}
		outPlanes[i] = b
	}

	bitDepths := make([]uint8, nc)
	sampling := make([][2]uint8, nc)
	for i := 0; i < nc; i++ {
		bitDepths[i] = cs.cdt[i].bitDepth
		sampling[i] = [2]uint8{cs.cdt[i].sx, cs.cdt[i].sy}
	}
	return &Components{
		Width:     uint32(p.wf),
		Height:    uint32(p.hf),
		Cpih:      p.cpih,
		BitDepths: bitDepths,
		Sampling:  sampling,
		Planes:    outPlanes,
	}, nil
}
