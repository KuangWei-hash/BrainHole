package jpegxs

import "fmt"

// Entropy decoding — ISO/IEC 21122-1:2022 Annex C.
//
// Port of the decoder side of the upstream Rust implementation
// (oxideav-jpegxs `src/entropy/`): the precinct header (Table C.1), the
// packet header (Table C.3), the significance / bitplane-count / data /
// sign sub-packets (Tables C.5, C.8, C.9, C.12–C.14) and the truncation
// position rule (Table C.10).

// bandGeometry is the per-band metadata of one precinct.
type bandGeometry struct {
	// wpb is Wpb[p,b] — coefficients per line in this band.
	wpb uint32
	// gain is G[b] from the WGT segment.
	gain uint8
	// priority is P[b] from the WGT segment.
	priority uint8
	// l0 is L0[p,b] — the first line index of the band in the precinct.
	l0 uint16
	// l1 is L1[p,b] — one past the last line index of the band.
	l1 uint16
	// exists reports whether the band is present in this precinct.
	exists bool
}

// precinctGeometry carries the per-band metadata plus the per-precinct
// constants the picture header supplies.
type precinctGeometry struct {
	bands []bandGeometry
	// ng is Ng — coefficients per code group (4).
	ng uint8
	// ss is Ss — code groups per significance group (8).
	ss uint8
	// br is Br — raw bitplane-count width in bits.
	br uint8
	// fs is Fs — 0: signs inside the data sub-packet, 1: separate sign
	// sub-packet (Table A.11).
	fs uint8
	// rm is Rm — run mode (Table A.12).
	rm uint8
	// rl is Rl — raw-mode selection flag (Table A.7).
	rl uint8
	// lh is Lh — long packet-header enforcement (Table A.7).
	lh uint8
	// shortPacketHeader records the picture header's `Wf * Nc < 32752`
	// condition, which together with lh == 0 selects the short header.
	shortPacketHeader bool
}

// ncg returns Ncg[p,b] = ceil(Wpb[p,b] / Ng) — Annex B.8.
func (g *precinctGeometry) ncg(b int) uint32 {
	if g.ng == 0 {
		return 0
	}
	band := g.bands[b]
	return (band.wpb + uint32(g.ng) - 1) / uint32(g.ng)
}

// ns returns Ns[p,b] = ceil(Ncg[p,b] / Ss) — Annex B.8.
func (g *precinctGeometry) ns(b int) uint32 {
	if g.ss == 0 {
		return 0
	}
	return (g.ncg(b) + uint32(g.ss) - 1) / uint32(g.ss)
}

// useShortPacketHeader reports whether the short packet header form of
// Table C.3 applies.
func (g *precinctGeometry) useShortPacketHeader() bool {
	return g.lh == 0 && g.shortPacketHeader
}

// packetEntry is one (band, line) pair a packet covers.
type packetEntry struct {
	band uint16
	line uint16
}

// packetLayout is the ordered list of (band, line) pairs of one packet.
// The bitplane-count, data and sign sub-packets all walk them in this
// order.
type packetLayout struct {
	entries []packetEntry
}

// precinctHeader is the parsed precinct header (Table C.1).
type precinctHeader struct {
	lprc uint32
	q    uint8
	r    uint8
	// d holds D[p,b] per band.
	d           []uint8
	headerBytes int
}

// packetHeader is the parsed packet header (Table C.3).
type packetHeader struct {
	dr          uint8
	ldat        uint32
	lcnt        uint32
	lsgn        uint32
	shortForm   bool
	headerBytes int
}

// bandCoefficients holds one band's decoded data: the bitplane counts M,
// the quantization indices v and the sign bits s.
type bandCoefficients struct {
	wpb      uint32
	numLines uint32
	v        []uint32
	s        []uint8
	m        []uint8
}

func newBandCoefficients(wpb, ncg, numLines uint32) bandCoefficients {
	line := int(wpb)
	n := int(numLines)
	return bandCoefficients{
		wpb:      wpb,
		numLines: numLines,
		v:        make([]uint32, line*n),
		s:        make([]uint8, line*n),
		m:        make([]uint8, int(ncg)*n),
	}
}

// precinctState is the per-precinct state carried across packet bodies.
type precinctState struct {
	coefficients []bandCoefficients
	// sigFlags holds Z[p,λ,b,j] of the packet being decoded.
	sigFlags []uint8
	// sigStart records where each packet entry's flags begin.
	sigStart []int
}

// precinctTop is the vertical-prediction predecessor carried from the
// precinct directly above into the precinct below (Table C.11).
type precinctTop struct {
	// lastM holds the bitplane counts of the last decoded line of each
	// band, indexed by band id then code group.
	lastM [][]uint8
	// t holds the truncation position of each band.
	t []uint8
}

// truncationPosition computes T[p,b] per Annex C.6.2, Table C.10:
//
//	r = (P[b] < R[p]) ? 1 : 0
//	T[p,b] = clamp(Q[p] - G[b] - r, 0, 15)
func truncationPosition(q, r, gain, priority uint8) uint8 {
	refine := int32(0)
	if priority < r {
		refine = 1
	}
	t := int32(q) - int32(gain) - refine
	if t < 0 {
		t = 0
	}
	if t > 15 {
		t = 15
	}
	return uint8(t)
}

// precinctTruncation computes every band's truncation position at once.
func precinctTruncation(geom *precinctGeometry, header *precinctHeader) []uint8 {
	out := make([]uint8, len(geom.bands))
	for i, band := range geom.bands {
		out[i] = truncationPosition(header.q, header.r, band.gain, band.priority)
	}
	return out
}

// parsePrecinctHeader parses the precinct header (Table C.1).
func parsePrecinctHeader(buf []byte, geom *precinctGeometry) (*precinctHeader, error) {
	const lprcMax = (1 << 20) - 1
	const qMax = 31

	r := newBitReader(buf)
	lprc, err := r.readBits(24)
	if err != nil {
		return nil, err
	}
	if lprc == 0 {
		return nil, fmt.Errorf("jpegxs entropy: Lprc[p] must be at least 1")
	}
	if lprc > lprcMax {
		return nil, fmt.Errorf("jpegxs entropy: Lprc[p] = %d exceeds 2^20-1", lprc)
	}
	q, err := r.readBits(8)
	if err != nil {
		return nil, err
	}
	if q > qMax {
		return nil, fmt.Errorf("jpegxs entropy: Q[p] = %d > 31", q)
	}
	rp, err := r.readBits(8)
	if err != nil {
		return nil, err
	}
	// Table C.1: R[p] ∈ [0, NL−1].
	nl := len(geom.bands)
	if nl > 0 && int(rp) > nl-1 {
		return nil, fmt.Errorf("jpegxs entropy: R[p] = %d exceeds NL-1 = %d (Annex C.2 Table C.1)", rp, nl-1)
	}

	d := make([]uint8, len(geom.bands))
	for i, band := range geom.bands {
		if !band.exists {
			continue
		}
		v, err := r.readBits(2)
		if err != nil {
			return nil, err
		}
		d[i] = uint8(v)
	}

	r.alignToByte()
	return &precinctHeader{
		lprc:        lprc,
		q:           uint8(q),
		r:           uint8(rp),
		d:           d,
		headerBytes: r.bytesConsumed(),
	}, nil
}

// parsePacketHeader parses the packet header (Table C.3).
func parsePacketHeader(buf []byte, geom *precinctGeometry) (*packetHeader, error) {
	const shortLdatMax = (1 << 15) - 1
	const shortLcntMax = (1 << 13) - 1
	const shortLsgnMax = (1 << 11) - 1
	const longLdatMax = (1 << 20) - 1
	const longLcntMax = (1 << 20) - 1
	const longLsgnMax = (1 << 15) - 1

	r := newBitReader(buf)
	dr, err := r.readBit()
	if err != nil {
		return nil, err
	}
	short := geom.useShortPacketHeader()
	var ldat, lcnt, lsgn uint32
	if short {
		if ldat, err = r.readBits(15); err != nil {
			return nil, err
		}
		if lcnt, err = r.readBits(13); err != nil {
			return nil, err
		}
		if lsgn, err = r.readBits(11); err != nil {
			return nil, err
		}
		if ldat > shortLdatMax || lcnt > shortLcntMax || lsgn > shortLsgnMax {
			return nil, fmt.Errorf("jpegxs entropy: short packet header field overflow")
		}
	} else {
		if ldat, err = r.readBits(20); err != nil {
			return nil, err
		}
		if lcnt, err = r.readBits(20); err != nil {
			return nil, err
		}
		if lsgn, err = r.readBits(15); err != nil {
			return nil, err
		}
		if ldat > longLdatMax || lcnt > longLcntMax || lsgn > longLsgnMax {
			return nil, fmt.Errorf("jpegxs entropy: long packet header field overflow")
		}
	}
	// The field widths are chosen so the header lands on a byte boundary:
	// 1+15+13+11 = 40 (short) and 1+20+20+15 = 56 (long).
	if !r.atByteBoundary() {
		return nil, fmt.Errorf("jpegxs entropy: packet header did not land on a byte boundary")
	}
	return &packetHeader{
		dr:          dr,
		ldat:        ldat,
		lcnt:        lcnt,
		lsgn:        lsgn,
		shortForm:   short,
		headerBytes: r.bytesConsumed(),
	}, nil
}

// forEachGroupBit reads ng bits and hands them out one at a time, first
// position first.
func forEachGroupBit(r *bitReader, ng int, f func(k int, bit uint32)) error {
	k := 0
	for k < ng {
		n := ng - k
		if n > 32 {
			n = 32
		}
		bits, err := r.readBits(uint8(n))
		if err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			f(k+i, (bits>>uint(n-1-i))&1)
		}
		k += n
	}
	return nil
}

// decodePacketBodyInto decodes one packet body into the precinct's
// running state and returns the number of bytes consumed.
//
// The caller has already parsed the precinct header (T[p,b] and D[p,b])
// and the packet header (Dr, Ldat, Lcnt, Lsgn).
func decodePacketBodyInto(buf []byte, geom *precinctGeometry, precinct *precinctHeader,
	packet *packetHeader, layout *packetLayout, state *precinctState, top *precinctTop) (int, error) {
	truncation := precinctTruncation(geom, precinct)

	// Lazy-initialise the per-band coefficient buffers.
	if len(state.coefficients) == 0 {
		state.coefficients = make([]bandCoefficients, len(geom.bands))
		for i, band := range geom.bands {
			if !band.exists {
				state.coefficients[i] = newBandCoefficients(0, 0, 0)
				continue
			}
			lines := uint32(0)
			if band.l1 > band.l0 {
				lines = uint32(band.l1) - uint32(band.l0)
			}
			state.coefficients[i] = newBandCoefficients(band.wpb, geom.ncg(i), lines)
		}
	}

	totalConsumed := 0

	// === Significance sub-packet =========================================
	{
		if totalConsumed > len(buf) {
			return 0, fmt.Errorf("jpegxs entropy: packet body truncated at sig sub-packet")
		}
		r := newBitReader(buf[totalConsumed:])
		state.sigFlags = state.sigFlags[:0]
		state.sigStart = state.sigStart[:0]
		for _, entry := range layout.entries {
			bi := int(entry.band)
			band := geom.bands[bi]
			state.sigStart = append(state.sigStart, len(state.sigFlags))
			if !band.exists {
				continue
			}
			dpb := precinct.d[bi]
			if packet.dr == 0 && (dpb&2) != 0 {
				ns := int(geom.ns(bi))
				for i := 0; i < ns; i++ {
					z, err := r.readBit()
					if err != nil {
						return 0, err
					}
					state.sigFlags = append(state.sigFlags, z)
				}
			}
		}
		r.alignToByte()
		totalConsumed += r.bytesConsumed()
	}

	// === Bitplane-count sub-packet =======================================
	lcnt := int(packet.lcnt)
	{
		if totalConsumed+lcnt > len(buf) {
			return 0, fmt.Errorf("jpegxs entropy: packet body truncated at bitplane-count sub-packet")
		}
		r := newBitReader(buf[totalConsumed : totalConsumed+lcnt])
		for e, entry := range layout.entries {
			bi := int(entry.band)
			band := geom.bands[bi]
			if !band.exists {
				continue
			}
			lineIndex := int(entry.line - band.l0)
			ncg := int(geom.ncg(bi))
			t := int32(truncation[bi])
			dpb := precinct.d[bi]
			sigStart := state.sigStart[e]

			// Bitplane-count upper bound min(2^Br − 1, 31) — Tables
			// C.12/C.13/C.14.
			brCap := int32(255)
			if geom.br < 8 {
				brCap = (int32(1) << geom.br) - 1
			}
			mMax := brCap
			if mMax > 31 {
				mMax = 31
			}

			// Vertical predictor source line (Table C.11 steps by one
			// band-grid line here).
			const sy = uint16(1)

			coef := &state.coefficients[bi]
			switch {
			case packet.dr == 1:
				// Raw mode: Br bits per code group (Table C.12).
				for g := 0; g < ncg; g++ {
					m, err := r.readBits(geom.br)
					if err != nil {
						return 0, err
					}
					if int32(m) > mMax {
						return 0, fmt.Errorf("jpegxs entropy: raw bitplane count %d exceeds the maximum %d (Table C.12 min(2^Br-1, 31))", m, mMax)
					}
					coef.m[lineIndex*ncg+g] = uint8(m)
				}
			case (dpb & 1) == 0:
				// No-prediction VLC, Table C.14: mtop = T, θ = 0.
				mtop := t
				for g := 0; g < ncg; g++ {
					sigGroup := g / int(geom.ss)
					z := uint8(0)
					if (dpb & 2) != 0 {
						if idx := sigStart + sigGroup; idx < len(state.sigFlags) {
							z = state.sigFlags[idx]
						}
					}
					deltaM := int32(0)
					if (dpb&2) == 0 || z == 0 {
						v, verr := decodeVLCSigned(r, mtop, t)
						if verr != nil {
							return 0, verr
						}
						deltaM = v
					}
					m := mtop + deltaM
					if m < 0 || m > mMax {
						return 0, fmt.Errorf("jpegxs entropy: decoded M[p,λ,b,g] = %d outside 0..=%d (Table C.14 min(2^Br-1, 31))", m, mMax)
					}
					coef.m[lineIndex*ncg+g] = uint8(m)
				}
			default:
				// Vertical prediction VLC, Table C.13.
				firstLineOfBand := entry.line < band.l0+sy
				// Cross-precinct predictor (Table C.11, first branch):
				// the predecessor is the precinct directly above.
				if firstLineOfBand && top == nil {
					return 0, fmt.Errorf("jpegxs entropy: vertical prediction selected at the top line of the topmost precinct of a slice (forbidden by ISO/IEC 21122-1 C.6.1/C.6.3)")
				}
				prevLineIndex := 0
				if !firstLineOfBand {
					prevLineIndex = int((entry.line - sy) - band.l0)
				}
				ttop := t
				if firstLineOfBand {
					if bi < len(top.t) {
						ttop = int32(top.t[bi])
					}
				}
				teff := t
				if ttop > teff {
					teff = ttop
				}
				for g := 0; g < ncg; g++ {
					sigGroup := g / int(geom.ss)
					z := uint8(0)
					if (dpb & 2) != 0 {
						if idx := sigStart + sigGroup; idx < len(state.sigFlags) {
							z = state.sigFlags[idx]
						}
					}
					mAbove := int32(0)
					if firstLineOfBand {
						if bi < len(top.lastM) && g < len(top.lastM[bi]) {
							mAbove = int32(top.lastM[bi][g])
						}
					} else {
						mAbove = int32(coef.m[prevLineIndex*ncg+g])
					}
					mtop := mAbove
					if teff > mtop {
						mtop = teff
					}
					deltaM := int32(0)
					switch {
					case (dpb&2) == 0 || z == 0:
						v, verr := decodeVLCSigned(r, mtop, t)
						if verr != nil {
							return 0, verr
						}
						deltaM = v
					case geom.rm == 0:
						deltaM = 0
					default:
						// Rm == 1 → Δm = T − mtop (so M = T).
						deltaM = t - mtop
					}
					m := mtop + deltaM
					if m < 0 || m > mMax {
						return 0, fmt.Errorf("jpegxs entropy: decoded vertical M[p,λ,b,g] = %d outside 0..=%d (Table C.13 min(2^Br-1, 31))", m, mMax)
					}
					coef.m[lineIndex*ncg+g] = uint8(m)
				}
			}
		}
		r.alignToByte()
		bodyConsumed := r.bytesConsumed()
		if bodyConsumed > lcnt {
			return 0, fmt.Errorf("jpegxs entropy: bitplane-count sub-packet read %d > Lcnt = %d", bodyConsumed, lcnt)
		}
		// Filler bytes follow up to Lcnt.
		totalConsumed += lcnt
	}

	// === Data sub-packet =================================================
	//
	// Table C.8 transmits whole code groups of Ng, so a band whose Wpb is
	// not a multiple of Ng carries meaningless tail coefficients past its
	// right edge. Their magnitudes are discarded, but Table C.9 NOTE 2
	// makes the sign sub-packet carry a sign bit for every tail
	// coefficient whose transmitted magnitude is non-zero — so the tail
	// magnitudes have to be tracked for the sign pass to stay aligned.
	ldat := int(packet.ldat)
	tailV := make([][]uint32, len(layout.entries))
	{
		if totalConsumed+ldat > len(buf) {
			return 0, fmt.Errorf("jpegxs entropy: packet body truncated at data sub-packet")
		}
		r := newBitReader(buf[totalConsumed : totalConsumed+ldat])
		for e, entry := range layout.entries {
			bi := int(entry.band)
			band := geom.bands[bi]
			if !band.exists {
				continue
			}
			lineIndex := int(entry.line - band.l0)
			ncg := int(geom.ncg(bi))
			t := uint32(truncation[bi])
			coef := &state.coefficients[bi]
			lineOffset := lineIndex * int(band.wpb)
			ng := int(geom.ng)
			wpb := int(band.wpb)
			tailLen := ncg*ng - wpb
			if tailLen < 0 {
				tailLen = 0
			}
			tail := make([]uint32, tailLen)
			tailV[e] = tail
			for g := 0; g < ncg; g++ {
				m := uint32(coef.m[lineIndex*ncg+g])
				base := g * ng
				// Initialise the in-band part of the code group.
				start := base
				if start > wpb {
					start = wpb
				}
				end := base + ng
				if end > wpb {
					end = wpb
				}
				if end > start {
					for i := start; i < end; i++ {
						coef.v[lineOffset+i] = 0
					}
				}
				if m > t {
					// Fs == 0: the sign bits come first inside the code
					// group.
					if geom.fs == 0 {
						if err := forEachGroupBit(r, ng, func(k int, bit uint32) {
							pos := base + k
							if pos < wpb {
								coef.s[lineOffset+pos] = uint8(bit)
							}
						}); err != nil {
							return 0, err
						}
					}
					// Bitplanes from MSB (M−1) down to T.
					nplanes := m - t
					for plane := int(nplanes) - 1; plane >= 0; plane-- {
						if err := forEachGroupBit(r, ng, func(k int, bit uint32) {
							pos := base + k
							if pos < wpb {
								coef.v[lineOffset+pos] |= bit << uint(plane)
							} else if pos-wpb < len(tail) {
								tail[pos-wpb] |= bit << uint(plane)
							}
						}); err != nil {
							return 0, err
						}
					}
				}
			}
		}
		r.alignToByte()
		bodyConsumed := r.bytesConsumed()
		if bodyConsumed > ldat {
			return 0, fmt.Errorf("jpegxs entropy: data sub-packet read %d > Ldat = %d", bodyConsumed, ldat)
		}
		totalConsumed += ldat
	}

	// === Sign sub-packet =================================================
	if geom.fs == 1 {
		lsgn := int(packet.lsgn)
		if totalConsumed+lsgn > len(buf) {
			return 0, fmt.Errorf("jpegxs entropy: packet body truncated at sign sub-packet")
		}
		r := newBitReader(buf[totalConsumed : totalConsumed+lsgn])
		for e, entry := range layout.entries {
			bi := int(entry.band)
			band := geom.bands[bi]
			if !band.exists {
				continue
			}
			lineIndex := int(entry.line - band.l0)
			ncg := int(geom.ncg(bi))
			ng := int(geom.ng)
			wpb := int(band.wpb)
			coef := &state.coefficients[bi]
			lineOffset := lineIndex * wpb
			for g := 0; g < ncg; g++ {
				for k := 0; k < ng; k++ {
					pos := g*ng + k
					if pos >= wpb {
						// Meaningless tail coefficient of the last code
						// group (Table C.9 NOTE 2): consume the sign bit
						// when the transmitted magnitude is non-zero so
						// the following signs stay aligned.
						if pos-wpb < len(tailV[e]) && tailV[e][pos-wpb] != 0 {
							if _, err := r.readBit(); err != nil {
								return 0, err
							}
						}
						continue
					}
					if coef.v[lineOffset+pos] != 0 {
						bit, err := r.readBit()
						if err != nil {
							return 0, err
						}
						coef.s[lineOffset+pos] = bit
					}
				}
			}
		}
		r.alignToByte()
		bodyConsumed := r.bytesConsumed()
		if bodyConsumed > lsgn {
			return 0, fmt.Errorf("jpegxs entropy: sign sub-packet read %d > Lsgn = %d", bodyConsumed, lsgn)
		}
		totalConsumed += lsgn
	}

	return totalConsumed, nil
}

// capturePrecinctTop snapshots the vertical-prediction predecessor of a
// precinct that has just been decoded, so the precinct below can predict
// from it (Table C.11).
func capturePrecinctTop(geom *precinctGeometry, precinct *precinctHeader, state *precinctState) *precinctTop {
	truncation := precinctTruncation(geom, precinct)
	top := &precinctTop{
		lastM: make([][]uint8, len(geom.bands)),
		t:     truncation,
	}
	for bi, band := range geom.bands {
		if !band.exists {
			continue
		}
		ncg := int(geom.ncg(bi))
		lastLine := int(band.l1) - int(band.l0) - 1
		if lastLine < 0 {
			lastLine = 0
		}
		row := make([]uint8, ncg)
		if bi < len(state.coefficients) {
			coef := state.coefficients[bi]
			for g := 0; g < ncg; g++ {
				idx := lastLine*ncg + g
				if idx < len(coef.m) {
					row[g] = coef.m[idx]
				}
			}
		}
		top.lastM[bi] = row
	}
	return top
}
