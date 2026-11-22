package jpegxs

import (
	"fmt"
)

// Cascade path: the multi-level DWT (NL,x > 1 or NL,y ≥ 1) with the
// per-precinct packet planner of Annex B.7 Table B.4.
//
// Port of the upstream Rust implementation's `write_slice` cascade branch
// and `encode_precinct_cascade` (oxideav-jpegxs `src/encoder.rs`).

// cascadeSlice describes one (β_pic, component) band slice of a
// precinct.
type cascadeSlice struct {
	// wpb is the per-precinct band width Wpb[p,b].
	wpb int
	// lines is the number of band rows this precinct holds.
	lines int
	// picBW is the picture-level band width.
	picBW int
	// picRowOffset is the band-grid row offset of this precinct.
	picRowOffset int
	// picColOffset is the band-grid column offset of this precinct.
	picColOffset int
	// compI is the component index.
	compI int
	// beta is the picture-level β slot (bitstream ordering and band index).
	beta uint32
	// localBeta is the chroma-local β — index into the component's own DWT
	// cascade output.
	localBeta uint32
	// exists reports whether this band exists for this component.
	exists bool
}

// packetJob is one packet's entries plus the (component, β) coordinate of
// each entry.
type packetJob struct {
	entries []*perBandEntry
	// coords holds (compIdx, beta) per entry.
	coords [][2]uint32
	// firstLineInPrecinct is true when at least one entry belongs to a band
	// that has not been seen yet in this precinct, so no in-precinct
	// vertical-prediction predecessor exists.
	firstLineInPrecinct bool
}

// jobForms holds every candidate form of one packet.
type jobForms struct {
	raw       *packetBytes
	noPred    *packetBytes
	noPredSig *packetBytes
	vert      *packetBytes
	vertSig   *packetBytes
}

// bandCoord keys the per-band maps.
type bandCoord struct {
	comp int
	beta uint32
}

// powH is the per-precinct band-row offset count 2^max(NL,y − dy, 0).
func powH(nly uint8, dy uint32) int {
	nlyU := uint32(nly)
	if dy >= nlyU || nlyU == 0 {
		return 1
	}
	return 1 << (nlyU - dy)
}

// writeSliceCascade emits the slices of a multi-level picture using the
// picture-level DWT cascade.
func writeSliceCascade(out *[]byte, cfg *encodeConfig, compPlanes [][]int32,
	npY, npX, hslRows int, writeSLH func(out *[]byte, t uint16)) error {
	w := int(cfg.width)
	h := int(cfg.height)
	nc := int(cfg.nc)
	nDecomposed := int(cfg.nc - cfg.sd)

	// Per-component forward DWT cascade.
	bandsPerComp := make([][][]int32, nc)
	for i := 0; i < nc; i++ {
		if i >= nDecomposed {
			// Suppressed (Sd) — no wavelet bands.
			bandsPerComp[i] = nil
			continue
		}
		wc := (w + int(cfg.sx[i]) - 1) / int(cfg.sx[i])
		hc := (h + int(cfg.sy[i]) - 1) / int(cfg.sy[i])
		nlyI := effectiveNLY(cfg.nly, cfg.sy[i])
		bands, err := forwardCascade2D(wc, hc, cfg.nlx, nlyI, compPlanes[i])
		if err != nil {
			return err
		}
		// Annex E.3 / Table E.13 forward fractional down-scaling.
		for b := range bands {
			fqDownscaleBand(bands[b], cfg.fq)
		}
		bandsPerComp[i] = bands
	}
	// Suppressed components are still subject to the Annex E.3 down-scale.
	if cfg.fq != 0 {
		for i := nDecomposed; i < nc; i++ {
			fqDownscaleBand(compPlanes[i], cfg.fq)
		}
	}

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
		cs = 8 * uint32(cfg.cw) * maxSx * (1 << cfg.nlx)
	}

	t := uint16(0)
	pyStart := 0
	for pyStart < npY {
		pyEnd := min(pyStart+hslRows, npY)
		writeSLH(out, t)
		for py := pyStart; py < pyEnd; py++ {
			for px := 0; px < npX; px++ {
				pbytes, err := encodePrecinctCascade(cfg, bandsPerComp, compPlanes, py, px, cs)
				if err != nil {
					return err
				}
				*out = append(*out, pbytes...)
			}
		}
		pyStart = pyEnd
		t++
	}
	return nil
}

// encodePrecinctCascade encodes one precinct using the multi-level
// cascade band layout.
func encodePrecinctCascade(cfg *encodeConfig, bandsPerComp [][][]int32, compPlanes [][]int32,
	py, px int, cs uint32) ([]byte, error) {
	w := int(cfg.width)
	h := int(cfg.height)
	nc := int(cfg.nc)
	nlx := cfg.nlx
	nly := cfg.nly
	nbetaPic := int(nBeta(nlx, nly))
	npX := int((uint32(w) + cs - 1) / cs)

	// Per-component "effective" decomposition levels.
	nlyI := make([]uint8, nc)
	for i := 0; i < nc; i++ {
		nlyI[i] = effectiveNLY(cfg.nly, cfg.sy[i])
	}

	sdU := int(cfg.sd)
	nDecomposed := nc - sdU

	// Collect the per-(β_pic, i) slices of this precinct.
	slices := make([]cascadeSlice, 0, nbetaPic*nDecomposed+sdU)
	for beta := 0; beta < nbetaPic; beta++ {
		for i := 0; i < nDecomposed; i++ {
			wc := (w + int(cfg.sx[i]) - 1) / int(cfg.sx[i])
			hc := (h + int(cfg.sy[i]) - 1) / int(cfg.sy[i])
			localBeta, ok := pictureBetaToLocalBeta(uint32(beta), cfg.nlx, cfg.nly, cfg.sy[i])
			if !ok {
				slices = append(slices, cascadeSlice{compI: i, beta: uint32(beta)})
				continue
			}
			key := betaToBandKey(localBeta, cfg.nlx, nlyI[i])
			picBW, picBH := cascadeBandDims(wc, hc, cfg.nlx, nlyI[i], localBeta)
			powEff := max(powH(nlyI[i], key.dy), 1)
			rowOffset := py * powEff
			lines := 0
			if rowOffset < picBH {
				lines = min(powEff, picBH-rowOffset)
			}
			sxI := int(cfg.sx[i])
			dx := int(key.dx)
			colsPerUniform := int(cs) / max(sxI*(1<<dx), 1)
			picColOffset := px * colsPerUniform
			remainingCols := max(picBW-picColOffset, 0)
			wpbThis := remainingCols
			if px+1 < npX {
				wpbThis = min(colsPerUniform, remainingCols)
			}
			slices = append(slices, cascadeSlice{
				wpb:          wpbThis,
				lines:        lines,
				picBW:        picBW,
				picRowOffset: rowOffset,
				picColOffset: picColOffset,
				compI:        i,
				beta:         uint32(beta),
				localBeta:    localBeta,
				exists:       true,
			})
		}
	}

	// Sd tail slices: one per suppressed component (β = 0, no DWT).
	hpPic := 1
	if nly != 0 {
		hpPic = 1 << nly
	}
	picRowOffsetSD := py * hpPic
	linesThisPrecinct := min(hpPic, max(h-picRowOffsetSD, 0))
	wpThis := int(cs)
	if px+1 >= npX {
		wpThis = int((uint32(w)-1)%cs + 1)
	}
	picColOffsetSD := px * int(cs)
	for sdIdx := 0; sdIdx < sdU; sdIdx++ {
		i := nDecomposed + sdIdx
		slices = append(slices, cascadeSlice{
			wpb:          min(wpThis, max(w-picColOffsetSD, 0)),
			lines:        linesThisPrecinct,
			picBW:        w,
			picRowOffset: picRowOffsetSD,
			picColOffset: picColOffsetSD,
			compI:        i,
			beta:         0,
			localBeta:    0,
			exists:       linesThisPrecinct > 0 && wpThis > 0,
		})
	}

	// Precinct header: Lprc(24) + Q(8) + R(8) + N_existing × D(2), padded
	// to a byte boundary.
	nExisting := 0
	for _, s := range slices {
		if s.exists {
			nExisting++
		}
	}
	headerBits := 24 + 8 + 8 + 2*nExisting
	headerBytes := (headerBits + 7) / 8
	precinctBytes := make([]byte, headerBytes)

	nlxU := uint32(nlx)
	nlyU := uint32(nly)
	beta1 := int(max(nlxU, nlyU) - min(nlxU, nlyU) + 1)

	// Per-band truncation: T[p,b] = clamp(Q − G[b] − r, 0, 15) per Annex
	// C.6.2 Table C.10.
	ndU32 := uint32(nDecomposed)
	nTotalBands := int(ndU32*uint32(nbetaPic)) + int(cfg.sd)
	existingSlotOfBand := make([]int, nTotalBands)
	for i := range existingSlotOfBand {
		existingSlotOfBand[i] = -1
	}
	{
		cursor := 0
		for beta := 0; beta < nbetaPic; beta++ {
			for i := 0; i < nDecomposed; i++ {
				if _, ok := pictureBetaToLocalBeta(uint32(beta), cfg.nlx, cfg.nly, cfg.sy[i]); ok {
					b := int(ndU32*uint32(beta)) + i
					existingSlotOfBand[b] = cursor
					cursor++
				}
			}
		}
		for i := 0; i < int(cfg.sd); i++ {
			b := int(ndU32*uint32(nbetaPic)) + i
			existingSlotOfBand[b] = cursor
			cursor++
		}
	}
	nExistingBands := 0
	for _, s := range existingSlotOfBand {
		if s >= 0 {
			nExistingBands++
		}
	}
	customWgt := len(cfg.bandGains) > 0 &&
		len(cfg.bandGains) == len(cfg.bandPriorities) &&
		len(cfg.bandGains) == nExistingBands

	tForBand := func(beta uint32, compI int) uint8 {
		var bandIndex uint32
		if compI < nDecomposed {
			bandIndex = ndU32*beta + uint32(compI)
		} else {
			bandIndex = ndU32*uint32(nbetaPic) + uint32(compI) - ndU32
		}
		if customWgt {
			if int(bandIndex) < len(existingSlotOfBand) {
				if slot := existingSlotOfBand[bandIndex]; slot >= 0 {
					g := int32(cfg.bandGains[slot])
					p := uint32(cfg.bandPriorities[slot])
					refine := int32(0)
					if p < uint32(cfg.rp) {
						refine = 1
					}
					return clampByte(int32(cfg.q) - g - refine)
				}
			}
		}
		refine := int32(0)
		if bandIndex < uint32(cfg.rp) {
			refine = 1
		}
		gain := int32(0)
		if compI < nDecomposed {
			localBeta, ok := pictureBetaToLocalBeta(beta, cfg.nlx, cfg.nly, cfg.sy[compI])
			if ok {
				key := betaToBandKey(localBeta, cfg.nlx, nlyI[compI])
				if key.tauX {
					gain++
				}
				if key.tauY {
					gain++
				}
			}
		}
		return clampByte(int32(cfg.q) - gain - refine)
	}

	// extractBandLine builds a one-line band slice from a per-component
	// band buffer.
	extractBandLine := func(s *cascadeSlice, lineOff int) ([]int32, bool) {
		if !s.exists || s.lines == 0 {
			return nil, false
		}
		if lineOff >= s.lines {
			return nil, false
		}
		// Wpb[p,b] = 0 still contributes an ENTRY to its packet: the
		// inclusion test is about line presence, and the Ncg = 0 group
		// count just makes every sub-packet contribution empty.
		if s.wpb == 0 {
			return []int32{}, true
		}
		picRow := s.picRowOffset + lineOff
		rowStart := picRow*s.picBW + s.picColOffset
		rowEnd := rowStart + s.wpb
		if s.compI >= nDecomposed {
			plane := compPlanes[s.compI]
			return plane[rowStart:rowEnd], true
		}
		bandBuf := bandsPerComp[s.compI][s.localBeta]
		return bandBuf[rowStart:rowEnd], true
	}

	// Phase 1 — collect every packet job in slice-walker emission order.
	var jobs []packetJob

	// First packet: β = 0 .. β1−1 × (Nc − Sd) wavelet components × line 0.
	{
		var entries []*perBandEntry
		var coords [][2]uint32
		for beta := 0; beta < beta1; beta++ {
			for i := 0; i < nDecomposed; i++ {
				sIdx := beta*nDecomposed + i
				s := &slices[sIdx]
				lineData, ok := extractBandLine(s, 0)
				if !ok {
					continue
				}
				entries = append(entries, &perBandEntry{
					wpb:  uint32(s.wpb),
					line: lineData,
					t:    tForBand(uint32(beta), i),
				})
				coords = append(coords, [2]uint32{uint32(i), uint32(beta)})
			}
		}
		if len(entries) != 0 {
			jobs = append(jobs, packetJob{
				entries:             entries,
				coords:              coords,
				firstLineInPrecinct: true,
			})
		}
	}

	// Proxy levels: per Annex B.7 Table B.4 the "start a new packet" flag
	// is set once per (λ, β) — before the component loop — so all existing
	// components of a (β, line) are coded JOINTLY in one packet.
	{
		beta0 := beta1
		firstSeen := map[uint32]bool{}
		for beta0 < nbetaPic {
			key0 := betaToBandKey(uint32(beta0), cfg.nlx, cfg.nly)
			powPic := powH(cfg.nly, key0.dy)
			for lambdaWithin := 0; lambdaWithin < powPic; lambdaWithin++ {
				for beta := beta0; beta < min(beta0+3, nbetaPic); beta++ {
					var entries []*perBandEntry
					var coords [][2]uint32
					for i := 0; i < nDecomposed; i++ {
						sIdx := beta*nDecomposed + i
						s := &slices[sIdx]
						if !s.exists {
							continue
						}
						syI := int(cfg.sy[i])
						picGridLine := lambdaWithin
						if syI != 0 && picGridLine%syI != 0 {
							continue
						}
						compLine := picGridLine / max(syI, 1)
						lineData, ok := extractBandLine(s, compLine)
						if !ok {
							continue
						}
						entries = append(entries, &perBandEntry{
							wpb:  uint32(s.wpb),
							line: lineData,
							t:    tForBand(uint32(beta), i),
						})
						coords = append(coords, [2]uint32{uint32(i), uint32(beta)})
					}
					if len(entries) != 0 {
						_, seen := firstSeen[uint32(beta)]
						isFirst := !seen
						firstSeen[uint32(beta)] = true
						jobs = append(jobs, packetJob{
							entries:             entries,
							coords:              coords,
							firstLineInPrecinct: isFirst,
						})
					}
				}
			}
			beta0 += 3
		}
	}

	// Sd tail: one packet per (line λ, suppressed component i).
	if sdU > 0 {
		sdFirstSlice := nbetaPic * nDecomposed
		for lambda := 0; lambda < linesThisPrecinct; lambda++ {
			for sdIdx := 0; sdIdx < sdU; sdIdx++ {
				sIdx := sdFirstSlice + sdIdx
				s := &slices[sIdx]
				if !s.exists {
					continue
				}
				i := nDecomposed + sdIdx
				lineData, ok := extractBandLine(s, lambda)
				if !ok {
					continue
				}
				jobs = append(jobs, packetJob{
					entries: []*perBandEntry{{
						wpb:  uint32(s.wpb),
						line: lineData,
						// Sd tail bands carry raw samples; T = clamp(Q − 0).
						t: clampByte(int32(cfg.q)),
					}},
					coords:              [][2]uint32{{uint32(i), 0}},
					firstLineInPrecinct: lambda == 0,
				})
			}
		}
	}

	// Phase 2 — for every job, evaluate all candidate forms.
	sizesD00 := map[bandCoord]int{}
	sizesD01 := map[bandCoord]int{}
	sizesD10 := map[bandCoord]int{}
	sizesD11 := map[bandCoord]int{}
	mTopCache := map[bandCoord][]uint8{}
	precomputed := make([]jobForms, 0, len(jobs))

	for _, job := range jobs {
		mPerEntry := make([][]uint8, len(job.entries))
		for i, e := range job.entries {
			m, err := computeMPerGroup(cfg, e)
			if err != nil {
				return nil, err
			}
			mPerEntry[i] = m
		}

		data, sgn := buildDataSubpackets(cfg, job.entries, mPerEntry)
		form := func(mode packetMode) (*packetBytes, error) {
			return buildPacketBodyAround(cfg, job.entries, mPerEntry, mode,
				append([]byte(nil), data...), append([]byte(nil), sgn...))
		}
		raw, err := form(packetMode{raw: true})
		if err != nil {
			return nil, err
		}
		noPred, err := form(packetMode{kind: vlcNoPred})
		if err != nil {
			return nil, err
		}
		noPredSig, err := form(packetMode{kind: vlcNoPredSig})
		if err != nil {
			return nil, err
		}

		// Vertical prediction is only attempted when EVERY entry of the
		// packet has a predecessor-line M cached for its (component, β).
		var vertPredecessor [][]uint8
		haveAllPredecessors := !job.firstLineInPrecinct
		if haveAllPredecessors {
			for _, coord := range job.coords {
				key := bandCoord{comp: int(coord[0]), beta: coord[1]}
				prevM, ok := mTopCache[key]
				if !ok {
					haveAllPredecessors = false
					break
				}
				vertPredecessor = append(vertPredecessor, prevM)
			}
		}
		var vert, vertSig *packetBytes
		if haveAllPredecessors {
			vert, err = form(packetMode{kind: vlcVertPred, predecessor: vertPredecessor})
			if err != nil {
				return nil, err
			}
			vertSig, err = form(packetMode{kind: vlcVertPredSig, predecessor: vertPredecessor})
			if err != nil {
				return nil, err
			}
		}

		// Update the cache for downstream packets.
		for entryIdx, coord := range job.coords {
			key := bandCoord{comp: int(coord[0]), beta: coord[1]}
			mTopCache[key] = mPerEntry[entryIdx]
		}

		pickD00 := min(raw.totalLen(), noPred.totalLen())
		pickD01 := min(raw.totalLen(), noPredSig.totalLen())
		pickD10 := raw.totalLen()
		if vert != nil {
			pickD10 = min(raw.totalLen(), vert.totalLen())
		}
		pickD11 := raw.totalLen()
		if vertSig != nil {
			pickD11 = min(raw.totalLen(), vertSig.totalLen())
		}
		for _, coord := range job.coords {
			key := bandCoord{comp: int(coord[0]), beta: coord[1]}
			sizesD00[key] += pickD00
			sizesD01[key] += pickD01
			sizesD10[key] += pickD10
			sizesD11[key] += pickD11
		}
		precomputed = append(precomputed, jobForms{
			raw:       raw,
			noPred:    noPred,
			noPredSig: noPredSig,
			vert:      vert,
			vertSig:   vertSig,
		})
	}

	// Phase 3 — commit D[p,b] ∈ {0,1,2,3} by total bytes, ONE value per
	// packet group.
	groupKey := func(coord bandCoord) uint32 {
		if coord.comp >= nDecomposed {
			return 0x4000_0000 | uint32(coord.comp) // Sd tail: one packet per band
		}
		if coord.beta < uint32(beta1) {
			return ^uint32(0) // first packet: β < β1 of every component
		}
		return coord.beta // proxy level: all components of a β share a packet
	}
	g00 := map[uint32]int{}
	g01 := map[uint32]int{}
	g10 := map[uint32]int{}
	g11 := map[uint32]int{}
	for coord := range sizesD00 {
		g := groupKey(coord)
		g00[g] += sizesD00[coord]
		g01[g] += sizesD01[coord]
		g10[g] += sizesD10[coord]
		g11[g] += sizesD11[coord]
	}
	dPerGroup := map[uint32]uint8{}
	for g, s00 := range g00 {
		s01 := g01[g]
		s10 := g10[g]
		s11 := g11[g]
		best := min(min(s00, s01), min(s10, s11))
		var d uint8
		switch {
		case s11 == best:
			d = 3 // sig=1, pred=1
		case s10 == best:
			d = 1 // sig=0, pred=1
		case s01 == best:
			d = 2 // sig=1, pred=0
		default:
			d = 0 // sig=0, pred=0
		}
		dPerGroup[g] = d
	}
	dPerBand := map[bandCoord]uint8{}
	for coord := range sizesD00 {
		dPerBand[coord] = dPerGroup[groupKey(coord)]
	}

	// Phase 4 — emit the packets in order, picking the form that matches
	// the band's chosen D[p,b].
	var entropy []byte
	for i, job := range jobs {
		forms := precomputed[i]
		dAny := uint8(0)
		for _, c := range job.coords {
			key := bandCoord{comp: int(c[0]), beta: c[1]}
			if d, ok := dPerBand[key]; ok && d > dAny {
				dAny = d
			}
		}
		predBit := dAny & 1
		sigBit := (dAny >> 1) & 1
		var chosen *packetBytes
		switch {
		case predBit == 1 && sigBit == 1:
			chosen = forms.raw
			if forms.vertSig != nil && forms.vertSig.totalLen() <= forms.raw.totalLen() {
				chosen = forms.vertSig
			}
		case predBit == 1:
			chosen = forms.raw
			if forms.vert != nil && forms.vert.totalLen() <= forms.raw.totalLen() {
				chosen = forms.vert
			}
		case sigBit == 1:
			chosen = forms.raw
			if forms.noPredSig.totalLen() <= forms.raw.totalLen() {
				chosen = forms.noPredSig
			}
		default:
			chosen = forms.raw
			if forms.noPred.totalLen() <= forms.raw.totalLen() {
				chosen = forms.noPred
			}
		}
		if err := writePacket(&entropy, chosen); err != nil {
			return nil, err
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
	// D[p,b] bits at offset 5+: two bits per existing band (Sig|Pred).
	bitCursor := 24 + 8 + 8
	for _, s := range slices {
		if !s.exists {
			continue
		}
		d := dPerBand[bandCoord{comp: s.compI, beta: s.beta}]
		writeDBit(precinctBytes, bitCursor, (d>>1)&1)
		bitCursor++
		writeDBit(precinctBytes, bitCursor, d&1)
		bitCursor++
	}
	precinctBytes = append(precinctBytes, entropy...)
	return precinctBytes, nil
}

// writeDBit writes one bit at bitPos (counted MSB-first) into bytes.
func writeDBit(bytes []byte, bitPos int, value uint8) {
	byteIdx := bitPos / 8
	off := 7 - (bitPos % 8)
	if byteIdx < len(bytes) {
		bytes[byteIdx] |= (value & 1) << off
	}
}

// clampByte clamps a signed value into the 0..15 truncation range.
func clampByte(v int32) uint8 {
	if v < 0 {
		v = 0
	}
	if v > 15 {
		v = 15
	}
	return uint8(v)
}
