package jpegxs

import (
	"fmt"
	"math/bits"
)

// Packet assembly — ISO/IEC 21122-1:2022 Annex C.
//
// Port of the encoder-side packet machinery of the upstream Rust
// implementation (oxideav-jpegxs `src/encoder.rs`): the data and sign
// sub-packets (Tables C.8 / C.9), the bitplane-count sub-packet
// (Tables C.12 / C.13 / C.14), the VLC signed codeword (Table C.15) and
// the packet header (Table C.3).

// maxGroup is the upper bound on Ng for the packed data-sub-packet
// writer (the standard fixes Ng = 4).
const maxGroup = 8

// vlcKind selects the bitplane-count coding mode of a packet.
type vlcKind int

const (
	// vlcNoPred is Table C.14 without prediction: mtop = T[p,b], θ = 0.
	vlcNoPred vlcKind = iota
	// vlcNoPredSig adds significance gating (D[p,b] & 2 = 1).
	vlcNoPredSig
	// vlcVertPred is Table C.13: mtop = max(M_above, T), θ = max(M_above−T, 0).
	vlcVertPred
	// vlcVertPredSig adds significance gating to the vertical predictor.
	vlcVertPredSig
)

// packetMode describes how a packet's bitplane counts are coded.
type packetMode struct {
	// raw selects the raw coding mode (Dr = 1) instead of VLC (Dr = 0).
	raw bool
	// kind selects the VLC variant; ignored when raw is set.
	kind vlcKind
	// predecessor holds the per-entry vertical-prediction M arrays.
	predecessor [][]uint8
}

// perBandEntry is one band line emitted in a packet.
type perBandEntry struct {
	// wpb is Wpb[p,b] — coefficients per line in this band.
	wpb uint32
	// line holds one line of wavelet coefficients (length wpb).
	line []int32
	// t is the per-band truncation position T[p,b].
	t uint8
}

// packetBytes is one packet's serialised form.
type packetBytes struct {
	// longHeader selects the long (7-byte) packet header of Table C.3.
	longHeader bool
	// dr is the raw-mode flag.
	dr uint8
	// sig is the significance sub-packet (empty when D[p,b] & 2 = 0).
	sig []byte
	// cnt is the bitplane-count sub-packet.
	cnt []byte
	// data is the data sub-packet.
	data []byte
	// sgn is the sign sub-packet.
	sgn []byte
}

// totalLen is the packet's length on the wire.
func (p *packetBytes) totalLen() int {
	// Header: 1+15+13+11 bits short, 1+20+20+15 bits long.
	header := 5
	if p.longHeader {
		header = 7
	}
	return header + len(p.sig) + len(p.cnt) + len(p.data) + len(p.sgn)
}

// absU32 returns the magnitude of a signed 32-bit value.
func absU32(v int32) uint32 {
	if v < 0 {
		return uint32(-int64(v))
	}
	return uint32(v)
}

// computeMPerGroup computes the per-code-group bitplane counts M[g] for
// one entry, applying the lossy lower bound M ≥ T so the dequantised
// magnitudes stay within the bitstream's M cap.
func computeMPerGroup(cfg *encodeConfig, entry *perBandEntry) ([]uint8, error) {
	wpb := int(entry.wpb)
	bandLine := entry.line
	t := uint32(entry.t)
	ngU := int(cfg.ng)
	ncg := (wpb + ngU - 1) / ngU
	mMaxForBr := uint32(255)
	if cfg.br < 8 {
		mMaxForBr = (1 << cfg.br) - 1
	}
	mPerGroup := make([]uint8, ncg)
	for g := 0; g < ncg; g++ {
		var maxMag uint32
		for k := 0; k < ngU; k++ {
			xpos := g*ngU + k
			var v int32
			if xpos < wpb {
				v = bandLine[xpos]
			}
			if mag := absU32(v); mag > maxMag {
				maxMag = mag
			}
		}
		m := uint32(0)
		if maxMag != 0 {
			m = uint32(bits.Len32(maxMag))
		}
		mEff := m
		if t > mEff {
			mEff = t
		}
		if mEff > mMaxForBr {
			return nil, fmt.Errorf("jpegxs encoder: code group %d bitplane count %d exceeds Br = %d (cap %d)",
				g, mEff, cfg.br, mMaxForBr)
		}
		mPerGroup[g] = uint8(mEff)
	}
	return mPerGroup, nil
}

// forwardQuantIndex maps a raw wavelet coefficient c to the on-wire
// quantization-index magnitude v[p,λ,b,x] for the active inverse
// quantizer type qpih.
//
//   - qpih == 0 — deadzone forward quantizer (Annex D.4, Table D.3):
//     v = |c| >> T.
//   - qpih == 1 — uniform forward quantizer (Annex D.5, Table D.4):
//     with ζ = M−T+1 and d = |c|,
//     v = ((d << ζ) − d + (1 << M)) >> (M + 1).
//
// Both branches return 0 when M ≤ T (the group carries no stored
// bitplanes).
func forwardQuantIndex(qpih uint8, c int32, m, t uint32) uint32 {
	if m <= t {
		return 0
	}
	d := absU32(c)
	if qpih == 1 {
		zeta := m - t + 1
		num := ((uint64(d) << zeta) + (1 << m)) - uint64(d)
		return uint32(num >> (m + 1))
	}
	return d >> t
}

// buildDataSubpackets builds the data sub-packet (Annex C.5.4,
// Table C.8) and the sign sub-packet (Annex C.5.5, Table C.9) of one
// packet. Neither depends on how the bitplane counts are coded.
func buildDataSubpackets(cfg *encodeConfig, entries []*perBandEntry, mPerEntry [][]uint8) (data, sgn []byte) {
	var dataWriter bitWriter
	var sgnWriter bitWriter
	fs1 := cfg.fs == 1
	ngU := int(cfg.ng)
	if ngU > maxGroup {
		ngU = maxGroup
	}
	ngBits := uint8(ngU)

	for i, entry := range entries {
		mPerGroup := mPerEntry[i]
		wpb := int(entry.wpb)
		bandLine := entry.line
		t := uint32(entry.t)
		for g, mU8 := range mPerGroup {
			m := uint32(mU8)
			// Coefficients of the code group; positions past the band
			// width read as zero (Annex C.5.4 NOTE 2).
			var c [maxGroup]int32
			start := min(g*ngU, wpb)
			end := min(start+ngU, wpb)
			copy(c[:end-start], bandLine[start:end])
			// Quantization indices by the active forward quantizer.
			var v [maxGroup]uint32
			for k := 0; k < ngU; k++ {
				v[k] = forwardQuantIndex(cfg.qpih, c[k], m, t)
			}
			if m > t {
				// Fs = 0: Ng sign bits ahead of the magnitudes of every
				// group that stores bitplanes (Table C.8).
				if !fs1 {
					var signs uint32
					for k := 0; k < ngU; k++ {
						signs = (signs << 1) | uint32(boolBit(c[k] < 0))
					}
					dataWriter.writeBits(signs, ngBits)
				}
				// M − T magnitude bitplanes, most significant first.
				for bplane := int(m-t) - 1; bplane >= 0; bplane-- {
					var plane uint32
					for k := 0; k < ngU; k++ {
						plane = (plane << 1) | ((v[k] >> uint(bplane)) & 1)
					}
					dataWriter.writeBits(plane, ngBits)
				}
			}
			// Fs = 1: one sign bit per coefficient whose stored index is
			// non-zero, in decoder read order (Table C.9).
			if fs1 {
				for k := 0; k < ngU; k++ {
					if v[k] != 0 {
						sgnWriter.writeBit(boolBit(c[k] < 0))
					}
				}
			}
		}
	}
	return dataWriter.intoBytes(), sgnWriter.intoBytes()
}

// boolBit converts a bool to the 0/1 the bitstream carries.
func boolBit(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

// buildPacketBodyAround wraps the shared data / sign sub-packets with
// the significance and bitplane-count sub-packets of one coding mode.
func buildPacketBodyAround(cfg *encodeConfig, entries []*perBandEntry, mPerEntry [][]uint8,
	mode packetMode, dataBytes, sgnBytes []byte) (*packetBytes, error) {
	var cntWriter bitWriter
	var sigWriter bitWriter
	ssU := int(cfg.ss) // code groups per significance group

	useSig := !mode.raw && (mode.kind == vlcNoPredSig || mode.kind == vlcVertPredSig)

	// Per-(entry, sig_group) significance flags from the M arrays.
	var sigFlagsPerEntry [][]bool
	if useSig {
		sigFlagsPerEntry = make([][]bool, len(entries))
		for entryIdx, mPerGroup := range mPerEntry {
			entry := entries[entryIdx]
			ncg := len(mPerGroup)
			t := entry.t
			ns := (ncg + ssU - 1) / ssU
			flags := make([]bool, ns)
			for j := 0; j < ns; j++ {
				g0 := j * ssU
				g1 := min(g0+ssU, ncg)
				significant := false
				for _, m := range mPerGroup[g0:g1] {
					if m > t {
						significant = true
						break
					}
				}
				if !significant && cfg.rm != 1 && mode.kind == vlcVertPredSig {
					// Vertical prediction (Table C.13): an insignificant
					// group reconstructs as M = mtop = max(Mtop, T) with
					// Δm = 0 (Rm = 0), and the data sub-packet includes
					// any group with M > T (Table C.8). Flagging such a
					// group insignificant while the predictor exceeds T
					// would inflate the reconstructed M above the
					// emitted data — keep it significant so the VLC
					// carries the true (negative) residual.
					predM := mode.predecessor[entryIdx]
					end := min(g1, len(predM))
					for _, m := range predM[g0:end] {
						if m > t {
							significant = true
							break
						}
					}
				}
				flags[j] = significant
			}
			sigFlagsPerEntry[entryIdx] = flags
		}
	} else {
		sigFlagsPerEntry = make([][]bool, len(entries))
		for i := range sigFlagsPerEntry {
			sigFlagsPerEntry[i] = []bool{}
		}
	}

	// Significance sub-packet: one bit per significance group across all
	// entries, in the same order as the bitplane-count sub-packet.
	if useSig {
		for _, sigFlags := range sigFlagsPerEntry {
			for _, significant := range sigFlags {
				// Table C.5: Z = 1 ⇒ insignificant, Z = 0 ⇒ significant.
				sigWriter.writeBit(boolBit(!significant))
			}
		}
		sigWriter.alignToByte()
	}

	for entryIdx, entry := range entries {
		t := uint32(entry.t)
		mPerGroup := mPerEntry[entryIdx]

		// groupSig reports whether a code group sits in a significant
		// significance group.
		groupSig := func(g int) bool {
			if !useSig {
				return true
			}
			j := g / ssU
			if j < len(sigFlagsPerEntry[entryIdx]) {
				return sigFlagsPerEntry[entryIdx][j]
			}
			return true
		}

		switch {
		case mode.raw:
			// Raw mode: Br bits per code group.
			for _, m := range mPerGroup {
				cntWriter.writeBits(uint32(m), cfg.br)
			}
		case mode.kind == vlcNoPred:
			// mtop = T[p,b]; θ = max(mtop−T, 0) = 0. Δm = M−mtop is
			// always ≥ 0 → unary sub-alphabet → x = Δm.
			for _, m := range mPerGroup {
				deltaM := int32(m) - int32(t)
				emitVLCSigned(&cntWriter, deltaM, 0)
			}
		case mode.kind == vlcNoPredSig:
			// Same as vlcNoPred but skip the VLC for insignificant groups.
			for g, m := range mPerGroup {
				if !groupSig(g) {
					continue
				}
				deltaM := int32(m) - int32(t)
				emitVLCSigned(&cntWriter, deltaM, 0)
			}
		case mode.kind == vlcVertPred:
			// Table C.13: mtop = max(M_above, T), θ = max(mtop−T, 0).
			predM := mode.predecessor[entryIdx]
			if len(predM) != len(mPerGroup) {
				return nil, fmt.Errorf("jpegxs encoder: vertical predictor M length %d != current %d",
					len(predM), len(mPerGroup))
			}
			for g, m := range mPerGroup {
				mAbove := int32(predM[g])
				mtop := mAbove
				if int32(t) > mtop {
					mtop = int32(t)
				}
				theta := mtop - int32(t)
				if theta < 0 {
					theta = 0
				}
				emitVLCSigned(&cntWriter, int32(m)-mtop, theta)
			}
		case mode.kind == vlcVertPredSig:
			predM := mode.predecessor[entryIdx]
			if len(predM) != len(mPerGroup) {
				return nil, fmt.Errorf("jpegxs encoder: vertical predictor M length %d != current %d",
					len(predM), len(mPerGroup))
			}
			for g, m := range mPerGroup {
				if !groupSig(g) {
					continue
				}
				mAbove := int32(predM[g])
				mtop := mAbove
				if int32(t) > mtop {
					mtop = int32(t)
				}
				theta := mtop - int32(t)
				if theta < 0 {
					theta = 0
				}
				emitVLCSigned(&cntWriter, int32(m)-mtop, theta)
			}
		}
	}

	sigBytes := sigWriter.intoBytes()
	cntBytes := cntWriter.intoBytes()
	dr := uint8(0)
	if mode.raw {
		dr = 1
	}
	return &packetBytes{
		longHeader: cfg.longPacketHeaders(),
		dr:         dr,
		sig:        sigBytes,
		cnt:        cntBytes,
		data:       dataBytes,
		sgn:        sgnBytes,
	}, nil
}

// writePacket appends a packet to out.
func writePacket(out *[]byte, pkt *packetBytes) error {
	lcnt := uint64(len(pkt.cnt))
	ldat := uint64(len(pkt.data))
	lsgn := uint64(len(pkt.sgn))
	// Field widths of Table C.3 (Ldat, Lcnt, Lsgn).
	wd, wc, ws := uint32(15), uint32(13), uint32(11)
	if pkt.longHeader {
		wd, wc, ws = 20, 20, 15
	}
	for _, f := range []struct {
		name  string
		len   uint64
		width uint32
	}{
		{"Ldat", ldat, wd},
		{"Lcnt", lcnt, wc},
		{"Lsgn", lsgn, ws},
	} {
		if f.len >= 1<<f.width {
			if pkt.longHeader {
				return fmt.Errorf("jpegxs encoder: %s = %d exceeds long packet header capacity (%d bits).",
					f.name, f.len, f.width)
			}
			return fmt.Errorf("jpegxs encoder: %s = %d %s (%d bits).",
				f.name, f.len, shortHeaderOverflow, f.width)
		}
	}
	var hdrBits uint64 = uint64(pkt.dr & 1)
	hdrBits = (hdrBits << wd) | ldat
	hdrBits = (hdrBits << wc) | lcnt
	hdrBits = (hdrBits << ws) | lsgn
	headerLen := int(1+wd+wc+ws) / 8
	header := make([]byte, headerLen)
	for k := 0; k < headerLen; k++ {
		header[k] = byte(hdrBits >> (8 * uint(headerLen-1-k)))
	}
	*out = append(*out, header...)
	// Significance sub-packet precedes the bitplane-count sub-packet
	// (Annex C.4, Table C.5).
	*out = append(*out, pkt.sig...)
	*out = append(*out, pkt.cnt...)
	*out = append(*out, pkt.data...)
	*out = append(*out, pkt.sgn...)
	return nil
}

// shortHeaderOverflow is the error text of a sub-packet that does not
// fit the short packet header; the driver recognises it to re-run the
// encode with Lh = 1.
const shortHeaderOverflow = "exceeds short packet header capacity"

// emitPacket emits one packet to out.
//
// The packet covers a list of (band, line) entries. The single-level
// (NL=1/1) path uses this — every packet has at most one line per band
// per precinct, so vertical prediction is never available. The cascade
// encoder uses the multi-form pipeline which adds vertical prediction
// and significance coding.
func emitPacket(out *[]byte, cfg *encodeConfig, entries []*perBandEntry) error {
	if len(entries) == 0 {
		return nil
	}
	mPerEntry := make([][]uint8, len(entries))
	for i, e := range entries {
		m, err := computeMPerGroup(cfg, e)
		if err != nil {
			return err
		}
		mPerEntry[i] = m
	}
	data, sgn := buildDataSubpackets(cfg, entries, mPerEntry)

	raw, err := buildPacketBodyAround(cfg, entries, mPerEntry,
		packetMode{raw: true}, append([]byte(nil), data...), append([]byte(nil), sgn...))
	if err != nil {
		return err
	}
	noPred, err := buildPacketBodyAround(cfg, entries, mPerEntry,
		packetMode{kind: vlcNoPred}, data, sgn)
	if err != nil {
		return err
	}
	chosen := raw
	if noPred.totalLen() <= raw.totalLen() {
		chosen = noPred
	}
	return writePacket(out, chosen)
}

// emitVLCSigned emits a VLC-encoded signed value for the predictor
// parameter theta (Annex C.7.1, Table C.15).
//
//	value > theta       → unary sub-alphabet, x = value + theta
//	0 < value <= theta  → signed-binary even codeword, x = 2·value
//	−theta <= value < 0 → signed-binary odd codeword, x = 2·(−value) − 1
//	value == 0          → x = 0 (single 0 bit)
//
// Then x ones are emitted MSB-first followed by a single 0 comma.
func emitVLCSigned(w *bitWriter, value, theta int32) {
	var x uint32
	switch {
	case value > theta:
		x = uint32(value + theta)
	case value > 0:
		x = uint32(2 * value)
	case value == 0:
		x = 0
	default:
		x = uint32(2*(-value) - 1)
	}
	w.writeOnes(x)
	w.writeBit(0)
}
