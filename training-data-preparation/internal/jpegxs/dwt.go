package jpegxs

import (
	"fmt"
)

// Forward 5/3 reversible DWT — ISO/IEC 21122-1:2022 Annex E.
//
// This is a port of the encoder-side half of the upstream Rust
// implementation (oxideav-jpegxs `src/dwt.rs`). Only the forward
// direction is ported: the Go package is an encoder, and the upstream
// decoder remains the reference decoder.
//
// The lifting kernels follow Tables E.4 (inverse even/odd) and E.10 /
// E.11 (forward) with the whole-sample symmetric extension of Annex E.6
// folded into the boundary cases, exactly as upstream does.

// maxDim caps the working dimension of a single DWT pass.
const maxDim = 1 << 17

// bandKey is the per-filter-type metadata of Annex B.3: the
// decomposition depth (dx, dy) and the high-pass flags (τx, τy).
type bandKey struct {
	dx   uint32
	dy   uint32
	tauX bool
	tauY bool
}

// betaCount returns the number of wavelet filter types Nβ for a
// (NL,x, NL,y) decomposition (Annex B.3).
func betaCount(nlx, nly uint8) int {
	mn := int(min(nlx, nly))
	mx := int(max(nlx, nly))
	return 2*mn + mx + 1
}

// betaToBandKey maps a filter index β to its (dx, dy, τx, τy) per
// Annex B.3.
func betaToBandKey(beta uint32, nlx, nly uint8) bandKey {
	nlxU := uint32(nlx)
	nlyU := uint32(nly)

	if nlyU == 0 {
		if beta == 0 {
			return bandKey{dx: nlxU, dy: 0, tauX: false, tauY: false}
		}
		return bandKey{dx: nlxU + 1 - beta, dy: 0, tauX: true, tauY: false}
	}

	beta1 := nlxU - nlyU + 1
	if beta < beta1 {
		if beta == 0 {
			return bandKey{dx: nlxU, dy: nlyU, tauX: false, tauY: false}
		}
		return bandKey{dx: nlxU + 1 - beta, dy: nlyU, tauX: true, tauY: false}
	}

	groupIn := beta - beta1
	triple := groupIn / 3
	within := groupIn % 3
	dy := nlyU - triple
	dx := dy
	switch within {
	case 0:
		return bandKey{dx: dx, dy: dy, tauX: true, tauY: false}
	case 1:
		return bandKey{dx: dx, dy: dy, tauX: false, tauY: true}
	default:
		return bandKey{dx: dx, dy: dy, tauX: true, tauY: true}
	}
}

// checkPassLen validates a 1-D pass length against the Annex E.6 NOTE
// minimum and the working cap.
func checkPassLen(z int) error {
	if z < 2 {
		return fmt.Errorf("jpegxs DWT: row width %d below the spec minimum of 2", z)
	}
	if z > maxDim {
		return fmt.Errorf("jpegxs DWT: row width %d out of range (1..=%d)", z, maxDim)
	}
	return nil
}

// analyseLine is the forward lifting of one line: interleaved x →
// low / high (Table E.11).
//
//	x.len() == low.len() + high.len() >= 2
func analyseLine(x []int32, low, high []int32) {
	nl, nh := len(low), len(high)

	// High phase.
	for n := 0; n < nh-1; n++ {
		high[n] = x[2*n+1] - ((x[2*n] + x[2*n+2]) >> 1)
	}
	last := nh - 1
	var right int32
	if nl > nh {
		right = x[2*last+2]
	} else {
		right = x[2*last]
	}
	high[last] = x[2*last+1] - ((x[2*last] + right) >> 1)

	// Low phase.
	low[0] = x[0] + ((high[0] + high[0] + 2) >> 2)
	for n := 1; n < nh; n++ {
		low[n] = x[2*n] + ((high[n-1] + high[n] + 2) >> 2)
	}
	if nl > nh {
		low[nh] = x[2*nh] + ((high[nh-1] + high[nh-1] + 2) >> 2)
	}
}

// forwardHorizontal1D applies the single-row forward horizontal filter
// (Annex E.11, Table E.10).
//
// input.len() is the row width w; low and high receive the low-pass /
// high-pass sub-bands of lengths (w+1)/2 and w/2.
func forwardHorizontal1D(input []int32, low, high []int32) error {
	w := len(input)
	if err := checkPassLen(w); err != nil {
		return err
	}
	lowLen := (w + 1) / 2
	highLen := w / 2
	if len(low) != lowLen || len(high) != highLen {
		return fmt.Errorf("jpegxs DWT: forward horizontal expected low.len()=%d high.len()=%d, got low.len()=%d high.len()=%d",
			lowLen, highLen, len(low), len(high))
	}
	analyseLine(input, low, high)
	return nil
}

// analyseRowsInPlace applies the forward vertical lifting in place: on
// return even rows hold the vertically low-pass lines, odd rows the
// high-pass lines (Table E.11 with whole rows as the unit).
func analyseRowsInPlace(buf []int32, w, h int) {
	// Odd rows: H[n] = X[2n+1] − ((X[2n] + X[2n+2]) >> 1).
	for y := 1; y < h; y += 2 {
		row := buf[y*w : (y+1)*w]
		for i := range row {
			var p, q int32
			if y > 0 {
				p = buf[(y-1)*w+i]
			} else {
				p = row[i]
			}
			if (y+1)*w+i < len(buf) {
				q = buf[(y+1)*w+i]
			} else {
				q = p
			}
			row[i] -= (p + q) >> 1
		}
	}
	// Even rows: L[n] = X[2n] + ((H[n−1] + H[n] + 2) >> 2).
	//
	// The neighbour choice mirrors upstream's match on (above, below):
	// at the top edge the row below substitutes for the missing row
	// above, at the bottom edge the row above substitutes for the
	// missing row below.
	for y := 0; y < h; y += 2 {
		row := buf[y*w : (y+1)*w]
		hasAbove := y > 0
		hasBelow := y+1 < h
		for i := range row {
			var p, q int32
			switch {
			case hasAbove && hasBelow:
				p = buf[(y-1)*w+i]
				q = buf[(y+1)*w+i]
			case !hasAbove && hasBelow:
				p = buf[(y+1)*w+i]
				q = buf[(y+1)*w+i]
			case hasAbove && !hasBelow:
				p = buf[(y-1)*w+i]
				q = buf[(y-1)*w+i]
			default:
				p, q = row[i], row[i]
			}
			row[i] += (p + q + 2) >> 2
		}
	}
}

// analyseJointLevel applies one joint level of Table E.8 with validated
// geometry. block (the h × w input) is the workspace of the vertical
// call and is left holding its interleaved output.
func analyseJointLevel(w, h int, block []int32, ll, hl, lh, hh []int32) {
	llW, hlW := (w+1)/2, w/2
	analyseRowsInPlace(block, w, h)
	for y := 0; y < h; y++ {
		src := block[y*w : (y+1)*w]
		n := y / 2
		var low, high []int32
		if y%2 == 0 {
			low = ll[n*llW : (n+1)*llW]
			high = hl[n*hlW : (n+1)*hlW]
		} else {
			low = lh[n*llW : (n+1)*llW]
			high = hh[n*hlW : (n+1)*hlW]
		}
		analyseLine(src, low, high)
	}
}

// forward2D applies the single-level 2-D forward 5/3 DWT.
//
// The four sub-bands have geometry ll = ll_h × ll_w, hl = ll_h × hl_w,
// lh = lh_h × ll_w, hh = lh_h × hl_w, where ll_w = (w+1)/2, hl_w = w/2,
// ll_h = (h+1)/2, lh_h = h/2.
func forward2D(w, h int, input []int32, ll, hl, lh, hh []int32) error {
	if w < 2 || h < 2 {
		return fmt.Errorf("jpegxs DWT: 2-D band dimensions (%dx%d) below the spec minimum of 2x2", w, h)
	}
	if w > maxDim || h > maxDim {
		return fmt.Errorf("jpegxs DWT: 2-D band dimensions (%dx%d) exceed cap %d", w, h, maxDim)
	}
	llW, hlW := (w+1)/2, w/2
	llH, lhH := (h+1)/2, h/2
	if len(input) != w*h ||
		len(ll) != llW*llH ||
		len(hl) != hlW*llH ||
		len(lh) != llW*lhH ||
		len(hh) != hlW*lhH {
		return fmt.Errorf("jpegxs DWT: forward_2d sub-band size mismatch")
	}

	block := make([]int32, len(input))
	copy(block, input)
	analyseJointLevel(w, h, block, ll, hl, lh, hh)
	return nil
}

// cascadeBandDims computes the picture-level dimensions of band β for a
// wc × hc component under (NL,x, NL,y) (Annex B.2).
func cascadeBandDims(wc, hc int, nlx, nly uint8, beta uint32) (int, int) {
	key := betaToBandKey(beta, nlx, nly)
	w := bandDim(wc, key.dx, key.tauX)
	h := bandDim(hc, key.dy, key.tauY)
	return w, h
}

// bandDim computes one axis of the cascade band geometry.
func bandDim(size int, d uint32, tau bool) int {
	if !tau {
		if d == 0 {
			return size
		}
		return (size + (1 << d) - 1) >> d
	}
	denomMinus1 := 1
	if d != 0 {
		denomMinus1 = 1 << (d - 1)
	}
	return ((size + denomMinus1 - 1) / denomMinus1) / 2
}

// forwardCascade2D applies the multi-level 2-D forward 5/3 DWT
// cascade. It returns one band buffer per filter index β (0 .. Nβ−1),
// each sized to the band geometry the inverse expects.
//
// The pipeline mirrors the inverse two-phase decomposition in reverse:
//
//  1. Joint horizontal+vertical levels d = 1 .. min(nlx, nly): each step
//     splits the running LL band into (LL, HL, LH, HH); the HL/LH/HH
//     bands land in their slots and LL feeds the next level.
//  2. Pure horizontal levels dx = min(nlx, nly)+1 .. nlx (only when
//     nlx > nly): a per-row 1-D horizontal split of LL_{dx−1,nly} into
//     (LL, HL).
func forwardCascade2D(wc, hc int, nlx, nly uint8, input []int32) ([][]int32, error) {
	if nlx == 0 && nly == 0 {
		if len(input) != wc*hc {
			return nil, fmt.Errorf("jpegxs DWT forward cascade: input %d != wc*hc %d", len(input), wc*hc)
		}
		return [][]int32{append([]int32(nil), input...)}, nil
	}
	if nly > nlx {
		return nil, fmt.Errorf("jpegxs DWT forward cascade: NL,y=%d > NL,x=%d not supported", nly, nlx)
	}
	if wc < 2 || hc < 2 {
		return nil, fmt.Errorf("jpegxs DWT forward cascade: component dims (%dx%d) below the spec minimum of 2x2", wc, hc)
	}
	if wc > maxDim || hc > maxDim {
		return nil, fmt.Errorf("jpegxs DWT forward cascade: component dims (%dx%d) exceed cap %d", wc, hc, maxDim)
	}
	if len(input) != wc*hc {
		return nil, fmt.Errorf("jpegxs DWT forward cascade: input %d != wc*hc %d", len(input), wc*hc)
	}

	nbeta := betaCount(nlx, nly)
	layout := make([]bandKey, nbeta)
	for beta := 0; beta < nbeta; beta++ {
		layout[beta] = betaToBandKey(uint32(beta), nlx, nly)
	}
	findBand := func(dx, dy uint32, tx, ty bool) (int, bool) {
		for i, k := range layout {
			if k.dx == dx && k.dy == dy && k.tauX == tx && k.tauY == ty {
				return i, true
			}
		}
		return 0, false
	}

	bandsOut := make([][]int32, nbeta)
	for beta := 0; beta < nbeta; beta++ {
		bw, bh := cascadeBandDims(wc, hc, nlx, nly, uint32(beta))
		bandsOut[beta] = make([]int32, bw*bh)
	}

	nlxU := uint32(nlx)
	nlyU := uint32(nly)
	dxyMin := min(nlxU, nlyU)

	currentLL := append([]int32(nil), input...)
	curW, curH := wc, hc

	// Phase A — joint levels.
	for d := uint32(1); d <= dxyMin; d++ {
		hlB, ok := findBand(d, d, true, false)
		if !ok {
			return nil, fmt.Errorf("jpegxs DWT forward cascade: no HL_{d,d} band at d=%d", d)
		}
		lhB, ok := findBand(d, d, false, true)
		if !ok {
			return nil, fmt.Errorf("jpegxs DWT forward cascade: no LH_{d,d} band at d=%d", d)
		}
		hhB, ok := findBand(d, d, true, true)
		if !ok {
			return nil, fmt.Errorf("jpegxs DWT forward cascade: no HH_{d,d} band at d=%d", d)
		}

		llW := (curW + 1) / 2
		hlWD := curW / 2
		llH := (curH + 1) / 2
		lhH := curH / 2
		if curW < 2 || curH < 2 {
			return nil, fmt.Errorf("jpegxs DWT: 2-D band dimensions (%dx%d) below the spec minimum of 2x2", curW, curH)
		}

		llBuf := make([]int32, llW*llH)
		hlBuf := make([]int32, hlWD*llH)
		lhBuf := make([]int32, llW*lhH)
		hhBuf := make([]int32, hlWD*lhH)

		analyseJointLevel(curW, curH, currentLL, llBuf, hlBuf, lhBuf, hhBuf)

		bandsOut[hlB] = hlBuf
		bandsOut[lhB] = lhBuf
		bandsOut[hhB] = hhBuf
		currentLL = llBuf
		curW, curH = llW, llH
	}

	// Phase B — pure horizontal levels (only when nlx > nly).
	for dxCur := dxyMin + 1; dxCur <= nlxU; dxCur++ {
		hlB, ok := findBand(dxCur, nlyU, true, false)
		if !ok {
			return nil, fmt.Errorf("jpegxs DWT forward cascade: no HL_{dx=%d,dy=%d} band", dxCur, nlyU)
		}
		nextW := (curW + 1) / 2
		hlWD := curW / 2
		nextLL := make([]int32, nextW*curH)
		hlBuf := make([]int32, hlWD*curH)
		for y := 0; y < curH; y++ {
			if err := forwardHorizontal1D(
				currentLL[y*curW:(y+1)*curW],
				nextLL[y*nextW:(y+1)*nextW],
				hlBuf[y*hlWD:(y+1)*hlWD],
			); err != nil {
				return nil, err
			}
		}
		bandsOut[hlB] = hlBuf
		currentLL = nextLL
		curW = nextW
	}

	// Final LL → LL_{nlx,nly} band slot.
	llB, ok := findBand(nlxU, nlyU, false, false)
	if !ok {
		return nil, fmt.Errorf("jpegxs DWT forward cascade: no LL_{NL,x=%d,NL,y=%d} band", nlx, nly)
	}
	bandsOut[llB] = currentLL
	return bandsOut, nil
}

// fqDownscaleBand applies the Annex E.3 / Table E.13 forward fractional
// down-scaling `c = sign(T) * ((|T| + r) >> Fq)` with `r = (1 << Fq) >> 1`.
//
// It is the exact inverse of the decoder's `c << Fq` reconstruction step
// and a no-op for Fq = 0, so the lossless / integer-transform path is
// byte-identical.
func fqDownscaleBand(band []int32, fq uint8) {
	if fq == 0 {
		return
	}
	fqU := uint32(fq)
	r := int32((1 << fqU) >> 1)
	for i, v := range band {
		mag := v
		if mag < 0 {
			mag = -mag
		}
		mag = (mag + r) >> fqU
		if v < 0 {
			mag = -mag
		}
		band[i] = mag
	}
}
