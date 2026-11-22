package jpegxs

import (
	"fmt"
)

// Star-Tetrix colour transform (Cpih = 3) — ISO/IEC 21122-1:2022
// Annex F, and the Component Registration marker tables it depends on.
//
// Port of the encoder-side half of the upstream Rust implementation
// (oxideav-jpegxs `src/colour_transform.rs`, `src/crg.rs`).

// crgDisplacement returns the component displacement vector (δx, δy) for
// component c under CFA pattern type Ct ∈ {0, 1} (Table F.10).
func crgDisplacement(ct uint8, c int) (dx, dy uint8, ok bool) {
	// Table F.10 — (δx, δy) for c = 0..3.
	ct0 := [4][2]uint8{{0, 1}, {1, 1}, {0, 0}, {1, 0}}
	ct1 := [4][2]uint8{{1, 1}, {0, 1}, {1, 0}, {0, 0}}
	if c < 0 || c >= 4 {
		return 0, 0, false
	}
	switch ct {
	case 0:
		return ct0[c][0], ct0[c][1], true
	case 1:
		return ct1[c][0], ct1[c][1], true
	default:
		return 0, 0, false
	}
}

// crgComponentAt is the inverse k[δx, δy] mapping (Table F.11).
func crgComponentAt(ct uint8, dx, dy uint8) (uint8, bool) {
	switch ct {
	case 0:
		switch {
		case dx == 0 && dy == 1:
			return 0, true
		case dx == 1 && dy == 1:
			return 1, true
		case dx == 0 && dy == 0:
			return 2, true
		case dx == 1 && dy == 0:
			return 3, true
		}
	case 1:
		switch {
		case dx == 1 && dy == 1:
			return 0, true
		case dx == 0 && dy == 1:
			return 1, true
		case dx == 1 && dy == 0:
			return 2, true
		case dx == 0 && dy == 0:
			return 3, true
		}
	}
	return 0, false
}

// floorDiv divides n by a positive d, rounding toward -∞. It differs
// from Go's truncating division only for negative n; the Star-Tetrix
// formulas use the floor convention.
func floorDiv(n, d int32) int32 {
	q := n / d
	if n%d != 0 && (n < 0) != (d < 0) {
		return q - 1
	}
	return q
}

// stOffset caches the constant part of one (component, rx, ry) sample
// offset of the Star-Tetrix access function.
//
// The component mapping of Annex F.5.7 Table F.12 depends only on the
// parity of (rx + δx) and (ry + δy) — both of which are fixed for a
// given offset — except that the horizontal and vertical reflections can
// flip rx / ry. All four reflection combinations are therefore resolved
// once, up front.
type stOffset struct {
	dx, dy int32
	// qx and qy hold the (qx, qy) displacement parity per reflection
	// combination; index 2 is "rx reflected", index 1 is "ry reflected".
	qx [4]uint8
	qy [4]uint8
	// out holds the component index per reflection combination.
	out [4]int
}

// starTetrix caches the per-image Star-Tetrix configuration for the
// inner access helper.
type starTetrix struct {
	wf int
	hf int
	cf uint8
	ct uint8
	// off holds the per (component, rx, ry) lookup table.
	off [4 * 3 * 3]stOffset
}

// newStarTetrix builds the access table for one image.
func newStarTetrix(wf, hf int, e1, e2, ct, cf uint8) (*starTetrix, error) {
	var delta [4][2]uint8
	for c := range delta {
		dx, dy, ok := crgDisplacement(ct, c)
		if !ok {
			return nil, fmt.Errorf("jpegxs forward Star-Tetrix: no displacement for c=%d, Ct=%d", c, ct)
		}
		delta[c] = [2]uint8{dx, dy}
	}
	st := &starTetrix{wf: wf, hf: hf, cf: cf, ct: ct}
	for c := 0; c < 4; c++ {
		dx := int32(delta[c][0])
		dy := int32(delta[c][1])
		for rx := int32(-1); rx <= 1; rx++ {
			for ry := int32(-1); ry <= 1; ry++ {
				o := &st.off[c*9+(int(rx)+1)*3+(int(ry)+1)]
				o.dx, o.dy = dx, dy
				for reflX := 0; reflX < 2; reflX++ {
					for reflY := 0; reflY < 2; reflY++ {
						rxEff, ryEff := rx, ry
						if reflX == 1 {
							rxEff = -rx
						}
						if reflY == 1 {
							ryEff = -ry
						}
						// rem_euclid(2) of a signed value equals its low
						// bit in two's complement.
						qx := uint8((rxEff + dx) & 1)
						qy := uint8((ryEff + dy) & 1)
						idx := reflX*2 + reflY
						o.qx[idx] = qx
						o.qy[idx] = qy
						cOut, ok := crgComponentAt(ct, qx, qy)
						if !ok {
							// Upstream panics here; the table is total
							// for Ct ∈ {0,1}.
							panic("jpegxs: crg component_at undefined for the requested displacement")
						}
						o.out[idx] = int(cOut)
					}
				}
			}
		}
	}
	return st, nil
}

// access returns the triple (c', x', y') of the sub-pixel offset
// (rx, ry) from component c at sample position (x, y) — Annex F.5.7
// Table F.12.
func (st *starTetrix) access(c, x, y int, rx, ry int32) (int, int, int) {
	o := &st.off[c*9+(int(rx)+1)*3+(int(ry)+1)]
	twoX := int32(2 * x)
	twoY := int32(2 * y)

	// Horizontal reflection guard (always on).
	reflX := twoX+rx+o.dx < 0 || twoX+rx+o.dx >= int32(2*st.wf)
	// Vertical reflection guard. Two flavours:
	//   * Cf == 3 — restrict to the same line (reflect any out-of-line
	//     access back into the line).
	//   * Otherwise — only reflect when past the sample-grid edges.
	reflY := false
	if st.cf == 3 && (ry+o.dy < 0 || ry+o.dy > 1) {
		reflY = true
	}
	if !reflY && (twoY+ry+o.dy < 0 || twoY+ry+o.dy >= int32(2*st.hf)) {
		reflY = true
	}
	if reflX {
		rx = -rx
	}
	if reflY {
		ry = -ry
	}
	idx := 0
	if reflX {
		idx |= 2
	}
	if reflY {
		idx |= 1
	}
	// After reflection both sums are non-negative, so the shift is an
	// exact halving.
	xOut := int((twoX + rx + o.dx) >> 1)
	yOut := int((twoY + ry + o.dy) >> 1)
	return o.out[idx], xOut, yOut
}

// read fetches the sample at access(c, x, y, rx, ry).
func (st *starTetrix) read(planes *[4][]int32, c, x, y int, rx, ry int32) int32 {
	cOut, xOut, yOut := st.access(c, x, y, rx, ry)
	return planes[cOut][yOut*st.wf+xOut]
}

// forwardStarTetrix applies the forward Star-Tetrix transform to the
// first four component planes.
//
// The input order is the component order of the codestream
// ([R, G1, G2, B] for an RGGB registration); on return the planes hold
// the transform's four output bands [Ya, Cb, Cr, Δ].
func forwardStarTetrix(planes [][]int32, wf, hf int, e1, e2, ct, cf uint8) error {
	// Accept any plane count ≥ 4 so CWD-suppressed trailing components
	// ride through unchanged (Annex F.2 Table F.1 — Ω = O for c ≥ 4 when
	// Cpih = 3).
	if len(planes) < 4 {
		return fmt.Errorf("jpegxs Cpih=3 (forward Star-Tetrix) requires at least 4 component planes, got %d", len(planes))
	}
	if ct > 1 {
		return fmt.Errorf("jpegxs forward Star-Tetrix: Ct=%d reserved for ISO/IEC use (Table F.9)", ct)
	}
	if cf != 0 && cf != 3 {
		return fmt.Errorf("jpegxs forward Star-Tetrix: Cf=%d reserved for ISO/IEC use (Table A.20)", cf)
	}
	want := wf * hf
	// Only the first four component planes participate in the transform.
	for i := 0; i < 4; i++ {
		if len(planes[i]) != want {
			return fmt.Errorf("jpegxs forward Star-Tetrix: component %d has %d samples, expected %d", i, len(planes[i]), want)
		}
	}
	st, err := newStarTetrix(wf, hf, e1, e2, ct, cf)
	if err != nil {
		return err
	}

	// One backing array for the six intermediate band buffers.
	buf := make([]int32, 6*want)
	w3_1 := buf[0*want : 1*want]
	w3_2 := buf[1*want : 2*want]
	w2_0 := buf[2*want : 3*want]
	w2_3 := buf[3*want : 4*want]
	w1_3 := buf[4*want : 5*want]
	o0 := buf[5*want : 6*want]

	// 1) Re-order the input planes [R, G1, G2, B] into the band-position
	//    layout [G2, B, R, G1].
	w4 := [4][]int32{planes[2], planes[3], planes[0], planes[1]}

	// 2) Forward inverse-of-CbCr-step: subtract the green-neighbour
	//    averages from blue (band 1) and red (band 2).
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			gl := st.read(&w4, 1, x, y, -1, 0)
			gr := st.read(&w4, 1, x, y, 1, 0)
			gt := st.read(&w4, 1, x, y, 0, -1)
			gb := st.read(&w4, 1, x, y, 0, 1)
			w3_1[y*wf+x] = w4[1][y*wf+x] - floorDiv(gl+gr+gt+gb, 4)

			gl = st.read(&w4, 2, x, y, -1, 0)
			gr = st.read(&w4, 2, x, y, 1, 0)
			gt = st.read(&w4, 2, x, y, 0, -1)
			gb = st.read(&w4, 2, x, y, 0, 1)
			w3_2[y*wf+x] = w4[2][y*wf+x] - floorDiv(gl+gr+gt+gb, 4)
		}
	}
	// After step 2: w3 = [G2 (band 0), Cb (band 1), Cr (band 2), G1 (band 3)].
	w3 := [4][]int32{w4[0], w3_1, w3_2, w4[3]}

	// 3) Forward inverse-of-Y-step: add the chroma weighted neighbour
	//    sum back into G2 (band 0) and G1 (band 3).
	twoE1 := int32(1) << e1
	twoE2 := int32(1) << e2
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			bl := st.read(&w3, 0, x, y, -1, 0)
			br := st.read(&w3, 0, x, y, 1, 0)
			rt := st.read(&w3, 0, x, y, 0, -1)
			rb := st.read(&w3, 0, x, y, 0, 1)
			num := twoE2*(bl+br) + twoE1*(rt+rb)
			w2_0[y*wf+x] = w3[0][y*wf+x] + floorDiv(num, 8)

			bt := st.read(&w3, 3, x, y, 0, -1)
			bb := st.read(&w3, 3, x, y, 0, 1)
			rl := st.read(&w3, 3, x, y, -1, 0)
			rr := st.read(&w3, 3, x, y, 1, 0)
			num = twoE2*(bt+bb) + twoE1*(rl+rr)
			w2_3[y*wf+x] = w3[3][y*wf+x] + floorDiv(num, 8)
		}
	}
	// After step 3: w2 = [Ya-pre (band 0), Cb, Cr, Δ-pre (band 3)].
	w2 := [4][]int32{w2_0, w3[1], w3[2], w2_3}

	// 4) Forward inverse-of-delta-step: subtract the Y corner-neighbour
	//    average from the delta band (band 3).
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			ylt := st.read(&w2, 3, x, y, -1, -1)
			yrt := st.read(&w2, 3, x, y, 1, -1)
			ylb := st.read(&w2, 3, x, y, -1, 1)
			yrb := st.read(&w2, 3, x, y, 1, 1)
			w1_3[y*wf+x] = w2[3][y*wf+x] - floorDiv(ylt+yrt+ylb+yrb, 4)
		}
	}
	// After step 4: w1 = [Ya, Cb, Cr, Δ].
	w1 := [4][]int32{w2[0], w2[1], w2[2], w1_3}

	// 5) Forward inverse-of-average-step: add the Δ corner-neighbour
	//    average back into the average luma band (band 0).
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			dlt := st.read(&w1, 0, x, y, -1, -1)
			drt := st.read(&w1, 0, x, y, 1, -1)
			dlb := st.read(&w1, 0, x, y, -1, 1)
			drb := st.read(&w1, 0, x, y, 1, 1)
			o0[y*wf+x] = w1[0][y*wf+x] + floorDiv(dlt+drt+dlb+drb, 8)
		}
	}

	// O[0] = Ya (average luma), O[1] = Cb, O[2] = Cr, O[3] = Δ.
	copy(planes[0], o0)
	copy(planes[1], w1[1])
	copy(planes[2], w1[2])
	copy(planes[3], w1_3)
	return nil
}
