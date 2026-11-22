package jpegxs

import "fmt"

// Inverse transforms — ISO/IEC 21122-1:2022 Annexes D, E, F and G.
//
// Port of the decoder-side half of the upstream Rust implementation
// (oxideav-jpegxs `src/dequant.rs`, `src/dwt.rs`, `src/colour_transform.rs`
// and `src/output.rs`).

// synthEven is the even-phase step of the inverse 5/3 lifting filter
// (Annex E.3).
func synthEven(x, a, b int32) int32 {
	return x - ((a + b + 2) >> 2)
}

// synthOdd is the odd-phase step of the inverse 5/3 lifting filter.
func synthOdd(x, a, b int32) int32 {
	return x + ((a + b) >> 1)
}

// synthLine reconstructs one line from its low- and high-frequency parts
// (Annex E.3).
func synthLine(low, high, out []int32) {
	nl := len(low)
	nh := len(high)
	if nh == 0 || len(out) != nl+nh {
		return
	}
	// Even phase.
	out[0] = synthEven(low[0], high[0], high[0])
	for n := 1; n < nh; n++ {
		out[2*n] = synthEven(low[n], high[n-1], high[n])
	}
	if nl > nh {
		out[2*nh] = synthEven(low[nh], high[nh-1], high[nh-1])
	}
	// Odd phase.
	for n := 0; n < nh-1; n++ {
		out[2*n+1] = synthOdd(high[n], out[2*n], out[2*n+2])
	}
	last := nh - 1
	right := out[2*last]
	if nl > nh {
		right = out[2*last+2]
	}
	out[2*last+1] = synthOdd(high[last], out[2*last], right)
}

// inverseDeadzone is the inverse of the deadzone quantiser (Annex D.4).
func inverseDeadzone(band *bandCoefficients, t uint8, ng uint8, out []int32) {
	wpb := int(band.wpb)
	ngU := int(ng)
	numLines := int(band.numLines)
	tU := uint32(t)
	for line := 0; line < numLines; line++ {
		mline := 0
		if ngU != 0 {
			mline = (wpb + ngU - 1) / ngU
		}
		for x := 0; x < wpb; x++ {
			g := 0
			if ngU != 0 {
				g = x / ngU
			}
			v := band.v[line*wpb+x]
			var m uint8
			if line*mline+g < len(band.m) {
				m = band.m[line*mline+g]
			}
			sign := band.s[line*wpb+x]
			var c int32
			if uint32(m) > tU && v != 0 {
				r := uint32(1) << tU >> 1
				sigma := int32(1) - 2*int32(sign)
				mag := int32((v << tU) + r)
				c = sigma * mag
			}
			out[line*wpb+x] = c
		}
	}
}

// inverseUniform is the inverse of the uniform quantiser (Annex D.5).
func inverseUniform(band *bandCoefficients, t uint8, ng uint8, out []int32) {
	wpb := int(band.wpb)
	ngU := int(ng)
	numLines := int(band.numLines)
	tU := uint32(t)
	for line := 0; line < numLines; line++ {
		mline := 0
		if ngU != 0 {
			mline = (wpb + ngU - 1) / ngU
		}
		for x := 0; x < wpb; x++ {
			g := 0
			if ngU != 0 {
				g = x / ngU
			}
			v := band.v[line*wpb+x]
			var m uint8
			if line*mline+g < len(band.m) {
				m = band.m[line*mline+g]
			}
			sign := band.s[line*wpb+x]
			var c int32
			if uint32(m) > tU && v != 0 {
				phi := uint64(v) << tU
				zeta := uint32(m) - tU + 1
				var rho uint64
				for phi > 0 {
					rho += phi
					if zeta >= 64 {
						break
					}
					phi >>= zeta
				}
				sigma := int32(1) - 2*int32(sign)
				c = sigma * int32(rho)
			}
			out[line*wpb+x] = c
		}
	}
}

// inverseQuantize applies the inverse of the active quantiser.
func inverseQuantize(qpih uint8, band *bandCoefficients, t uint8, ng uint8, out []int32) {
	if qpih == 1 {
		inverseUniform(band, t, ng, out)
		return
	}
	inverseDeadzone(band, t, ng, out)
}

// inverseStarTetrix applies the inverse Star-Tetrix transform
// (Annex F, Tables F.5 – F.8).
//
// The four planes hold the transform's output bands [Ya, Cb, Cr, Δ] on
// entry and the CFA components [R, G1, G2, B] on return.
func inverseStarTetrix(planes *[4][]int32, wf, hf int, e1, e2, ct, cf uint8) error {
	if ct > 1 {
		return fmt.Errorf("jpegxs decode: Star-Tetrix Ct=%d reserved for ISO/IEC use (Table F.9)", ct)
	}
	if cf != 0 && cf != 3 {
		return fmt.Errorf("jpegxs decode: Star-Tetrix Cf=%d reserved for ISO/IEC use (Table A.20)", cf)
	}
	want := wf * hf
	for i := 0; i < 4; i++ {
		if len(planes[i]) != want {
			return fmt.Errorf("jpegxs decode: Star-Tetrix component %d has %d samples, expected %d", i, len(planes[i]), want)
		}
	}
	st, err := newStarTetrix(wf, hf, e1, e2, ct, cf)
	if err != nil {
		return err
	}

	// Snapshot O[c,x,y] before mutating the planes.
	o := [4][]int32{
		append([]int32(nil), planes[0]...),
		append([]int32(nil), planes[1]...),
		append([]int32(nil), planes[2]...),
		append([]int32(nil), planes[3]...),
	}

	// Step 1 — inverse average (Table F.5):
	//   ω1[0,x,y] = O[0] − ⌊(Δlt + Δrt + Δlb + Δrb) / 8⌋
	w1_0 := make([]int32, want)
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			dlt := st.read(&o, 0, x, y, -1, -1)
			drt := st.read(&o, 0, x, y, 1, -1)
			dlb := st.read(&o, 0, x, y, -1, 1)
			drb := st.read(&o, 0, x, y, 1, 1)
			w1_0[y*wf+x] = o[0][y*wf+x] - floorDiv(dlt+drt+dlb+drb, 8)
		}
	}
	w1 := [4][]int32{w1_0, o[1], o[2], o[3]}

	// Step 2 — inverse delta (Table F.6):
	//   ω2[3,x,y] = ω1[3] + ⌊(Ylt + Yrt + Ylb + Yrb) / 4⌋
	w2_3 := make([]int32, want)
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			ylt := st.read(&w1, 3, x, y, -1, -1)
			yrt := st.read(&w1, 3, x, y, 1, -1)
			ylb := st.read(&w1, 3, x, y, -1, 1)
			yrb := st.read(&w1, 3, x, y, 1, 1)
			w2_3[y*wf+x] = w1[3][y*wf+x] + floorDiv(ylt+yrt+ylb+yrb, 4)
		}
	}
	w2 := [4][]int32{w1[0], w1[1], w1[2], w2_3}

	// Step 3 — inverse Y step (Table F.7).
	twoE1 := int32(1) << e1
	twoE2 := int32(1) << e2
	w3_0 := make([]int32, want)
	w3_3 := make([]int32, want)
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			bl := st.read(&w2, 0, x, y, -1, 0)
			br := st.read(&w2, 0, x, y, 1, 0)
			rt := st.read(&w2, 0, x, y, 0, -1)
			rb := st.read(&w2, 0, x, y, 0, 1)
			num := twoE2*(bl+br) + twoE1*(rt+rb)
			w3_0[y*wf+x] = w2[0][y*wf+x] - floorDiv(num, 8)

			bt := st.read(&w2, 3, x, y, 0, -1)
			bb := st.read(&w2, 3, x, y, 0, 1)
			rl := st.read(&w2, 3, x, y, -1, 0)
			rr := st.read(&w2, 3, x, y, 1, 0)
			num = twoE2*(bt+bb) + twoE1*(rl+rr)
			w3_3[y*wf+x] = w2[3][y*wf+x] - floorDiv(num, 8)
		}
	}
	w3 := [4][]int32{w3_0, w2[1], w2[2], w3_3}

	// Step 4 — inverse CbCr (Table F.8).
	w4_1 := make([]int32, want)
	w4_2 := make([]int32, want)
	for y := 0; y < hf; y++ {
		for x := 0; x < wf; x++ {
			gl := st.read(&w3, 1, x, y, -1, 0)
			gr := st.read(&w3, 1, x, y, 1, 0)
			gt := st.read(&w3, 1, x, y, 0, -1)
			gb := st.read(&w3, 1, x, y, 0, 1)
			w4_1[y*wf+x] = w3[1][y*wf+x] + floorDiv(gl+gr+gt+gb, 4)

			gl = st.read(&w3, 2, x, y, -1, 0)
			gr = st.read(&w3, 2, x, y, 1, 0)
			gt = st.read(&w3, 2, x, y, 0, -1)
			gb = st.read(&w3, 2, x, y, 0, 1)
			w4_2[y*wf+x] = w3[2][y*wf+x] + floorDiv(gl+gr+gt+gb, 4)
		}
	}

	// Component re-order (Table F.4):
	//   Ω[0] = red, Ω[1] = first green, Ω[2] = second green, Ω[3] = blue.
	copy(planes[0], w4_2)
	copy(planes[1], w3_3)
	copy(planes[2], w3_0)
	copy(planes[3], w4_1)
	return nil
}

// applyOutputScaling performs the Annex G output scaling, the DC level
// shift and the clipping, then packs the samples.
func applyOutputScaling(omega []int32, bw, bc uint8) ([]byte, error) {
	if bc < 8 || bc > 16 {
		return nil, fmt.Errorf("jpegxs decode: B[i]=%d out of supported range 8..=16", bc)
	}
	if bw < 8 || bw > 20 {
		return nil, fmt.Errorf("jpegxs decode: Bw=%d out of supported range 8..=20", bw)
	}
	zeta := int32(bw) - int32(bc)
	m := (int32(1) << bc) - 1
	dcBias := int32(1) << (bw - 1)
	zetaU := zeta
	if zetaU < 0 {
		zetaU = 0
	}
	half := int32(0)
	if zetaU > 0 {
		half = int32(1) << (zetaU - 1)
	}
	recon := make([]int32, len(omega))
	for i, v := range omega {
		v = satAdd32(v, dcBias)
		if zetaU != 0 {
			v = satAdd32(v, half) >> zetaU
		}
		if v < 0 {
			v = 0
		}
		if v > m {
			v = m
		}
		recon[i] = v
	}
	if bc <= 8 {
		out := make([]byte, len(recon))
		for i, v := range recon {
			out[i] = byte(v)
		}
		return out, nil
	}
	out := make([]byte, len(recon)*2)
	for i, v := range recon {
		s := uint16(v)
		out[i*2] = byte(s & 0xff)
		out[i*2+1] = byte(s >> 8)
	}
	return out, nil
}

// satAdd32 adds two values with saturation at the int32 bounds.
func satAdd32(a, b int32) int32 {
	sum := int64(a) + int64(b)
	if sum > int64(^uint32(0)>>1) {
		return int32(^uint32(0) >> 1)
	}
	if sum < -int64(^uint32(0)>>1)-1 {
		return -int32(^uint32(0)>>1) - 1
	}
	return int32(sum)
}
