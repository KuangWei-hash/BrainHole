package jpegxs

import (
	"testing"
)

// Wavelet-transform conformance against ISO/IEC 21122-1:2022 Annex E.
//
// The standard specifies the *inverse* transform (Table E.1: at a joint
// wavelet level, horizontal synthesis of the low and of the high rows,
// then vertical synthesis) and describes the forward transform as its
// reverse (Table E.8: vertical analysis, then horizontal analysis of
// both halves).
//
// The helpers below are a literal transcription of the standard's tables,
// written independently of the encoder: they are the yardstick, not a
// second copy of the implementation. A Go encoder that agrees with them
// agrees with the standard.
//
// The two possible pass orders (vertical-first per Table E.8, and
// horizontal-first) produce *different* integer coefficients on the
// pictures used here, so the comparison can tell them apart.

type stdBand struct {
	w, h int
	t    []int64
}

// annexEForwardLine is Table E.5 + E.12 on one line: the whole-sample
// symmetric extension of Annex E.6, the odd (high-pass) lifting step and
// the even (low-pass) lifting step.
func annexEForwardLine(line []int64) (lo, hi []int64) {
	z := len(line)
	x := make([]int64, z+4)
	copy(x[2:2+z], line)
	// Whole-sample symmetric extension.
	for i := 1; i <= 2; i++ {
		x[2-i] = x[2+i]
		x[2+z+i-1] = x[2+z-i-1]
	}
	y := append([]int64(nil), x...)
	// High-pass step: i = -1, 1, 3, … < Z + 1, with the offset of 2 folded in.
	for k := 1; k < z+3; k += 2 {
		y[k] = x[k] - ((x[k-1] + x[k+1]) >> 1)
	}
	// Low-pass step.
	for i := 0; i < z; i += 2 {
		y[2+i] = x[2+i] + ((y[2+i-1] + y[2+i+1] + 2) >> 2)
	}
	body := y[2 : 2+z]
	lo = make([]int64, 0, (z+1)/2)
	hi = make([]int64, 0, z/2)
	for i, v := range body {
		if i%2 == 0 {
			lo = append(lo, v)
		} else {
			hi = append(hi, v)
		}
	}
	return lo, hi
}

// stdHorFwd is Table E.10: the rows of src into a low and a high band.
func stdHorFwd(src stdBand) (stdBand, stdBand) {
	var lo, hi []int64
	for y := 0; y < src.h; y++ {
		row := src.t[y*src.w : (y+1)*src.w]
		l, g := annexEForwardLine(row)
		lo = append(lo, l...)
		hi = append(hi, g...)
	}
	return stdBand{w: (src.w + 1) / 2, h: src.h, t: lo},
		stdBand{w: src.w / 2, h: src.h, t: hi}
}

// stdVerFwd is Table E.11: the columns of src into a low and a high band.
func stdVerFwd(src stdBand) (stdBand, stdBand) {
	lh, hh := (src.h+1)/2, src.h/2
	lo := make([]int64, src.w*lh)
	hi := make([]int64, src.w*hh)
	for x := 0; x < src.w; x++ {
		col := make([]int64, src.h)
		for y := 0; y < src.h; y++ {
			col[y] = src.t[y*src.w+x]
		}
		l, g := annexEForwardLine(col)
		for y, v := range l {
			lo[y*src.w+x] = v
		}
		for y, v := range g {
			hi[y*src.w+x] = v
		}
	}
	return stdBand{w: src.w, h: lh, t: lo}, stdBand{w: src.w, h: hh, t: hi}
}

// stdBeta is the Annex B.3 filter index of a band.
func stdBeta(nlx, nly, dx, dy, tx, ty int) int {
	if dx > dy {
		return nlx - dx + tx
	}
	return nlx - nly + tx + 2*ty + 3*(nly-dy)
}

// stdTableE8 is the forward transform of Annex E, Table E.8, for one
// component. The bands come back in filter-type (β) order.
//
// exchanged runs each joint level horizontally first — a *different*
// integer transform — and exists only to show that the test pictures can
// tell the two pass orders apart.
func stdTableE8(picture stdBand, nlx, nly int, exchanged bool) [][]int64 {
	out := make([][]int64, 2*nly+nlx+1)
	ll := picture
	for d := 1; d <= nly; d++ {
		var llD, hlD, lhD, hhD stdBand
		if exchanged {
			l, g := stdHorFwd(ll)
			llD, lhD = stdVerFwd(l)
			hlD, hhD = stdVerFwd(g)
		} else {
			l, g := stdVerFwd(ll)
			llD, hlD = stdHorFwd(l)
			lhD, hhD = stdHorFwd(g)
		}
		out[stdBeta(nlx, nly, d, d, 1, 0)] = hlD.t
		out[stdBeta(nlx, nly, d, d, 0, 1)] = lhD.t
		out[stdBeta(nlx, nly, d, d, 1, 1)] = hhD.t
		ll = llD
	}
	for dx := nly + 1; dx <= nlx; dx++ {
		llD, hlD := stdHorFwd(ll)
		out[stdBeta(nlx, nly, dx, nly, 1, 0)] = hlD.t
		ll = llD
	}
	out[0] = ll.t
	return out
}

// stdNoise is a deterministic bits-wide noise generator.
func stdNoise(n int, bits uint8, seed uint32) []int64 {
	s := seed*0x9E3779B9 + 0x7F4A7C15 | 1
	out := make([]int64, n)
	for i := range out {
		s ^= s << 13
		s ^= s >> 17
		s ^= s << 5
		out[i] = int64((s >> 8) & ((1 << bits) - 1))
	}
	return out
}

// TestForwardDWTMatchesTableE8 checks the encoder's forward wavelet
// transform against the literal transcription of Annex E above.
//
// The coefficients compared are the ones the encoder's picture-level
// cascade produces, so this is a check against the standard rather than
// against another implementation.
func TestForwardDWTMatchesTableE8(t *testing.T) {
	// (NL,x, NL,y) pairs the encoder supports and the pictures they
	// must separate the two pass orders on.
	shapes := []struct{ nlx, nly int }{
		{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0},
		{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1},
	}
	sizes := [][2]int{{32, 8}, {33, 9}, {37, 16}, {64, 23}, {80, 2}, {95, 34}}

	total, separating, jointTotal := 0, 0, 0
	for _, sh := range shapes {
		nlx, nly := sh.nlx, sh.nly
		for _, sz := range sizes {
			w, h := sz[0], sz[1]
			if w < (1<<nlx) || h < (1<<nly) {
				continue
			}
			plane := stdNoise(w*h, 8, uint32(w*5+h+nlx*11+nly))

			// The encoder's picture-level forward transform. Annex G: the
			// samples enter the transform as O = sample − 2^(B−1) =
			// sample − 128.
			plane32 := make([]int32, len(plane))
			for i, v := range plane {
				plane32[i] = int32(v) - 128
			}
			bands, err := forwardCascade2D(w, h, uint8(nlx), uint8(nly), plane32)
			if err != nil {
				t.Fatalf("%dx%d NL=%d/%d: forward cascade: %v", w, h, nlx, nly, err)
			}

			// Annex G: O = sample − 2^(B−1) = sample − 128.
			shifted := make([]int64, w*h)
			for i, v := range plane {
				shifted[i] = v - 128
			}
			want := stdTableE8(stdBand{w: w, h: h, t: shifted}, nlx, nly, false)

			if len(bands) != len(want) {
				t.Fatalf("%dx%d NL=%d/%d: %d filter types, want %d", w, h, nlx, nly, len(bands), len(want))
			}
			for b := range want {
				if len(bands[b]) != len(want[b]) {
					t.Fatalf("%dx%d NL=%d/%d β=%d: %d coefficients, want %d",
						w, h, nlx, nly, b, len(bands[b]), len(want[b]))
				}
				for i := range want[b] {
					if int64(bands[b][i]) != want[b][i] {
						t.Fatalf("%dx%d NL=%d/%d β=%d coefficient %d: got %d, want %d (Annex E Table E.8)",
							w, h, nlx, nly, b, i, bands[b][i], want[b][i])
					}
				}
			}
			total++
			if !stdBandsEqual(stdTableE8(stdBand{w: w, h: h, t: shifted}, nlx, nly, true), want) {
				separating++
			}
			if nly >= 1 {
				// A joint level is where the two pass orders can differ.
				jointTotal++
			}
		}
	}
	if total < 30 {
		t.Fatalf("only %d configurations ran", total)
	}
	if jointTotal == 0 || separating != jointTotal {
		t.Fatalf("only %d of %d joint-level pictures separate the two pass orders; the check cannot tell them apart",
			separating, jointTotal)
	}
}

// TestForwardDWTMatchesTableE8Streaming checks the NL,x = 1 / NL,y = 0
// streaming path — the one the BrainHole pipeline uses — against the same
// literal.
//
// The streaming path transforms one precinct row at a time; with NL,y = 0
// there is no vertical decomposition, so the picture-level coefficients
// are the per-row horizontal transform.
func TestForwardDWTMatchesTableE8Streaming(t *testing.T) {
	// Only NL,x = 1 takes the streaming path; NL,x > 1 goes through the
	// picture-level cascade (checked above).
	const nlx = 1
	for _, sz := range [][2]int{{80, 2}, {64, 23}, {37, 16}, {33, 9}} {
		w, h := sz[0], sz[1]
		plane := stdNoise(w*h, 8, uint32(w*7+h*3+nlx))

		// The streaming path's coefficients: each row transformed on its
		// own, exactly as the encoder's precinct loop does.
		llW, hlW := (w+1)/2, w/2
		ll := make([]int32, llW*h)
		hl := make([]int32, hlW*h)
		for y := 0; y < h; y++ {
			row := make([]int32, w)
			for x := 0; x < w; x++ {
				row[x] = int32(plane[y*w+x]) - 128
			}
			if err := forwardHorizontal1D(row, ll[y*llW:(y+1)*llW], hl[y*hlW:(y+1)*hlW]); err != nil {
				t.Fatal(err)
			}
		}

		shifted := make([]int64, w*h)
		for i, v := range plane {
			shifted[i] = v - 128
		}
		want := stdTableE8(stdBand{w: w, h: h, t: shifted}, nlx, 0, false)

		// β = 0 is LL, β = 1 is HL.
		for i := range ll {
			if int64(ll[i]) != want[0][i] {
				t.Fatalf("%dx%d NL=%d/0 β=0 coefficient %d: got %d, want %d",
					w, h, nlx, i, ll[i], want[0][i])
			}
		}
		for i := range hl {
			if int64(hl[i]) != want[1][i] {
				t.Fatalf("%dx%d NL=%d/0 β=1 coefficient %d: got %d, want %d",
					w, h, nlx, i, hl[i], want[1][i])
			}
		}
	}
}

// stdBandsEqual compares two β-ordered band lists.
func stdBandsEqual(a, b [][]int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

// TestForwardDWTBandKeyMatchesStandardBeta cross-checks the encoder's
// Annex B.3 filter-type enumeration against the standard's.
func TestForwardDWTBandKeyMatchesStandardBeta(t *testing.T) {
	for nlx := 1; nlx <= 5; nlx++ {
		for nly := 0; nly <= 1 && nly <= nlx; nly++ {
			n := int(nBeta(uint8(nlx), uint8(nly)))
			for beta := 0; beta < n; beta++ {
				key := betaToBandKey(uint32(beta), uint8(nlx), uint8(nly))
				dx, dy := int(key.dx), int(key.dy)
				tx, ty := 0, 0
				if key.tauX {
					tx = 1
				}
				if key.tauY {
					ty = 1
				}
				want := stdBeta(nlx, nly, dx, dy, tx, ty)
				if want != beta {
					t.Fatalf("NL=%d/%d: β=%d maps to (dx=%d, dy=%d, τx=%d, τy=%d) whose standard index is %d",
						nlx, nly, beta, dx, dy, tx, ty, want)
				}
			}
		}
	}
}
