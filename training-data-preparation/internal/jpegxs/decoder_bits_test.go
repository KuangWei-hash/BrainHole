package jpegxs

import (
	"bytes"
	"testing"
)

// vlcCodeword builds the MSB-first bit pattern of one Table C.15
// codeword: x consecutive 1-bits followed by a single 0 comma bit.
func vlcCodeword(x int) ([]byte, int) {
	totalBits := x + 1
	buf := make([]byte, (totalBits+7)/8)
	for i := 0; i < x; i++ {
		buf[i/8] |= 0x80 >> (i % 8)
	}
	return buf, totalBits
}

// TestVLCZeroCodeword checks the neutral codeword: a lone 0 comma bit
// decodes to 0.
func TestVLCZeroCodeword(t *testing.T) {
	r := newBitReader([]byte{0x00})
	v, err := decodeVLCSigned(r, 0, 0)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if v != 0 {
		t.Fatalf("value = %d, want 0", v)
	}
	if r.bytesConsumed() != 1 {
		t.Fatalf("consumed %d bytes, want 1", r.bytesConsumed())
	}
}

// TestVLCUnaryAlphabet checks the unary sub-alphabet: with θ = 0 any
// x ≥ 1 maps to x.
func TestVLCUnaryAlphabet(t *testing.T) {
	buf, _ := vlcCodeword(2)
	r := newBitReader(buf)
	v, err := decodeVLCSigned(r, 0, 0)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if v != 2 {
		t.Fatalf("value = %d, want 2", v)
	}
}

// TestVLCSignedBinaryAlphabet checks the signed sub-alphabet: θ = 2 so
// 2θ = 4, and x ≤ 4 selects the signed binary mapping.
func TestVLCSignedBinaryAlphabet(t *testing.T) {
	cases := []struct {
		x    int
		want int32
	}{
		{1, -1},
		{2, 1},
		{3, -2},
		{4, 2},
		{5, 3}, // x > 2θ → unary sub-alphabet → x − θ
	}
	for _, tc := range cases {
		buf, _ := vlcCodeword(tc.x)
		r := newBitReader(buf)
		v, err := decodeVLCSigned(r, 2, 0)
		if err != nil {
			t.Fatalf("x=%d: decode: %v", tc.x, err)
		}
		if v != tc.want {
			t.Fatalf("x=%d: value = %d, want %d", tc.x, v, tc.want)
		}
	}
}

// TestVLCRejectsLossOfSync checks that 32 consecutive 1-bits are
// reported as a loss of synchronisation (Table C.15).
func TestVLCRejectsLossOfSync(t *testing.T) {
	r := newBitReader([]byte{0xff, 0xff, 0xff, 0xff})
	if _, err := decodeVLCSigned(r, 0, 0); err == nil {
		t.Fatal("32 consecutive 1-bits must be rejected")
	}
	// The decoder must stop at the 32nd bit and not read further.
	if got := r.bytesConsumed(); got != 4 {
		t.Fatalf("consumed %d bytes, want 4 (the decoder must not over-read)", got)
	}
}

// TestVLCLargestLegalCodeword checks that 31 ones closed by a comma bit
// is the largest legal codeword.
func TestVLCLargestLegalCodeword(t *testing.T) {
	r := newBitReader([]byte{0xff, 0xff, 0xff, 0xfe})
	v, err := decodeVLCSigned(r, 0, 0)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if v != 31 {
		t.Fatalf("value = %d, want 31", v)
	}
	if got := r.bytesConsumed(); got != 4 {
		t.Fatalf("consumed %d bytes, want 4", got)
	}
}

// TestVLCLeadingOnesIsNotLeadingZeros is the regression guard for the
// unary reader: counting the leading ones must not be confused with
// counting the leading zeros. A window whose first bit is 0 decodes as
// the neutral codeword, not as a 63-one run.
func TestVLCLeadingOnesIsNotLeadingZeros(t *testing.T) {
	for _, b := range []byte{0x00, 0x0f, 0x55, 0x80, 0xa6} {
		r := newBitReader([]byte{b, 0x00})
		if _, err := decodeVLCSigned(r, 0, 0); err != nil {
			t.Fatalf("byte %#02x: unexpected loss of synchronisation: %v", b, err)
		}
	}
}

// TestVLCCodewordRoundTrip checks that the encoder's emitVLCSigned and
// the decoder's decodeVLCSigned agree bit-exactly for every codeword
// the encoder can produce.
func TestVLCCodewordRoundTrip(t *testing.T) {
	for theta := int32(0); theta <= 6; theta++ {
		for value := -theta; value <= 12; value++ {
			var w bitWriter
			emitVLCSigned(&w, value, theta)
			encoded := w.intoBytes()
			r := newBitReader(encoded)
			got, err := decodeVLCSigned(r, int32(value+theta), 0)
			if err != nil {
				t.Fatalf("theta=%d value=%d: decode: %v", theta, value, err)
			}
			// decodeVLCSigned's θ is max(pred−trunc, 0); the encoder's
			// codeword for `value` under `theta` has to be reproduced by
			// the decoder when it is handed the same predictor.
			r2 := newBitReader(encoded)
			got2, err := decodeVLCSigned(r2, int32(theta), 0)
			if err != nil {
				t.Fatalf("theta=%d value=%d: decode: %v", theta, value, err)
			}
			_ = got
			if got2 != value {
				t.Fatalf("theta=%d value=%d: decoded %d", theta, value, got2)
			}
		}
	}
}

// TestBitReaderMSBFirst checks the basic MSB-first bit order.
func TestBitReaderMSBFirst(t *testing.T) {
	r := newBitReader([]byte{0xa6, 0xc3})
	for _, want := range []uint32{0b1010, 0b0110, 0b1100, 0b0011} {
		got, err := r.readBits(4)
		if err != nil {
			t.Fatalf("readBits: %v", err)
		}
		if got != want {
			t.Fatalf("readBits(4) = %04b, want %04b", got, want)
		}
	}
	if _, err := r.readBit(); err == nil {
		t.Fatal("reading past the end must fail")
	}
}

// TestBitReaderAlignToByte checks the pad(8) behaviour Annex C uses
// between sub-packets.
func TestBitReaderAlignToByte(t *testing.T) {
	r := newBitReader([]byte{0xa6, 0x12})
	if _, err := r.readBits(3); err != nil {
		t.Fatal(err)
	}
	if r.atByteBoundary() {
		t.Fatal("the cursor should sit mid-byte")
	}
	r.alignToByte()
	if !r.atByteBoundary() {
		t.Fatal("alignToByte must reach a byte boundary")
	}
	got, err := r.readBits(8)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0x12 {
		t.Fatalf("readBits(8) = %#02x, want 0x12", got)
	}
}

// TestBitReaderBytesConsumed checks the byte accounting.
func TestBitReaderBytesConsumed(t *testing.T) {
	r := newBitReader([]byte{0xa6, 0x12, 0x34})
	if r.bytesConsumed() != 0 {
		t.Fatalf("bytesConsumed = %d, want 0", r.bytesConsumed())
	}
	r.readBits(3)
	if r.bytesConsumed() != 1 {
		t.Fatalf("bytesConsumed = %d, want 1", r.bytesConsumed())
	}
	r.readBits(5)
	if r.bytesConsumed() != 1 {
		t.Fatalf("bytesConsumed = %d, want 1", r.bytesConsumed())
	}
	r.readBits(8)
	if r.bytesConsumed() != 2 {
		t.Fatalf("bytesConsumed = %d, want 2", r.bytesConsumed())
	}
}

// TestVLCMatchesRustReference cross-checks the decoder against the
// reference values of the upstream Rust implementation's own unit
// tests (oxideav-jpegxs src/entropy/bits.rs).
func TestVLCMatchesRustReference(t *testing.T) {
	cases := []struct {
		buf  []byte
		pred int32
		trnc int32
		want int32
	}{
		{[]byte{0b00000000}, 0, 0, 0},
		{[]byte{0b11000000}, 0, 0, 2},
		{[]byte{0b10000000}, 2, 0, -1},
		{[]byte{0b11000000}, 2, 0, 1},
		{[]byte{0b11100000}, 2, 0, -2},
		{[]byte{0b11110000}, 2, 0, 2},
		{[]byte{0b11111000}, 2, 0, 3},
		{[]byte{0xff, 0xff, 0xff, 0xfe}, 0, 0, 31},
	}
	for _, tc := range cases {
		r := newBitReader(tc.buf)
		got, err := decodeVLCSigned(r, tc.pred, tc.trnc)
		if err != nil {
			t.Fatalf("%#x: decode: %v", tc.buf, err)
		}
		if got != tc.want {
			t.Fatalf("%#x: value = %d, want %d", tc.buf, got, tc.want)
		}
	}
}

// TestVLCRejectsOverlongRun feeds a 40-one window and checks the
// loss-of-synchronisation error, mirroring the upstream test.
func TestVLCRejectsOverlongRun(t *testing.T) {
	r := newBitReader([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0x00})
	if _, err := decodeVLCSigned(r, 0, 0); err == nil {
		t.Fatal("40 consecutive 1-bits must be rejected")
	}
}

// TestEmitVLCSignedShape checks that the encoder's codeword writer
// produces exactly the bit pattern the decoder expects.
func TestEmitVLCSignedShape(t *testing.T) {
	for _, tc := range []struct {
		x     int
		value int32
		theta int32
	}{
		{0, 0, 0},
		{1, -1, 2},
		{2, 1, 2},
		{3, -2, 2},
		{4, 2, 2},
		{5, 3, 2},
		{31, 31, 0},
	} {
		var w bitWriter
		emitVLCSigned(&w, tc.value, tc.theta)
		want, _ := vlcCodeword(tc.x)
		if !bytes.Equal(w.intoBytes(), want) {
			t.Fatalf("value=%d theta=%d: codeword = %08b, want %08b", tc.value, tc.theta, w.intoBytes(), want)
		}
	}
}
