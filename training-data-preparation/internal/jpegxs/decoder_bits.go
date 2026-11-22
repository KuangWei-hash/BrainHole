package jpegxs

import (
	"fmt"
	"math/bits"
)

// MSB-first bit reader — ISO/IEC 21122-1:2022 Annex C.
//
// Port of the decoder-side half of the upstream Rust implementation
// (oxideav-jpegxs `src/entropy/bits.rs`). Bits in a JPEG XS codestream
// are packed MSB-first into bytes with no byte stuffing; the reader
// keeps a byte cursor plus a bit offset inside the current byte.
//
// The reader is deliberately strict: a read past the end of the buffer
// is an error rather than a zero fill, so a malformed stream fails
// instead of silently decoding to garbage.

type bitReader struct {
	buf []byte
	// bytePos is the index of the current byte.
	bytePos int
	// bitPos is the bit offset inside buf[bytePos]; 0 = MSB, 8 means
	// the current byte is exhausted and the cursor must advance.
	bitPos uint8
}

func newBitReader(buf []byte) *bitReader {
	return &bitReader{buf: buf}
}

// remainingBits counts the bits left from the current cursor to the end
// of the buffer.
func (r *bitReader) remainingBits() int {
	if r.bytePos >= len(r.buf) {
		return 0
	}
	inByte := 8 - int(r.bitPos)
	trailing := len(r.buf) - r.bytePos - 1
	return inByte + trailing*8
}

// bytesConsumed returns the whole bytes consumed so far, rounding up when
// the cursor sits mid-byte.
func (r *bitReader) bytesConsumed() int {
	if r.bitPos == 0 {
		return r.bytePos
	}
	return r.bytePos + 1
}

// atByteBoundary reports whether the cursor sits on a byte boundary.
func (r *bitReader) atByteBoundary() bool {
	return r.bitPos == 0
}

// alignToByte skips the remainder of the current byte. Annex C uses this
// for the pad(8) trailers of the sub-packets and the precinct header.
func (r *bitReader) alignToByte() {
	if r.bitPos != 0 {
		r.bytePos++
		r.bitPos = 0
	}
}

// skipBytes skips n whole bytes of filler. The cursor must be on a byte
// boundary first.
func (r *bitReader) skipBytes(n int) error {
	if !r.atByteBoundary() {
		return fmt.Errorf("jpegxs entropy: skip_bytes invoked off a byte boundary")
	}
	if r.bytePos+n > len(r.buf) {
		return fmt.Errorf("jpegxs entropy: filler skip %d exceeds remaining %d", n, len(r.buf)-r.bytePos)
	}
	r.bytePos += n
	return nil
}

// readBit reads one bit, MSB-first.
func (r *bitReader) readBit() (uint8, error) {
	if r.bytePos >= len(r.buf) {
		return 0, fmt.Errorf("jpegxs entropy: read_bit past end of buffer")
	}
	bit := (r.buf[r.bytePos] >> (7 - r.bitPos)) & 0x01
	r.bitPos++
	if r.bitPos == 8 {
		r.bitPos = 0
		r.bytePos++
	}
	return bit, nil
}

// peekWord returns the next bits of the stream left-aligned in a 64-bit
// word (zero beyond the end of the buffer) together with how many of them
// are real.
func (r *bitReader) peekWord() (uint64, int) {
	var rest []byte
	if r.bytePos < len(r.buf) {
		rest = r.buf[r.bytePos:]
	}
	n := len(rest)
	if n > 8 {
		n = 8
	}
	var bytes [8]byte
	copy(bytes[:], rest[:n])
	word := uint64(bytes[0])<<56 | uint64(bytes[1])<<48 | uint64(bytes[2])<<40 | uint64(bytes[3])<<32 |
		uint64(bytes[4])<<24 | uint64(bytes[5])<<16 | uint64(bytes[6])<<8 | uint64(bytes[7])
	word <<= r.bitPos
	valid := n*8 - int(r.bitPos)
	if valid < 0 {
		valid = 0
	}
	return word, valid
}

// advance moves the cursor forward by n bits.
func (r *bitReader) advance(n int) {
	pos := int(r.bitPos) + n
	r.bytePos += pos / 8
	r.bitPos = uint8(pos % 8)
}

// readBits reads n bits MSB-first (n ≤ 32).
func (r *bitReader) readBits(n uint8) (uint32, error) {
	if n == 0 {
		return 0, nil
	}
	if n > 32 {
		return 0, fmt.Errorf("jpegxs entropy: read_bits(%d) > 32", n)
	}
	word, valid := r.peekWord()
	if int(n) > valid {
		return 0, fmt.Errorf("jpegxs entropy: read_bit past end of buffer")
	}
	r.advance(int(n))
	return uint32(word >> (64 - uint32(n))), nil
}

// readUnaryCapped consumes a unary run: the number of one-bits before the
// terminating zero bit, which is consumed too. It reports an error when 32
// one-bits arrive without a terminator (exactly those 32 bits are
// consumed), and a hard error when the stream ends inside the run.
func (r *bitReader) readUnaryCapped() (uint32, bool, error) {
	word, valid := r.peekWord()
	// Leading ones of the window = leading zeros of its complement.
	// Go's math/bits has no LeadingOnes, so invert and count.
	ones := bits.LeadingZeros64(^word)
	if ones >= 32 {
		r.advance(32)
		return 0, false, nil
	}
	if ones >= valid {
		return 0, false, fmt.Errorf("jpegxs entropy: read_bit past end of buffer")
	}
	r.advance(ones + 1)
	return uint32(ones), true, nil
}

// decodeVLCSigned decodes a variable-length signed quantity —
// ISO/IEC 21122-1:2022 Annex C.7.1, Table C.15.
//
// It reads a unary count of one-bits terminated by a zero comma bit and
// maps the count x to a signed value:
//
//   - x > 2θ (θ = max(r−t, 0)) → unary sub-alphabet → x − θ (positive)
//   - 0 < x ≤ 2θ → signed binary sub-alphabet: odd x → −⌈x/2⌉, even x → x/2
//   - x == 0 → 0
//
// Table C.15's loop stops counting the moment x reaches 32 without
// consuming a further bit, and x ≥ 32 is a hard loss-of-synchronisation
// error. Matching that exactly keeps the reader in lockstep with the
// source on a malformed stream.
func decodeVLCSigned(r *bitReader, pred, trunc int32) (int32, error) {
	theta := pred - trunc
	if theta < 0 {
		theta = 0
	}
	x, ok, err := r.readUnaryCapped()
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, fmt.Errorf("jpegxs entropy: vlc decoder saw 32 consecutive 1-bits (lost synchronisation, ISO/IEC 21122-1 Table C.15)")
	}
	xi := int32(x)
	switch {
	case xi > 2*theta:
		return xi - theta, nil
	case xi > 0:
		if xi&1 == 1 {
			return -((xi + 1) / 2), nil
		}
		return xi / 2, nil
	default:
		return 0, nil
	}
}
