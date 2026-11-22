package jpegxs

// MSB-first bit writer. Bits collect in a 64-bit accumulator and leave
// as whole bytes; a trailing partial byte is zero-padded on
// alignToByte / intoBytes.
//
// The accumulator deliberately keeps the already-emitted high bits
// (they are shifted out by the byte extraction), mirroring the upstream
// Rust writer bit for bit.

type bitWriter struct {
	bytes []byte
	// acc holds pending bits in the low `fill` positions.
	acc  uint64
	fill uint32
}

// writeBit appends a single bit.
func (w *bitWriter) writeBit(bit uint8) {
	w.writeBits(uint32(bit&1), 1)
}

// writeBits appends the low n bits of value (n ≤ 32), most significant
// first.
func (w *bitWriter) writeBits(value uint32, n uint8) {
	if n == 0 {
		return
	}
	nu := uint32(n)
	masked := uint64(value) & ((1 << nu) - 1)
	w.acc = (w.acc << nu) | masked
	w.fill += nu
	for w.fill >= 8 {
		w.fill -= 8
		w.bytes = append(w.bytes, byte(w.acc>>w.fill))
	}
}

// writeOnes appends count one-bits (count ≤ 32).
func (w *bitWriter) writeOnes(count uint32) {
	if count > 0 {
		w.writeBits(^uint32(0)>>(32-count), uint8(count))
	}
}

// alignToByte zero-pads the pending partial byte.
func (w *bitWriter) alignToByte() {
	if w.fill != 0 {
		w.bytes = append(w.bytes, byte(w.acc<<(8-w.fill)))
		w.fill = 0
	}
}

// intoBytes returns the accumulated bytes, zero-padding the tail.
func (w *bitWriter) intoBytes() []byte {
	w.alignToByte()
	return w.bytes
}
