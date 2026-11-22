package dataset

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
)

// BHXS stream versions.
//
// v1 carries only the shared-header-v1 JPEG XS payload of each block.
//
// v2 adds the optional low-resolution decoded mosaic (160x4 RGGB, 640
// bytes) that schema v4 stores next to every sample. The field is opt-in
// through StreamConfig.IncludeDecoded: a bandwidth-sensitive live link
// keeps sending the payload only, while a dataset export can carry the
// second model input without a decoder at training time.
var (
	streamMagicV1 = [5]byte{'B', 'H', 'X', 'S', 1}
	streamMagicV2 = [5]byte{'B', 'H', 'X', 'S', 2}
)

type StreamConfig struct {
	Header      SharedHeader `json:"header"`
	Width       int          `json:"rgb_width"`
	Height      int          `json:"rgb_height"`
	BlockHeight int          `json:"block_height"`
	FPS         int          `json:"fps"`
	Phases      int          `json:"phases"`
	Padding     int          `json:"model_padding_bytes"`
	Mosaic      string       `json:"mosaic"`
	// IncludeDecoded reports whether every block carries the 640-byte
	// decoded low-resolution mosaic after its payload.
	IncludeDecoded bool `json:"include_decoded,omitempty"`
}

func streamConfig(h SharedHeader) StreamConfig {
	return StreamConfig{h, Width, Height, BlockHeight, FPS, 16, MaxEncodedBytes, "RGGB", false}
}

// streamConfigV2 builds the v2 configuration.
func streamConfigV2(h SharedHeader, includeDecoded bool) StreamConfig {
	c := streamConfig(h)
	c.IncludeDecoded = includeDecoded
	return c
}

func (c StreamConfig) validate() error {
	if c.Width != Width || c.Height != Height || c.BlockHeight != BlockHeight || c.FPS != FPS || c.Phases != 16 || c.Padding != MaxEncodedBytes || c.Mosaic != "RGGB" {
		return fmt.Errorf("unsupported stream geometry/settings")
	}
	// Validate marker/layout syntax using a dummy payload; no codec decoding is claimed.
	full, err := c.Header.Restore([]byte{0})
	if err != nil {
		return err
	}
	h, _, err := SplitCodestream(full, c.Header.EncoderIdentity)
	if err != nil {
		return err
	}
	if h.ID != c.Header.ID {
		return fmt.Errorf("noncanonical shared header")
	}
	return nil
}

func writeExact(w io.Writer, b []byte) error {
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

// StreamWriter writes blocks as soon as they arrive, never buffering an entire frame.
// The transport must provide reliable, ordered bytes (file/TCP/QUIC stream).
type StreamWriter struct {
	w      io.Writer
	Config StreamConfig
	frame  int64
	block  int
	failed error
	// includeDecoded mirrors Config.IncludeDecoded and decides whether the
	// decoded mosaic is appended to each block packet.
	includeDecoded bool
}

// NewStreamWriter returns a v1 writer that carries the payload only.
func NewStreamWriter(w io.Writer, h SharedHeader) (*StreamWriter, error) {
	return newStreamWriter(w, h, streamMagicV1, streamConfig(h))
}

// NewStreamWriterV2 returns a v2 writer. When includeDecoded is set every
// block packet also carries the 640-byte decoded mosaic.
func NewStreamWriterV2(w io.Writer, h SharedHeader, includeDecoded bool) (*StreamWriter, error) {
	return newStreamWriter(w, h, streamMagicV2, streamConfigV2(h, includeDecoded))
}

func newStreamWriter(w io.Writer, h SharedHeader, magic [5]byte, c StreamConfig) (*StreamWriter, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	b, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	if len(b) > 65536 {
		return nil, fmt.Errorf("stream configuration too large")
	}
	preamble := append([]byte(nil), magic[:]...)
	preamble = binary.BigEndian.AppendUint32(preamble, uint32(len(b)))
	preamble = append(preamble, b...)
	if err = writeExact(w, preamble); err != nil {
		return nil, err
	}
	return &StreamWriter{w: w, Config: c, frame: -1, block: BlockCount, includeDecoded: c.IncludeDecoded}, nil
}

// WriteBlock writes one block's payload.
func (s *StreamWriter) WriteBlock(frame int64, block int, payload []byte) error {
	if s.includeDecoded {
		return fmt.Errorf("stream writer expects the decoded mosaic of every block")
	}
	return s.writeBlock(frame, block, payload, nil)
}

// WriteBlockDecoded writes one block's payload together with the decoded
// low-resolution mosaic. The writer must have been created with
// NewStreamWriterV2 and includeDecoded set.
func (s *StreamWriter) WriteBlockDecoded(frame int64, block int, payload, decoded []byte) error {
	if !s.includeDecoded {
		return fmt.Errorf("stream writer does not carry decoded mosaics")
	}
	if len(decoded) != DecodedBytes {
		return fmt.Errorf("decoded mosaic is %d bytes, want %d", len(decoded), DecodedBytes)
	}
	return s.writeBlock(frame, block, payload, decoded)
}

func (s *StreamWriter) writeBlock(frame int64, block int, payload, decoded []byte) error {
	if s.failed != nil {
		return s.failed
	}
	if frame < 0 || block < 0 || block >= BlockCount {
		return fmt.Errorf("invalid stream sample index")
	}
	if _, err := s.Config.Header.Restore(payload); err != nil {
		return err
	}
	var packet []byte
	if s.block == BlockCount {
		if block != 0 || (s.frame >= 0 && (s.frame == math.MaxInt64 || frame != s.frame+1)) {
			return fmt.Errorf("expected first block of next consecutive frame")
		}
		packet = binary.BigEndian.AppendUint64(packet, uint64(frame))
	} else if frame != s.frame || block != s.block {
		return fmt.Errorf("out-of-order stream block")
	}
	packet = binary.BigEndian.AppendUint16(packet, uint16(len(payload)))
	packet = append(packet, payload...)
	if s.includeDecoded {
		packet = append(packet, decoded...)
	}
	if err := writeExact(s.w, packet); err != nil {
		s.failed = err
		return err
	}
	s.frame = frame
	s.block = block + 1
	return nil
}

func (s *StreamWriter) Finish() error {
	if s.failed != nil {
		return s.failed
	}
	if s.block != BlockCount {
		return fmt.Errorf("stream ends mid-frame")
	}
	return nil
}

type WireSample struct {
	Frame        int64
	Block, Phase int
	HeaderID     string
	Payload      []byte
	// Decoded is the low-resolution mosaic of the block. It is nil unless
	// the stream carries the decoded mosaics.
	Decoded []byte
}

type StreamReader struct {
	r      io.Reader
	Config StreamConfig
	frame  int64
	block  int
	failed error
	// version is the BHXS stream version the preamble declared.
	version int
}

func NewStreamReader(r io.Reader) (*StreamReader, error) {
	var preamble [9]byte
	if _, err := io.ReadFull(r, preamble[:]); err != nil {
		return nil, err
	}
	version := 0
	switch {
	case preamble[0] == streamMagicV1[0] && preamble[1] == streamMagicV1[1] && preamble[2] == streamMagicV1[2] && preamble[3] == streamMagicV1[3]:
		switch preamble[4] {
		case streamMagicV1[4]:
			version = 1
		case streamMagicV2[4]:
			version = 2
		}
	}
	if version == 0 {
		return nil, fmt.Errorf("unsupported BHXS magic/version")
	}
	n := binary.BigEndian.Uint32(preamble[5:])
	if n == 0 || n > 65536 {
		return nil, fmt.Errorf("invalid stream configuration size")
	}
	b := make([]byte, int(n))
	if _, err := io.ReadFull(r, b); err != nil {
		return nil, err
	}
	var c StreamConfig
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	if version == 1 && c.IncludeDecoded {
		return nil, fmt.Errorf("BHXS v1 stream must not carry decoded mosaics")
	}
	return &StreamReader{r: r, Config: c, frame: -1, block: BlockCount, version: version}, nil
}

// Next returns one model input immediately after that block's bytes arrive.
func (s *StreamReader) Next() (sample WireSample, err error) {
	if s.failed != nil {
		return sample, s.failed
	}
	defer func() {
		if err != nil {
			s.failed = err
		}
	}()
	if s.block == BlockCount {
		var frameBytes [8]byte
		_, err = io.ReadFull(s.r, frameBytes[:])
		if err != nil {
			return sample, err
		}
		frame := binary.BigEndian.Uint64(frameBytes[:])
		if frame > math.MaxInt64 {
			return sample, fmt.Errorf("frame index overflow")
		}
		if s.frame >= 0 && (s.frame == math.MaxInt64 || int64(frame) != s.frame+1) {
			return sample, fmt.Errorf("nonconsecutive frame index")
		}
		s.frame = int64(frame)
		s.block = 0
	}
	var size [2]byte
	if _, err = io.ReadFull(s.r, size[:]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return sample, err
	}
	n := int(binary.BigEndian.Uint16(size[:]))
	if n < 1 || n > MaxEncodedBytes {
		return sample, fmt.Errorf("payload size outside limits")
	}
	payload := make([]byte, n)
	if _, err = io.ReadFull(s.r, payload); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return sample, err
	}
	if _, err = s.Config.Header.Restore(payload); err != nil {
		return sample, err
	}
	var decoded []byte
	if s.Config.IncludeDecoded {
		decoded = make([]byte, DecodedBytes)
		if _, err = io.ReadFull(s.r, decoded); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return sample, err
		}
	}
	sample = WireSample{Frame: s.frame, Block: s.block, Phase: Phase(s.frame), HeaderID: s.Config.Header.ID, Payload: payload, Decoded: decoded}
	s.block++
	return sample, nil
}
