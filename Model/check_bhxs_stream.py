"""Verify a BHXS stream file: preamble, configuration and every block framing."""
import json
import struct
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    with path.open("rb") as stream:
        preamble = stream.read(9)
        if preamble[:4] != b"BHXS":
            raise SystemExit(f"bad magic {preamble[:5]!r}")
        version = preamble[4]
        length = struct.unpack(">I", preamble[5:9])[0]
        config = json.loads(stream.read(length).decode("utf-8"))
        decoded_size = 640 if config.get("include_decoded") else 0
        samples = 0
        payload_total = 0
        frames = 0
        while True:
            if samples % 30 == 0:
                # The first block of every frame is preceded by the frame index.
                frame_bytes = stream.read(8)
                if not frame_bytes:
                    break
                if len(frame_bytes) < 8:
                    raise SystemExit("truncated frame index")
                frame = struct.unpack(">q", frame_bytes)[0]
                frames += 1
            head = stream.read(2)
            if not head:
                break
            if len(head) < 2:
                raise SystemExit("truncated block header")
            size = struct.unpack(">H", head)[0]
            if size < 1 or size > 640:
                raise SystemExit(f"invalid payload size {size}")
            if len(stream.read(size)) != size:
                raise SystemExit("truncated payload")
            if decoded_size:
                if len(stream.read(decoded_size)) != decoded_size:
                    raise SystemExit("truncated decoded mosaic")
            samples += 1
            payload_total += size
        if stream.read(1):
            raise SystemExit("trailing bytes")
    print(json.dumps({
        "file": str(path), "bytes": path.stat().st_size, "version": version,
        "include_decoded": bool(config.get("include_decoded")),
        "samples": samples, "frames": frames, "payload_bytes": payload_total,
        "payload_only_bps": (payload_total + 2 * samples + 4 * frames) * 8 / (frames / 30),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
