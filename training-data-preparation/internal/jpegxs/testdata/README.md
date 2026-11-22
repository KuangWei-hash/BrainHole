# JPEG XS regression vectors

`q4-entropy-marker.hex` and `q4-entropy-marker-decoded.hex` are lowercase
hexadecimal text encodings of the original 640-byte RAW fixtures. Tests decode
them in memory; the original `.rggb` files remain local and are ignored by Git.
The decoded bytes must match the SHA-256 values recorded in
`q4-entropy-marker.json`; that historical provenance is unchanged.

This storage conversion follows the 2026-11-22 policy to publish code and
documents without uploading binary images. Each text vector represents one
160 x 4 Bayer block needed by the entropy-marker regression test.
