use oxideav_jpegxs::{
    Components, EncodeOptions, Plane, StarTetrixParams, decode_components, encode_components,
};
use std::{env, error::Error, fs, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const RAW_WIDTH: usize = 160;
const RAW_HEIGHT: usize = 4;
const PLANE_WIDTH: usize = RAW_WIDTH / 2;
const PLANE_HEIGHT: usize = RAW_HEIGHT / 2;

// The CFA sample grid is deinterleaved into R, G1, G2, B, each 80x2.
// The codestream picture grid is therefore 80x2 with four registered components,
// representing the original 160x4 mosaic, not four 160x4 planes.
fn components(raw: &[u8]) -> Result<Components> {
    if raw.len() != RAW_WIDTH * RAW_HEIGHT {
        return Err(format!("expected 640 bytes of 160x4 RGGB8, got {}", raw.len()).into());
    }
    let mut planes = vec![vec![0u8; PLANE_WIDTH * PLANE_HEIGHT]; 4];
    for y in 0..RAW_HEIGHT {
        for x in 0..RAW_WIDTH {
            let c = (y % 2) * 2 + x % 2;
            planes[c][(y / 2) * PLANE_WIDTH + x / 2] = raw[y * RAW_WIDTH + x];
        }
    }
    let planes = planes
        .into_iter()
        .map(|data| Plane::new(PLANE_WIDTH, data))
        .collect();
    Ok(Components::new(
        PLANE_WIDTH as u32,
        PLANE_HEIGHT as u32,
        3,
        vec![8; 4],
        vec![(1, 1); 4],
        planes,
    )?)
}

fn mosaic(c: &Components) -> Result<Vec<u8>> {
    if c.width != PLANE_WIDTH as u32
        || c.height != PLANE_HEIGHT as u32
        || c.cpih != 3
        || c.planes.len() != 4
        || c.bit_depths != vec![8; 4]
        || c.sampling != vec![(1, 1); 4]
    {
        return Err("not the bridge's 160x4 RGGB8 Star-Tetrix component layout".into());
    }
    if c.planes
        .iter()
        .any(|p| p.stride != PLANE_WIDTH || p.data.len() != PLANE_WIDTH * PLANE_HEIGHT)
    {
        return Err("unexpected decoded plane geometry".into());
    }
    let mut raw = vec![0; RAW_WIDTH * RAW_HEIGHT];
    for y in 0..RAW_HEIGHT {
        for x in 0..RAW_WIDTH {
            raw[y * RAW_WIDTH + x] =
                c.planes[(y % 2) * 2 + x % 2].data[(y / 2) * PLANE_WIDTH + x / 2];
        }
    }
    Ok(raw)
}

fn encode(c: &Components, q: u8, lx: u8, ly: u8) -> Result<Vec<u8>> {
    let opts = EncodeOptions::default()
        .with_quantization(q)
        .with_levels(Some(lx), Some(ly))
        .with_star_tetrix(Some(StarTetrixParams::new(0, 0, 0, 0)));
    Ok(encode_components(c, &opts)?)
}

struct Options {
    mode: String,
    input: PathBuf,
    output: Option<PathBuf>,
    target: usize,
    max: usize,
    lx: u8,
    ly: u8,
    q: Option<u8>,
}

fn parse() -> Result<Options> {
    let mut args = env::args().skip(1);
    let mode = args.next().ok_or("usage: brainhole-jpegxs encode|decode|sweep --input FILE [--output FILE] [--target-bytes 80] [--max-bytes 640] [--levels-x 1] [--levels-y 0] [--quantization 0..15]")?;
    if !["encode", "decode", "decode-batch", "sweep"].contains(&mode.as_str()) {
        return Err("expected encode, decode, decode-batch or sweep".into());
    }
    let mut o = Options {
        mode,
        input: PathBuf::new(),
        output: None,
        target: 80,
        max: 640,
        lx: 1,
        ly: 0,
        q: None,
    };
    while let Some(key) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--input" => o.input = value.into(),
            "--output" => o.output = Some(value.into()),
            "--target-bytes" => o.target = value.parse()?,
            "--max-bytes" => o.max = value.parse()?,
            "--levels-x" => o.lx = value.parse()?,
            "--levels-y" => o.ly = value.parse()?,
            "--quantization" => o.q = Some(value.parse()?),
            _ => return Err(format!("unknown argument {key}").into()),
        }
    }
    if o.input.as_os_str().is_empty() || (o.mode != "sweep" && o.output.is_none()) {
        return Err("--input and --output are required (sweep needs only --input)".into());
    }
    if o.target == 0
        || o.target > o.max
        || !(1..=5).contains(&o.lx)
        || o.ly > 1
        || o.ly > o.lx
        || o.q.is_some_and(|q| q > 15)
    {
        return Err("invalid byte budgets, levels (x=1..5, y=0..1) or quantization".into());
    }
    if let Some(output) = &o.output {
        if fs::exists(output)? {
            return Err("output already exists; refusing overwrite".into());
        }
    }
    Ok(o)
}

fn run(o: Options) -> Result<()> {
    let raw = fs::read(&o.input)?;
    if o.mode == "decode-batch" {
        fs::write(o.output.ok_or("missing output")?, decode_batch(&raw)?)?;
        return Ok(());
    }
    if o.mode == "decode" {
        let decoded = mosaic(&decode_components(&raw)?)?;
        fs::write(o.output.ok_or("missing output")?, decoded)?;
        return Ok(());
    }
    let c = components(&raw)?;
    if o.mode == "sweep" {
        println!("levels_x,levels_y,quantization,bytes");
        for lx in 1..=5 {
            for ly in 0..=1 {
                for q in 0..=15 {
                    let bytes = encode(&c, q, lx, ly)?;
                    // Check each measured stream actually decodes with the upstream decoder.
                    let restored = mosaic(&decode_components(&bytes)?)?;
                    if q == 0 && restored != raw {
                        return Err("lossless sweep round trip mismatch".into());
                    }
                    println!("{lx},{ly},{q},{}", bytes.len());
                }
            }
        }
        return Ok(());
    }
    let mut target_fit = None;
    let mut cap_fit = None;
    let mut smallest = usize::MAX;
    // Search actual streams; do not assume output lengths are monotonic in Q.
    for q in 0..=15 {
        if o.q.is_some_and(|forced| forced != q) {
            continue;
        }
        let bytes = encode(&c, q, o.lx, o.ly)?;
        smallest = smallest.min(bytes.len());
        if bytes.len() <= o.target && target_fit.is_none() {
            target_fit = Some((q, bytes.clone()));
        }
        if bytes.len() <= o.max && cap_fit.is_none() {
            cap_fit = Some((q, bytes));
        }
    }
    // Prefer the lowest Q that meets the target. If impossible, choose the lowest Q
    // within the configured cap, without claiming the target compression ratio.
    let (q, bytes) = target_fit.or(cap_fit).ok_or_else(|| {
        format!(
            "no stream fits {} bytes; minimum observed size is {smallest}",
            o.max
        )
    })?;
    let restored = mosaic(&decode_components(&bytes)?)?;
    if q == 0 && restored != raw {
        return Err("lossless round trip mismatch".into());
    }
    eprintln!(
        "raw_bytes=640 encoded_bytes={} ratio={:.3} q={} target_met={} levels={}/{} cpih=3 grid=80x2 CFA=160x4 RGGB",
        bytes.len(),
        640.0 / bytes.len() as f64,
        q,
        bytes.len() <= o.target,
        o.lx,
        o.ly
    );
    fs::write(o.output.ok_or("missing output")?, bytes)?;
    Ok(())
}

// Offline evaluation protocol: repeated u32 BE lengths + complete codestreams.
// Output is concatenated 640-byte RAW blocks, preserving input order.
fn decode_batch(mut input: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    while !input.is_empty() {
        if input.len() < 4 {
            return Err("truncated batch length".into());
        }
        let length = u32::from_be_bytes(input[..4].try_into()?) as usize;
        input = &input[4..];
        if length == 0 || length > 640 || input.len() < length {
            return Err("invalid or truncated batch codestream".into());
        }
        output.extend(mosaic(&decode_components(&input[..length])?)?);
        input = &input[length..];
    }
    Ok(output)
}

fn main() {
    if let Err(err) = parse().and_then(run) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_preserves_order_and_rejects_truncation() {
        let patterns = patterns();
        // Q=0 random streams can exceed the production transport limit;
        // batch test uses only the two compact uniform patterns.
        let mut compact = Vec::new();
        for raw in &patterns[..2] {
            let bytes = encode(&components(raw).unwrap(), 0, 1, 0).unwrap();
            compact.extend((bytes.len() as u32).to_be_bytes());
            compact.extend(bytes);
        }
        assert_eq!(decode_batch(&compact).unwrap(), patterns[..2].concat());
        compact.pop();
        assert!(decode_batch(&compact).is_err());
        assert!(decode_batch(&[0, 0]).is_err());
        assert!(decode_batch(&[0, 0, 0, 0]).is_err());
    }

    fn patterns() -> Vec<Vec<u8>> {
        let mut seed = 7u32;
        vec![
            vec![0; 640],
            vec![128; 640],
            (0..640)
                .map(|i| ((i * 7 + i / 160 * 11) % 256) as u8)
                .collect(),
            (0..640)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 24) as u8
                })
                .collect(),
        ]
    }

    #[test]
    fn lossless_cfa_roundtrip_at_actual_geometry() {
        for raw in patterns() {
            let c = components(&raw).unwrap();
            assert_eq!(mosaic(&c).unwrap(), raw);
            for lx in 1..=5 {
                for ly in 0..=1 {
                    let bytes = encode(&c, 0, lx, ly).unwrap();
                    assert_eq!(&bytes[..2], &[0xff, 0x10]);
                    assert_eq!(
                        mosaic(&decode_components(&bytes).unwrap()).unwrap(),
                        raw,
                        "levels {lx}/{ly}"
                    );
                }
            }
        }
    }

    #[test]
    fn lossy_streams_decode_and_keep_cfa_geometry() {
        for raw in patterns() {
            let c = components(&raw).unwrap();
            for q in 1..=15 {
                let bytes = encode(&c, q, 1, 0).unwrap();
                assert_eq!(
                    mosaic(&decode_components(&bytes).unwrap()).unwrap().len(),
                    640
                );
            }
        }
    }
}
