#!/usr/bin/env python
"""Download Temporal SR / burst SR papers from arXiv by title lookup.

Resolves arXiv IDs via the arXiv API (no guessed IDs), then fetches the PDF.
Writes Model/v11/來源清單-論文.json (UTF-8, no BOM).
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "refs", "papers")
MANIFEST = os.path.join(ROOT, "來源清單-論文.json")
UA = "BrainHoleResearch/1.0 (local research download)"

PAPERS = [
    ("BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond", "basicvsr"),
    ("BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment", "basicvsrpp"),
    ("Temporally Deformable Alignment Network for Video Super-resolution", "tdan"),
    ("Recurrent Video Restoration Transformer with Guided Deformable Attention", "rvrt"),
    ("Enhancing Video Super-Resolution via Implicit Resampling-based Alignment", "ema-vsr"),
    ("Fast Online Video Super-Resolution with Deformable Attention Pyramid", "fast-online-vsr-dap"),
    ("Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention", "mia-vsr"),
    ("Learning a Deep Single Image Super-Resolution Network for Burst Images", "dbsr-official"),
    ("Deep Burst Super-Resolution", "deep-burst-sr"),
    ("Learning to Super-Resolve Burst Photos", "cycled-shift"),
    ("Improving Burst Super-Resolution with Swin Transformer and Flow-Guided Deformable Alignment", "bsrt"),
    ("Learning Trajectory-Aware Transformer for Video Super-Resolution", "ttvsr"),
    ("Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction", "mst"),
    ("Real-World Video Super-Resolution: A Benchmark Dataset and a Decomposition-based Learning Scheme", "realvsr"),
    ("Investigating Tradeoffs in Real-World Video Super-Resolution", "realbasicvsr"),
    ("VideoGigaGAN: Towards Detail-rich Video Super-Resolution", "videogigagan"),
    ("Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution", "uav"),
    ("Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution", "mgld-vsr"),
    ("StableVSR: Scaling up Video Super-Resolution with Temporally Consistent Detail Synthesis", "stablevsr"),
    ("Revealing an Effective Recipe for Taming Robust Video Super-Resolution Against Complex Degradations", "diffvsr"),
    ("FlashVSR: Towards Real-time Video Super-Resolution", "flashvsr"),
    ("Generative Multi-frame Super-Resolution", "genmfsr"),
    ("Frame-Recurrent Super-Resolution", "frvsr"),
    ("Temporal GAN for Real-Time Video Super-Resolution", "tecogan"),
    ("Learning Space-Time Video Super-Resolution with a Heterogeneous Network", "motnet"),
]


def http_get(url, retries=3, timeout=90):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def arxiv_lookup(title):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5"
    xml = http_get(url).decode("utf-8", "replace")
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    best = None
    for e in entries:
        eid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        title_m = re.search(r"<title>(.*?)</title>", e, re.S)
        if not eid or not title_m:
            continue
        t = re.sub(r"\s+", " ", title_m.group(1)).strip()
        norm_t = re.sub(r"[^a-z0-9]", "", t.lower())
        norm_q = re.sub(r"[^a-z0-9]", "", title.lower())
        if norm_t == norm_q or norm_q in norm_t or norm_t in norm_q:
            published = re.search(r"<published>([^<]+)</published>", e)
            best = (eid.group(1), t, published.group(1) if published else "")
            break
        if best is None:
            published = re.search(r"<published>([^<]+)</published>", e)
            best = (eid.group(1), t, published.group(1) if published else "")
    return best


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for title, slug in PAPERS:
        try:
            found = arxiv_lookup(title)
        except Exception as exc:  # noqa: BLE001
            print(f"LOOKUP-FAIL {slug}: {exc}", flush=True)
            records.append({"slug": slug, "requested_title": title, "status": "lookup_failed",
                            "error": str(exc), "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            continue
        if not found:
            print(f"NO-ID {slug}", flush=True)
            records.append({"slug": slug, "requested_title": title, "status": "no_id",
                            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            continue
        aid, real_title, published = found
        filename = f"{slug}_arXiv-{aid.replace('/', '_')}.pdf"
        path = os.path.join(OUT_DIR, filename)
        rec = {
            "slug": slug,
            "requested_title": title,
            "title": real_title,
            "arxiv_id": aid,
            "published": published,
            "abs_url": f"https://arxiv.org/abs/{aid}",
            "pdf_url": f"https://arxiv.org/pdf/{aid}",
            "local_path": os.path.relpath(path, ROOT).replace("\\", "/"),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            rec["status"] = "cached"
        else:
            try:
                data = http_get(rec["pdf_url"])
                if not data.startswith(b"%PDF"):
                    raise RuntimeError("not a PDF payload")
                with open(path, "wb") as fh:
                    fh.write(data)
                rec["status"] = "downloaded"
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "download_failed"
                rec["error"] = str(exc)
                print(f"DL-FAIL {slug}: {exc}", flush=True)
                continue
        rec["bytes"] = os.path.getsize(path)
        with open(path, "rb") as fh:
            rec["sha256"] = hashlib.sha256(fh.read()).hexdigest()
        print(f"OK {slug} arXiv:{aid} {rec['bytes']/1e6:.2f}MB {rec['status']}", flush=True)
        records.append(rec)
        time.sleep(2.0)

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    ok = sum(1 for r in records if r.get("status") in ("downloaded", "cached"))
    print(f"DONE ok={ok}/{len(records)} manifest={MANIFEST}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
