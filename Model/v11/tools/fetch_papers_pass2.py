#!/usr/bin/env python
"""Second pass: fetch papers whose exact arXiv id is known, plus CVF open-access PDFs.

Verifies the arXiv title via id_list before downloading; records result in
Model/v11/來源清單-論文2.json (UTF-8, no BOM).
"""
import hashlib
import json
import os
import re
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "refs", "papers")
MANIFEST = os.path.join(ROOT, "來源清單-論文2.json")
UA = "BrainHoleResearch/1.0 (local research download)"

ARXIV_BY_ID = [
    ("2403.05477", "stablevsr", "StableVSR"),
    ("2505.19464", "flashvsr", "FlashVSR"),
    ("2603.19187", "genmfsr", "Generative Multi-frame Super-Resolution"),
    ("1711.07873", "frvsr", "Frame-Recurrent Super-Resolution"),
    ("1811.09393", "tecogan", "Temporal GAN for Real-Time Video Super-Resolution"),
    ("2007.07563", "motnet", "Learning Space-Time Video Super-Resolution with a Heterogeneous Network"),
]

CVF = [
    ("https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Learning_a_Deep_Single_Image_Super-Resolution_Network_for_Burst_Images_CVPR_2021_paper.pdf",
     "dbsr-official", "Learning a Deep Single Image Super-Resolution Network for Burst Images (CVPR 2021)"),
    ("https://openaccess.thecvf.com/content_CVPR_2020/papers/Nah_Learning_to_Super-Resolve_Burst_Photos_CVPR_2020_paper.pdf",
     "cycled-shift", "Learning to Super-Resolve Burst Photos (CVPR 2020)"),
    ("https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Real-World_Video_Super-Resolution_A_Benchmark_Dataset_and_a_Decomposition-Based_Learning_Scheme_ICCV_2021_paper.pdf",
     "realvsr", "Real-World Video Super-Resolution: A Benchmark Dataset and a Decomposition-based Learning Scheme (ICCV 2021)"),
]


def http_get(url, retries=3, timeout=120):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def arxiv_title(aid):
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    xml = http_get(url).decode("utf-8", "replace")
    m = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()


def save(url, slug, title, extra=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    ext = ".pdf"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", url.rsplit("/", 1)[-1])[:60]
    filename = f"{slug}_{safe}" if safe.lower().endswith(ext) else f"{slug}.pdf"
    path = os.path.join(OUT_DIR, filename)
    rec = {"slug": slug, "title": title, "pdf_url": url, "source": "arxiv" if "arxiv" in url else "cvf",
           "local_path": os.path.relpath(path, ROOT).replace("\\", "/"),
           "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        rec["status"] = "cached"
    else:
        try:
            data = http_get(url)
            if not data.startswith(b"%PDF"):
                raise RuntimeError("not a PDF payload")
            with open(path, "wb") as fh:
                fh.write(data)
            rec["status"] = "downloaded"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "download_failed"
            rec["error"] = str(exc)
            print(f"DL-FAIL {slug}: {exc}", flush=True)
            return rec
    rec["bytes"] = os.path.getsize(path)
    with open(path, "rb") as fh:
        rec["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    print(f"OK {slug} {rec['bytes']/1e6:.2f}MB {rec['status']} :: {rec['title'][:70]}", flush=True)
    return rec


def main():
    records = []
    for aid, slug, expect in ARXIV_BY_ID:
        try:
            real = arxiv_title(aid)
        except Exception as exc:  # noqa: BLE001
            real = ""
            print(f"LOOKUP-FAIL {slug}: {exc}", flush=True)
        match = bool(real) and expect.lower().split(":")[0].split("(")[0].strip().lower() in real.lower()
        print(f"CHECK {aid} expected~'{expect[:40]}' actual='{real[:60]}' match={match}", flush=True)
        rec = save(f"https://arxiv.org/pdf/{aid}", slug, real or expect,
                   {"arxiv_id": aid, "title_match": match, "requested_title": expect})
        records.append(rec)
        time.sleep(2.0)
    for url, slug, title in CVF:
        records.append(save(url, slug, title))
        time.sleep(2.0)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    ok = sum(1 for r in records if r.get("status") in ("downloaded", "cached"))
    print(f"DONE ok={ok}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
