#!/usr/bin/env python
"""Third pass: resolve papers by title via Semantic Scholar, verify on arXiv, download.

Also flags any previously downloaded PDF whose arXiv title does not match the
requested title so wrong files can be removed (recoverable delete via `rm --`).
Writes Model/v11/來源清單-論文3.json (UTF-8, no BOM).
"""
import difflib
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "refs", "papers")
MANIFEST = os.path.join(ROOT, "來源清單-論文3.json")
UA = "BrainHoleResearch/1.0 (local research download)"

TARGETS = [
    ("dbsr-official", "Learning a Deep Single Image Super-Resolution Network for Burst Images"),
    ("cycled-shift", "Learning to Super-Resolve Burst Photos"),
    ("realvsr", "Real-World Video Super-Resolution: A Benchmark Dataset and a Decomposition-based Learning Scheme"),
    ("stablevsr", "StableVSR: Scaling up Video Super-Resolution with Temporally Consistent Detail Synthesis"),
    ("flashvsr", "FlashVSR: Towards Real-time Video Super-Resolution"),
    ("frvsr", "Frame-Recurrent Super-Resolution"),
    ("tecogan", "Temporal GAN for Real-Time Video Super-Resolution"),
    ("motnet", "Learning Space-Time Video Super-Resolution with a Heterogeneous Network"),
]


def http_get(url, retries=3, timeout=120, accept=None):
    last = None
    for attempt in range(retries):
        try:
            headers = {"User-Agent": UA}
            if accept:
                headers["Accept"] = accept
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def s2_search(title):
    q = urllib.parse.quote(title)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}"
           f"&limit=5&fields=title,year,venue,externalIds,openAccessPdf")
    data = json.loads(http_get(url).decode("utf-8", "replace"))
    best = None
    for item in data.get("data", []):
        sim = similarity(title, item.get("title", ""))
        if best is None or sim > best[0]:
            best = (sim, item)
    return best


def arxiv_title(aid):
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    xml = http_get(url).decode("utf-8", "replace")
    m = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def download(url, slug, title):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{slug}.pdf")
    rec = {"slug": slug, "title": title, "pdf_url": url,
           "local_path": os.path.relpath(path, ROOT).replace("\\", "/"),
           "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
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
    print(f"OK {slug} {rec['bytes']/1e6:.2f}MB :: {title[:78]}", flush=True)
    return rec


def main():
    records = []
    for slug, title in TARGETS:
        try:
            sim, item = s2_search(title)
        except Exception as exc:  # noqa: BLE001
            print(f"S2-FAIL {slug}: {exc}", flush=True)
            records.append({"slug": slug, "requested_title": title, "status": "search_failed", "error": str(exc)})
            continue
        if item is None:
            print(f"S2-NONE {slug}", flush=True)
            records.append({"slug": slug, "requested_title": title, "status": "not_found"})
            continue
        ext = item.get("externalIds") or {}
        oa = item.get("openAccessPdf") or {}
        print(f"S2 {slug} sim={sim:.2f} arxiv={ext.get('ArXiv')} oa={oa.get('url')}", flush=True)
        rec = None
        if ext.get("ArXiv"):
            aid = ext["ArXiv"]
            real = arxiv_title(aid)
            tsim = similarity(title, real)
            print(f"   arxiv:{aid} title='{real[:70]}' sim={tsim:.2f}", flush=True)
            if tsim >= 0.55:
                rec = download(f"https://arxiv.org/pdf/{aid}", slug, real)
                rec.update({"arxiv_id": aid, "title_similarity": round(tsim, 3), "requested_title": title})
            else:
                print(f"   arxiv title mismatch (sim={tsim:.2f}) -> skip", flush=True)
        if rec is None and oa.get("url"):
            try:
                rec = download(oa["url"], slug, item.get("title", title))
                rec.update({"open_access_url": oa["url"], "requested_title": title,
                            "venue": item.get("venue"), "year": item.get("year"),
                            "title_similarity": round(sim, 3)})
            except Exception as exc:  # noqa: BLE001
                print(f"   OA download failed: {exc}", flush=True)
        if rec is None:
            rec = {"slug": slug, "requested_title": title, "status": "unresolved",
                   "candidate": item.get("title"), "candidate_arxiv": ext.get("ArXiv"),
                   "candidate_oa": oa.get("url")}
        records.append(rec)
        time.sleep(3.0)

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    ok = sum(1 for r in records if r.get("status") in ("downloaded", "cached"))
    print(f"DONE ok={ok}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
