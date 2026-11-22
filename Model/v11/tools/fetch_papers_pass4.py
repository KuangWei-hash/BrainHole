#!/usr/bin/env python
"""第四輪：以 DBLP + arXiv 雙路徑解析並下載剩餘論文，逐筆記錄標題與來源。

修正前三輪因猜測 ID 或搜尋限流造成的錯誤 PDF。
輸出 Model/v11/來源清單-論文4.json（UTF-8, no BOM）。
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
MANIFEST = os.path.join(ROOT, "來源清單-論文4.json")
UA = "BrainHoleResearch/1.0 (local research download)"

TARGETS = [
    ("dbsr-official", "Learning a Deep Single Image Super-Resolution Network for Burst Images"),
    ("cycled-shift", "Learning to Super-Resolve Burst Photos"),
    ("realvsr", "Real-World Video Super-Resolution: A Benchmark Dataset and a Decomposition-based Learning Scheme"),
    ("flashvsr", "FlashVSR: Towards Real-time Video Super-Resolution"),
    ("frvsr", "Frame-Recurrent Super-Resolution"),
    ("tecogan", "Temporal GAN for Real-Time Video Super-Resolution"),
    ("motnet", "Learning Space-Time Video Super-Resolution with a Heterogeneous Network"),
]


def _ssl_context():
    try:
        import certifi
        import ssl
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return None


SSL_CTX = _ssl_context()


def http_get(url, retries=3, timeout=120):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def dblp_search(title):
    url = "https://dblp.org/search/publ/api?q=" + urllib.parse.quote(title) + "&format=json&h=8"
    data = json.loads(http_get(url).decode("utf-8", "replace"))
    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    out = []
    for h in hits:
        info = h.get("info", {})
        ees = info.get("ee", [])
        if isinstance(ees, dict):
            ees = [ees]
        for ee in ees:
            out.append({"title": info.get("title", ""), "url": ee.get("url", ""),
                        "year": info.get("year"), "venue": info.get("venue"),
                        "type": ee.get("type")})
    out.sort(key=lambda x: -sim(title, x["title"]))
    return out


def arxiv_candidates(title):
    queries = [f'ti:"{title}"', f'all:"{title}"']
    words = [w for w in re.split(r"[^A-Za-z0-9]+", title) if len(w) > 2][:8]
    if words:
        queries.append("ti:" + " AND ti:".join(words[:4]))
    seen = set()
    out = []
    for query in queries:
        try:
            url = ("http://export.arxiv.org/api/query?search_query="
                   + urllib.parse.quote(query) + "&max_results=12")
            xml = http_get(url).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            print(f"  arxiv query fail ({query[:40]}): {exc}", flush=True)
            continue
        for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
            eid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            if not eid or not t:
                continue
            aid = eid.group(1)
            if aid in seen:
                continue
            seen.add(aid)
            out.append({"id": aid, "title": re.sub(r"\s+", " ", t.group(1)).strip()})
        time.sleep(2.0)
    out.sort(key=lambda x: -sim(title, x["title"]))
    return out


def download(url, slug, title, source, extra=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{slug}.pdf")
    rec = {"slug": slug, "title": title, "pdf_url": url, "source": source,
           "local_path": os.path.relpath(path, ROOT).replace("\\", "/"),
           "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
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
    print(f"OK {slug} {rec['bytes']/1e6:.2f}MB [{source}] :: {title[:72]}", flush=True)
    return rec


def main():
    records = []
    for slug, title in TARGETS:
        rec = None
        try:
            cands = arxiv_candidates(title)
        except Exception as exc:  # noqa: BLE001
            cands = []
            print(f"ARXIV-FAIL {slug}: {exc}", flush=True)
        for c in cands[:3]:
            s = sim(title, c["title"])
            print(f"  arxiv cand {c['id']} sim={s:.2f} :: {c['title'][:66]}", flush=True)
            if s >= 0.6:
                rec = download(f"https://arxiv.org/pdf/{c['id']}", slug, c["title"], "arxiv",
                               {"arxiv_id": c["id"], "title_similarity": round(s, 3),
                                "requested_title": title})
                break
        if rec is None:
            try:
                hits = dblp_search(title)
            except Exception as exc:  # noqa: BLE001
                hits = []
                print(f"DBLP-FAIL {slug}: {exc}", flush=True)
            for h in hits[:4]:
                s = sim(title, h["title"])
                print(f"  dblp cand sim={s:.2f} [{h['type']}] {h['url'][:90]}", flush=True)
                if s >= 0.6 and h["url"].lower().endswith(".pdf"):
                    rec = download(h["url"], slug, h["title"], "dblp",
                                   {"title_similarity": round(s, 3), "requested_title": title,
                                    "venue": h["venue"], "year": h["year"]})
                    break
        if rec is None:
            rec = {"slug": slug, "requested_title": title, "status": "unresolved",
                   "arxiv_candidates": cands[:5]}
            print(f"UNRESOLVED {slug}", flush=True)
        records.append(rec)
        time.sleep(2.0)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    ok = sum(1 for r in records if r.get("status") == "downloaded")
    print(f"DONE ok={ok}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
