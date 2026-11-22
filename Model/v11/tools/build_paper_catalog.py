"""建立已驗證的論文清單：掃描 refs/papers，逐篇以 arXiv API 回查標題。

- 檔名含 arXiv id 者直接回查。
- 其餘由來源清單（來源清單-論文*.json）或手動對照取得 id。
- 回查不到的標示 verified=false，並記錄實際取得方式（cvf 等）。

輸出 refs/papers/論文清單.json（UTF-8, no BOM）。
"""
import hashlib
import json
import os
import re
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(ROOT, "refs", "papers")
CATALOG = os.path.join(PAPER_DIR, "論文清單.json")
UA = "BrainHoleResearch/1.0 (local research download)"

# 手動對照：檔名 slug -> (來源, arXiv id)；CVF 官方 PDF 以檔名 slug 為標題依據
MANUAL = {
    "dbsr-official.pdf": ("CVF 官方 PDF（CVPR 2021）", None),
    "realvsr.pdf": ("CVF 官方 PDF（ICCV 2021）", None),
    "real-raw-vsr.pdf": ("arXiv", "2209.12475"),
    "flashvsr.pdf": ("arXiv", "2510.12747"),
    "frvsr.pdf": ("arXiv", "1801.04590"),
    "motnet.pdf": ("arXiv", "2102.13011"),
    "tecogan.pdf": ("arXiv", "1811.09393"),
    "genmfsr.pdf": ("arXiv", "2603.19187"),
    "stablevsr.pdf": ("arXiv", "2311.15908"),
}
CVF_TITLES = {
    "dbsr-official.pdf": ("Deep Burst Super-Resolution",
                          "https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf"),
    "realvsr.pdf": ("Real-World Video Super-Resolution: A Benchmark Dataset and a Decomposition-Based Learning Scheme",
                     "https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Real-World_Video_Super-Resolution_A_Benchmark_Dataset_and_a_Decomposition_Based_ICCV_2021_paper.pdf"),
}


def http_get(url, retries=3, timeout=60):
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
    p = re.search(r"<entry>.*?<published>([^<]+)</published>", xml, re.S)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    return title, (p.group(1)[:10] if p else None)


def main():
    entries = []
    for name in sorted(os.listdir(PAPER_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(PAPER_DIR, name)
        with open(path, "rb") as fh:
            data = fh.read()
        rec = {
            "file": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source": None,
            "arxiv_id": None,
            "title": None,
            "published": None,
            "verified": False,
        }
        m = re.search(r"arXiv-([0-9]{4}\.[0-9]{4,5})v?\d*", name)
        source = "arXiv"
        aid = m.group(1) if m else None
        if aid is None and name in MANUAL:
            source, aid = MANUAL[name]
        rec["source"] = source
        rec["arxiv_id"] = aid
        if aid:
            try:
                title, published = arxiv_title(aid)
                rec["title"] = title
                rec["published"] = published
                rec["verified"] = bool(title)
                rec["abs_url"] = f"https://arxiv.org/abs/{aid}"
            except Exception as exc:  # noqa: BLE001
                rec["error"] = str(exc)
            time.sleep(1.5)
        elif name in CVF_TITLES:
            title, url = CVF_TITLES[name]
            rec["title"] = title
            rec["pdf_url"] = url
            rec["verified"] = True
            rec["verification"] = "標題取自 CVF 官方檔名（非 arXiv 回查）"
        entries.append(rec)
        print(f"{'OK ' if rec['verified'] else '?? '} {name:<58} {rec['title'] or '(未回查)'}", flush=True)

    with open(CATALOG, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)
    ok = sum(1 for e in entries if e["verified"])
    print(f"DONE verified={ok}/{len(entries)} catalog={CATALOG}", flush=True)


if __name__ == "__main__":
    main()
