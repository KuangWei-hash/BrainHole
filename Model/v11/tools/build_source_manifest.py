"""產生 Model/v11/來源清單.json：repo（URL、commit、授權、位元組數）＋論文（標題、SHA-256）
＋量測產物（JSON、腳本）。

輸出 UTF-8, no BOM。
"""
import hashlib
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.path.join(ROOT, "third_party")
PAPER_DIR = os.path.join(ROOT, "refs", "papers")

REPO_META = {
    "BasicVSR_PlusPlus": ("https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git", "Apache-2.0",
                          "flow-guided deformable alignment + grid propagation（雙向，違反因果性）"),
    "RVRT": ("https://github.com/JingyunLiang/RVRT.git", "CC BY-NC-4.0",
             "遞迴 VSR + 引導式可變形注意力"),
    "FlashVSR": ("https://github.com/OpenImagingLab/FlashVSR.git", "Apache-2.0",
                 "即時擴散式 VSR（對應 1.11ms/區塊預算）"),
    "deep-burst-sr": ("https://github.com/goutamgmb/deep-burst-sr.git", "CC BY-NC-SA-4.0（非商用）",
                      "DBSR：自適應加權 burst 融合 + 空間色對齊"),
    "TTVSR": ("https://github.com/researchmm/TTVSR.git", "MIT",
              "測試時訓練（離線，違反因果性，僅作對照）"),
    "TecoGAN": ("https://github.com/thunil/TecoGAN.git", "Apache-2.0",
                "FRVSR + 時序 GAN"),
    "DiffVSR": ("https://github.com/xh9998/DiffVSR.git", "無 LICENSE 檔（授權不明）",
                "漸進式退化處理與潛在過渡"),
    "Awesome-Super-Resolution": ("https://github.com/ChaofWang/Awesome-Super-Resolution.git",
                                  "無 LICENSE 檔", "SR 論文索引"),
    "awesome-video-enhancement": ("https://github.com/liuzhen03/awesome-video-enhancement.git",
                                  "無 LICENSE 檔", "影片增強論文索引"),
    "Video_Super_Resolution_Ref": ("https://github.com/qianx77/Video_Super_Resolution_Ref.git",
                                   "無 LICENSE 檔", "真實世界 VSR 索引"),
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.islink(fp):
                continue
            total += os.path.getsize(fp)
    return total


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    repos = []
    for name in sorted(os.listdir(TP)):
        path = os.path.join(TP, name)
        if not os.path.isdir(path):
            continue
        url, lic, role = REPO_META.get(name, (None, "未登錄", "未登錄"))
        repos.append({
            "name": name,
            "path": os.path.relpath(path, ROOT).replace("\\", "/"),
            "url": url,
            "commit": git_sha(path),
            "license": lic,
            "role": role,
            "bytes_on_disk": dir_bytes(path),
        })

    papers = []
    catalog_path = os.path.join(PAPER_DIR, "論文清單.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, encoding="utf-8") as fh:
            for item in json.load(fh):
                papers.append(item)

    runs = []
    for name in sorted(os.listdir(ROOT)):
        if name.startswith("runs-") and name.endswith(".json"):
            p = os.path.join(ROOT, name)
            runs.append({
                "file": name,
                "bytes": os.path.getsize(p),
                "sha256": sha256_of(p),
            })

    tools = []
    tdir = os.path.join(ROOT, "tools")
    for name in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, name)
        if os.path.isfile(p):
            tools.append({"file": f"tools/{name}", "bytes": os.path.getsize(p),
                          "sha256": sha256_of(p)})

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "外部程式碼與論文僅供研究參考；各項授權見 license 欄位，非商用授權者不得直接複製其程式碼。",
        "repos": repos,
        "papers": papers,
        "measurement_outputs": runs,
        "tools": tools,
    }
    path = os.path.join(ROOT, "來源清單.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"repos={len(repos)} papers={len(papers)} runs={len(runs)} tools={len(tools)} -> {path}")


if __name__ == "__main__":
    main()
