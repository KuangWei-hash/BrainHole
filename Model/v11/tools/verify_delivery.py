"""交付前檢查：Model/v11 的 JSON 可解析（無 BOM）、Markdown 內相對連結是否存在。"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def check_json():
    bad = []
    count = 0
    for base, _, files in os.walk(ROOT):
        if "third_party" in base:
            continue
        for name in files:
            if not name.endswith(".json"):
                continue
            p = os.path.join(base, name)
            count += 1
            with open(p, "rb") as fh:
                head = fh.read(3)
            if head == b"\xef\xbb\xbf":
                bad.append(f"BOM: {os.path.relpath(p, REPO)}")
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    json.load(fh)
            except Exception as exc:  # noqa: BLE001
                bad.append(f"PARSE {os.path.relpath(p, REPO)}: {exc}")
    return count, bad


def check_links():
    missing = []
    checked = 0
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ("third_party",)]
        for name in files:
            if not name.endswith(".md"):
                continue
            p = os.path.join(base, name)
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            # 去掉 fenced code block，避免把程式碼中的 [x](y) 誤判為連結
            text = re.sub(r"```.*?```", "", text, flags=re.S)
            text = re.sub(r"`[^`\n]*`", "", text)
            for target in LINK_RE.findall(text):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                rel = target.split("#")[0]
                if not rel:
                    continue
                checked += 1
                resolved = os.path.normpath(os.path.join(base, rel.replace("/", os.sep)))
                if not os.path.exists(resolved):
                    missing.append(f"{os.path.relpath(p, REPO)} -> {target}")
    return checked, missing


def main():
    jcount, jbad = check_json()
    lcount, lmissing = check_links()
    print(f"JSON 檢查：{jcount} 個檔案，問題 {len(jbad)}")
    for item in jbad:
        print("  ", item)
    print(f"連結檢查：{lcount} 條相對連結，缺 {len(lmissing)}")
    for item in lmissing:
        print("  ", item)
    return 1 if (jbad or lmissing) else 0


if __name__ == "__main__":
    sys.exit(main())
