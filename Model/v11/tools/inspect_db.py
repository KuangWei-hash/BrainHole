import glob
import os
import sqlite3

root = r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料"
paths = sorted(glob.glob(os.path.join(root, "*q4*.sqlite")))
for p in paths:
    print("=" * 70)
    print(os.path.basename(p), f"{os.path.getsize(p)/1e9:.3f} GB")
    c = sqlite3.connect(p)
    tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    print("  tables:", tables)
    cols = [(r[1], r[2]) for r in c.execute("pragma table_info(samples)")]
    print("  samples cols:", cols)
    n = list(c.execute("select count(*) from samples"))[0][0]
    print("  rows:", n)
    row = list(c.execute("select * from samples limit 1"))[0]
    names = [r[1] for r in c.execute("pragma table_info(samples)")]
    for name, val in zip(names, row):
        kind = f"len={len(val)}" if isinstance(val, (bytes, str)) else repr(val)
        print(f"    {name}: {kind}")
    # frame/block structure
    for q in ("select min(frame), max(frame) from samples",
              "select count(distinct frame) from samples",
              "select count(distinct block_index) from samples"):
        try:
            print("  ", q, "->", list(c.execute(q)))
        except Exception as exc:
            print("  ", q, "-> ERR", exc)
    c.close()
