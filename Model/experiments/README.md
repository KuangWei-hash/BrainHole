# 實驗紀錄保存規則

總覽見 [實驗總表](../實驗總表.md)，文字解讀見各次 `Model/實驗紀錄-*.md`。
使用者要求所有實驗都留下紀錄，包括短測試、負面結果、失敗與中斷。

## 保存內容

- `records/*.json`：每次執行的狀態、原始 argv、起訖時間、資料來源、環境、來源程式快照位置、設定、每輪曲線、最佳輪、評估數值與產物清單。
- `details/*.blocks.jsonl.gz`：完整逐區塊評估結果，gzip 無損壓縮；紀錄中保存未壓縮內容的 SHA-256。
- `source_snapshots/<雜湊>/`：後續自動記錄執行時的 Python 程式、套件版本檔、模型規格、驗收標準及本目錄工作規則的不可變快照。
- `runs/`：權重、原始數值檔、圖片與影片的實際位置。紀錄保存路徑、大小與最終 SHA-256，原始產物不覆寫。

依 2026-11-22 使用者更新的 GitHub 保存範圍，Git 保留本文件與 `records/*.json` 的文字摘要、設定及曲線。`details/` 的完整逐區塊數值和 `source_snapshots/` 的重複程式快照加入忽略清單，仍照常產生並保存在本機；不因忽略而刪除紀錄。Git checkout 不包含這些本機證據，需要完整重現時須另行保留它們。
權重、圖片、影片本體仍在 `runs/`，沒有複製進本目錄。

## 自動記錄

既有 `train.py`、`evaluate.py`、`predict.py`、`baseline.py`、`benchmark.py` 入口已接入 `run_logged`。
每次 invocation 使用獨立紀錄 ID，執行前先寫 running；每輪訓練更新曲線，結束後寫 completed／failed／interrupted。
執行失敗保留例外與 traceback，重複使用舊輸出路徑的失敗嘗試不會冒用舊結果。
訓練或推理輸出不可覆寫；請另選 `--out`。

狀態與品質是兩件事：`completed` 只是程式執行完成，品質是否通過看評估 gates 與驗收紀錄。
`partial_steps` 表示訓練或驗證被步數限制截短，不能當完整模型成績。
總表的驗證數量為「實際評估／可用集合」；截短實驗按 batch size 與 val-steps 推導實際評估數。
被強制終止或斷電可能保留最後 running 狀態與已存曲線，不能自行當作完成。

資料指紋是執行前檔案的 SHA-256；SQLite 如有非空 WAL，也登記該檔。
這不是資料庫的交易式備份。資料來源與快取應固定，避免實驗期間修改資料。
環境中的 available_gpu 表示硬體可用，不代表該次一定用 GPU；實際裝置以訓練 config 或量測 report 為準。

新增腳本使用 `run_logged`，直接呼叫底層函式或互動式臨時實驗則須自行補建等價紀錄。
測試、視覺觀察與比較結論仍要寫入 Markdown，不能只留下 JSON。

## 與專案 README 同步

每次新結果或目視／方向判斷產生後，先保存本目錄紀錄及對應的 Markdown 解讀，再依 [模型工作規則](../AGENTS.md) 檢查根目錄 `README.md` 和 `Model/README.md`。首頁維持最終目標、最新研究結論、目前視覺基準、限制與下一步的摘要，注明狀態日期並連結實驗證據；完整歷史仍由本目錄及實驗總表承載。

失敗、退步與中斷也可能改變研究方向，須一併檢查。流程檢查未改變專案結論時，保留詳細紀錄即可；較低 MAE 不自動代表通過目視驗收。

## 歷史補登

```powershell
.\Model\.venv\Scripts\python.exe Model/record_experiment.py --backfill
```

從 `runs/` 補登可確認的設定、每輪曲線、報告、診斷與產物，遇到已登記的輸出不再新增副本。
歷史原始指令、起訖時間、原始程式快照沒有保存的欄位為 null，不使用目前環境冒充當時環境。
目前快照無法補回以前已修改的程式，因此歷史補登不是完整重現保證。

只更新總表：

```powershell
.\Model\.venv\Scripts\python.exe Model/record_experiment.py
```

既有 JSON 與數值是原始觀測，不應為了符合後來的解釋而改寫。需要修正時另外註明原因與來源。
