# v9 無 ReLU：4→32→4，Q=5

## 目的與設定

依使用者要求將 v9 中間通道從 16 擴至 32：`Conv3×3 4→32 → Conv3×3 32→4`，無 ReLU，共 2,340 參數，FP32 純參數 9,360 bytes。CLI 預設 `--v9-layers 2 --v9-variant plain32`；舊 variants 與權重保留相容。FSR 評估標籤改從 checkpoint 設定取得，避免將 plain32 誤標為 plain16。

只用本幀本條 decoded RAW＋phase，固定相位對齊 bilinear 放大、兩層卷積殘差修正，輸出 640×16 RAW RGGB；無 encoded 分支、歷史幀、鄰條或中間啟用函數。末層零初始化。

為與 plain16 隔離通道數影響，沿用 Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`）、`runs/decoded-q5-v4.npz`。train frame 0…719，共 21,600 條；validation frame 720…899，共 5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪，不截短、不 warm-start，最低驗證 MAE 選 best.pt。此次不混入 Q=4 或獨立影片資料。

主要比較基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、plain16 與 bilinear。RGB 參考為目標 RAW 的共同顯色結果，與 RAW 投影指標分開報告。單 seed、同影片保留集不代表新影片泛化；視覺與串流效能另行驗收。

## 執行狀態

實作完成，32 項測試全數通過，完整五輪訓練與 FSR 完整評估均已完成，最佳第 3 輪。新增測試涵蓋通道／參數量、16 相初始等於 bilinear、兩層與 decoded 梯度、encoded 不影響輸出、checkpoint 重建；既有測試同時驗證舊版與官方 FSR tile 隔離。測試輸出為 `runs/checks-v9-plain32-q5-all/tests.log`。所有執行透過 run_logged 留存，逐輪更新曲線，不覆寫舊產物。

首次測試啟動使用 PowerShell 傳入 Python `-c`，因引號被移除而於解析階段發生 SyntaxError，尚未載入測試或 run_logged；改用可保存輸出與原始 argv 的 `run_checks.py --out` 入口重試。

## 訓練曲線

來源：[逐輪原始數值](./runs/plain32-v9-q5/metrics.jsonl)。每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 13.406169 | 16.774693 | 20.012983 |
| 2 | 13.302527 | 16.725497 | 20.060407 |
| **3** | **13.281934** | **16.684918** | **20.065681** |
| 4 | 13.273739 | 16.725033 | 20.037406 |
| 5 | 13.268684 | 16.711931 | 20.045365 |

最佳第 3 輪存為 [best.pt](./runs/plain32-v9-q5/best.pt)，第 5 輪為 [last.pt](./runs/plain32-v9-q5/last.pt)，設定見 [config.json](./runs/plain32-v9-q5/config.json)。

## 完整評估結果

frame 720…899、5,400 條，使用最佳第 3 輪；RAW／RGB 指標來源為 [本次 report](./runs/eval-fsr1-v9-plain32-q5/report.json)。plain16 列引自 [既有 FSR report](./runs/eval-fsr1-v9-q5/report.json)，同資料指紋、快取、窗口與五輪預算，最佳為第 5 輪；本次沒有重跑 plain16。FSR 與 bilinear 數值和該次 report 一致。

| 方法 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: |
| v9 plain32 | 16.684918 | 20.065681 | 15.562386 | 20.765904 |
| v9 plain16（既有同條件） | 16.626771 | 20.079337 | 15.482858 | 20.788811 |
| FSR 1 EASU 2x＋2x＋RCAS 0 | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR 1 EASU 2x＋2x | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

plain32 的 RAW MAE 相較 plain16 增加 0.058146（約 0.35%），PSNR 降低 0.013656 dB；RGB 兩項也略差。參數從 1,172 增至 2,340，此次五輪、單 seed 擴寬沒有改善，plain16 仍是這兩個版本中較佳的數值候選。此負面結果保留，不以挑選更多輪或不同資料取代。

plain32 的平均 RAW／RGB 指標仍優於兩種 FSR 對照。依本次 [逐條原始結果](./runs/eval-fsr1-v9-plain32-q5/blocks.jsonl) 的 RAW MAE 嚴格小於對照計數：勝 RCAS FSR 為 5,250／5,400（97.22%），勝無 RCAS FSR 為 4,992／5,400（92.44%），勝 RAW bilinear 為 5,396／5,400（99.93%）。這是數值初篩結果，不是所有條都改善，也不是完整品質驗收通過；未作場景分組分析。

已查看 frame 750 的六欄比較 PNG，方法標籤與排列正常，plain32 仍有明顯水平條帶、模糊及細節流失。離線比較影片產生 180 幀／30 FPS；未完整觀看影片，不宣告視覺或時間穩定性驗收通過。獨立影片泛化與即時串流效能尚未驗證。

## 產物與自動紀錄

- [六欄比較圖](./runs/eval-fsr1-v9-plain32-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-plain32-q5/comparison.mp4)、[官方 CLI 中間產物](./runs/eval-fsr1-v9-plain32-q5/official-cli/)
- [測試紀錄](./experiments/records/checks--checks-v9-plain32-q5-all--run-07aa63b4d522.json)、[訓練紀錄](./experiments/records/training--plain32-v9-q5--run-85c5a3c01af3.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-plain32-q5--run-4237fcd22858.json)
- [自動實驗索引](./實驗總表.md)，包含資料與權重指紋、程式快照與完整狀態。

## 重現

在儲存庫根目錄執行；重跑須選新的輸出目錄：

```powershell
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 2 --v9-variant plain32 --history 1 --device cuda --epochs 5 --out Model/runs/plain32-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/plain32-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-plain32-q5
```
