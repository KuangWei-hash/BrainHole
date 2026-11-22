# v9 四層 256 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，將四層 relu128 擴寬為 **4→256→256→256→4**，四層均為 3×3，前三層後各有 ReLU。共 **1,198,852 參數／FP32 純參數 4,795,408 bytes**；128／64 通道分別為 304,516／78,532 參數。只改中間寬度，深度、kernel 與 ReLU 排列相同；效能與部署成本另行驗證。

卷積仍在高解析度 CFA 四平面 8×320 網格上執行，感受野仍 9×9，同色位置輸出 RAW 半徑 ±8 像素。只讀本幀本條實際 decoded RAW＋phase，維持固定相位對齊 bilinear、卷積殘差加回、交錯與 clamp [0,1]，輸出 640×16 RAW8 RGGB。沒有 encoded 分支、歷史幀或鄰條。末層零初始化，初始等於 bilinear。

明確指定 `--v9-layers 4 --v9-variant relu256 --v9-kernel-width 3`；CLI 預設仍兩層 relu16，舊 variants 與權重保持相容。FSR 評估名稱為 `v9-relu256-layers4`。

## 資料與比較契約

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`），快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪、不截短、不 warm-start，按最低驗證 MAE 選 best.pt。

主要基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、四層 128／64／32 通道及 bilinear。舊模型引用相同資料、快取、窗口與五輪預算的既有 report，128／64 通道最佳第 4 輪、32 通道第 3 輪，不假稱重跑。RAW 投影與 RGB 參考分開，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片反覆選模只供數值初篩，不能證明新影片泛化；完整視覺、新影片與即時串流效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作完成，43 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 3 輪。新增測試驗證四層 256 通道 Conv／ReLU 結構、kernel 與參數大小、16 相初始等於 bilinear、四層及 decoded 梯度、encoded 不變性與 checkpoint 重建；舊版測試同時通過。[測試 log](./runs/checks-v9-relu256-four-q5-all/tests.log) 已保存。run_logged／update_active_record 保存所有狀態、曲線、快照與指紋，不覆寫舊產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu256-four-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.603777 | 15.849505 | 20.351354 |
| 2 | 11.906084 | 15.659895 | 20.440798 |
| **3** | **11.670564** | **15.513671** | **20.484319** |
| 4 | 11.542025 | 15.571101 | 20.478517 |
| 5 | 11.431012 | 15.562579 | 20.472622 |

[best.pt](./runs/relu256-four-v9-q5/best.pt) 為第 3 輪，[last.pt](./runs/relu256-four-v9-q5/last.pt) 為第 5 輪，設定見 [config.json](./runs/relu256-four-v9-q5/config.json)。每輪訓練約 143 秒、驗證約 16 秒，為離線過程耗時，不是即時串流延遲。

## 完整評估與解讀

Q=5 frame 720…899、5,400 條，以五輪最低驗證 MAE 的第 3 輪評估。來源為 [本次 report](./runs/eval-fsr1-v9-relu256-four-q5/report.json)；其他寬度引用 [128 通道 report](./runs/eval-fsr1-v9-relu128-four-q5/report.json)、[64 通道 report](./runs/eval-fsr1-v9-relu64-four-q5/report.json)、[32 通道 report](./runs/eval-fsr1-v9-relu32-four-q5/report.json)。資料、快取、窗口及五輪預算一致，FSR／bilinear 數值相同；舊模型未重跑或覆寫。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 四層 relu256 | 1,198,852 | 15.513671 | 20.484319 | 14.185627 | 21.320860 |
| 四層 relu128 | 304,516 | 15.234079 | 20.551496 | 13.830817 | 21.415726 |
| 四層 relu64 | 78,532 | 15.548606 | 20.430942 | 14.135203 | 21.275638 |
| 四層 relu32 | 20,836 | 15.731815 | 20.350482 | 14.302455 | 21.195823 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

256 通道相較 128 通道的 RAW MAE 增加 0.279592（約 **1.84%**）、PSNR 降低 0.067177 dB；RGB MAE 增加 0.354809（約 2.57%）、PSNR 降低 0.094866 dB。四項平均指標均退步，這次相同五輪、單 seed 的擴寬沒有改善，**128 通道仍是這組寬度比較較佳的平均數值候選**。不能單憑此結果斷定過擬合、最佳深度／寬度上限或更長訓練必然有效。

參數比 128 通道增加 894,336，總數約 **3.94 倍**，純 FP32 參數約 4.80 MB。每輪訓練約 143 秒，比 128 通道約 44 秒更久，但這是離線訓練耗時，非 batch=1 串流延遲，不宣稱已完成部署效能驗收。品質未改善、成本明顯增加，本次不建議以 256 通道替換 128 通道。

依 frame/block 對齊 [本次逐條結果](./runs/eval-fsr1-v9-relu256-four-q5/blocks.jsonl) 與 128 通道逐條結果，256 通道 RAW MAE 嚴格勝過 128 通道僅 **908／5,400（16.81%）**。對主要 FSR＋RCAS 為 **5,347／5,400（99.02%）**，低於 128 通道的 99.30%；無 RCAS FSR 為 5,321／5,400（98.54%），RAW bilinear 為 5,383／5,400（99.69%）。平均指標仍優於 FSR，不代表擴寬優於原模型。按獨立場景與退化原因分析尚待完成。

單 seed、同影片反覆選模的限制保留，不能宣告新影片泛化。CLI 預設仍兩層 relu16，四層 relu256 明確指定；所有舊權重與此次負面結果均保留。

已查看 frame 750 六欄比較 PNG，256 通道標籤與排列正常，仍有水平條帶、模糊與細節流失。離線比較影片已輸出 180 幀／30 FPS，未完整觀看；不宣告完整視覺、時間穩定性、新影片或即時串流驗收通過。

## 產物與自動紀錄

- [最佳權重](./runs/relu256-four-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu256-four-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu256-four-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu256-four-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu256-four-q5-all--run-9545581f6823.json)、[訓練紀錄](./experiments/records/training--relu256-four-v9-q5--run-2c83878f3c51.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu256-four-q5--run-6d96a6e4b9c5.json)
- [自動實驗索引](./實驗總表.md)，保存完整曲線、狀態、快照與資料／權重指紋。

## 重現

在儲存庫根目錄執行；重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu256-four-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 4 --v9-variant relu256 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu256-four-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu256-four-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu256-four-q5
```
