# v9 四層 24 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，將目前四層 relu16 擴寬為 **4→24→24→24→4**，四層均為 3×3，前三層後各有 ReLU。共 **12,172 參數／FP32 純參數 48,688 bytes**；四層 16 通道為 5,812 參數。此次只改中間寬度，深度、kernel 與啟用函數排列相同；參數與運算量增加，效能須另外實測。

高解析度 CFA 四平面網格為 8×320，四層感受野仍 9×9，同色位置輸出 RAW 半徑 ±8 像素；擴寬不增加視野。只讀本幀本條實際 decoded RAW＋phase，固定相位對齊 bilinear、卷積殘差加回、交錯與 clamp [0,1]，輸出 640×16 RAW8 RGGB。沒有 encoded 分支、前幀或鄰條。末層零初始化，初始等於 bilinear。

明確指定 `--v9-layers 4 --v9-variant relu24 --v9-kernel-width 3`；CLI 預設保留兩層 relu16。舊 checkpoint 與 variants 保持相容，FSR 評估名稱為 `v9-relu24-layers4`。

## 資料與比較契約

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`，SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`，快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪、不 warm-start、不截短，依最低驗證 MAE 選 best.pt。

主要比較基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、四層 relu16 及 bilinear。四層 relu16 引用相同資料、快取、窗口與五輪預算的既有 report（最佳第 5 輪），不假稱重跑。RAW 投影與 RGB 參考分開報告，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片且反覆選模，僅供初篩，不能證明新影片泛化；視覺、獨立影片與即時效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作完成，39 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 5 輪。新增四層 relu24 測試驗證通道、kernel、Conv／ReLU 結構與參數大小，16 相零初始化等於 bilinear、四層及 decoded 梯度、encoded 不變性、checkpoint 重建；舊版測試同時通過。[測試 log](./runs/checks-v9-relu24-four-q5-all/tests.log) 已保存。run_logged／update_active_record 保存所有狀態、曲線、快照與指紋，不覆寫舊產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu24-four-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.644371 | 16.255954 | 20.207194 |
| 2 | 12.227681 | 16.147017 | 20.235881 |
| 3 | 12.087758 | 15.913706 | 20.354037 |
| 4 | 11.999864 | 15.967358 | 20.348202 |
| **5** | **11.925323** | **15.875731** | **20.355058** |

[best.pt](./runs/relu24-four-v9-q5/best.pt) 與 [last.pt](./runs/relu24-four-v9-q5/last.pt) 均為第 5 輪，設定見 [config.json](./runs/relu24-four-v9-q5/config.json)。

## 完整評估與解讀

Q=5 frame 720…899、5,400 條，以五輪最低驗證 MAE 的第 5 輪評估。來源為 [本次 report](./runs/eval-fsr1-v9-relu24-four-q5/report.json)；16 通道來源為 [既有四層 report](./runs/eval-fsr1-v9-relu16-four-q5/report.json)，也是五輪最佳第 5 輪。資料、快取、窗口與五輪預算一致，FSR／bilinear 數值相同；未重跑或覆寫舊模型。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **四層 relu24** | **12,172** | **15.875731** | **20.355058** | **14.553716** | **21.175977** |
| 四層 relu16 | 5,812 | 16.028068 | 20.323953 | 14.642700 | 21.145604 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

24 通道相較 16 通道的 RAW MAE 降低 0.152337（約 **0.95%**），PSNR 增加 0.031106 dB；RGB MAE 降低 0.088984（約 0.61%），PSNR 增加 0.030374 dB。參數增加 6,360（約 **109.43%**、總數為 2.09 倍）；本次擴寬有小幅平均品質效益，成本亦明顯增加，不能從參數量直接推斷實際速度。

依 frame/block 對齊 [本次逐條結果](./runs/eval-fsr1-v9-relu24-four-q5/blocks.jsonl) 與 16 通道逐條結果，24 通道 RAW MAE 嚴格勝過 16 通道為 **3,432／5,400（63.56%）**。對 FSR＋RCAS 為 **5,084／5,400（94.15%）**，高於 16 通道的 91.83%；無 RCAS FSR 為 5,030／5,400（93.15%），RAW bilinear 為 5,284／5,400（97.85%）。仍有退化區塊，按場景與失敗條分析尚待完成。

本次四層 24 通道是這組寬度比較中較佳的平均數值候選，但只是單 seed、同影片與五輪預算，不能保證繼續擴寬有益或新影片表現更佳。CLI 預設保持兩層 relu16，四層 relu24 以明確選項選擇；未量測效能，不宣稱部署更合適。

已查看 frame 750 比較 PNG，24 通道標籤與排列正常，仍有水平條帶、模糊與細節流失。離線影片已輸出 180 幀／30 FPS，未完整觀看；不宣告視覺、時間穩定性、新影片或即時串流驗收通過。所有負面與成功觀測均保留。

## 產物與自動紀錄

- [最佳權重](./runs/relu24-four-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu24-four-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu24-four-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu24-four-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu24-four-q5-all--run-36356cc4dae0.json)、[訓練紀錄](./experiments/records/training--relu24-four-v9-q5--run-1c49220ccb1a.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu24-four-q5--run-0a71d409aef1.json)
- [自動實驗索引](./實驗總表.md)，保存完整狀態、曲線、快照與資料／權重指紋。

## 重現

在儲存庫根目錄執行，重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu24-four-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 4 --v9-variant relu24 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu24-four-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu24-four-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu24-four-q5
```
