# v9 三層 16 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，以目前較佳的兩層 relu16、3×3 為基礎，加一層中間卷積：`Conv3×3 4→16 → ReLU → Conv3×3 16→16 → ReLU → Conv3×3 16→4`。設定 `--v9-layers 3 --v9-variant relu16 --v9-kernel-width 3`，共 **3,492 參數／FP32 純參數 13,968 bytes**，原兩層為 1,172 參數。此次同時增加深度、ReLU 次數、參數量與感受野，不能將結果完全歸因於其中一項。

仍在高解析度四平面 8×320 上卷積，平面感受野從 5×5 增為 7×7，同色位置輸出 RAW 半徑從 ±4 增為 ±6 像素；非先前 3×9 寬核實驗。只輸入本幀本條實際 decoded RAW＋phase，沿用固定相位對齊 bilinear、四平面殘差修正與 clamp [0,1]，輸出 640×16 RAW8 RGGB；沒有 encoded 學習分支、前幀或鄰條。末層零初始化，初始等於 bilinear。

CLI 仍預設兩層，三層明確指定；舊 checkpoint 缺 layers 時仍以兩層重建。三層 FSR 評估名稱為 `v9-relu16-layers3`，避免與既有 relu16 混淆。warm-start 要求相同深度、kernel 與 variant；本次從頭訓練。

## 資料與比較契約

沿用 Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`）、`runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，完整五輪、不截短、不 warm-start，按最低驗證 MAE 選 best.pt。

主要比較基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、兩層 relu16 與 bilinear。兩層與寬核結果引用同資料指紋、快取、窗口及五輪預算的既有 report，不假稱重跑。RAW 投影與 RGB 參考分開；RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片保留集僅供數值初篩；完整視覺、新影片泛化與即時串流效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作完成，36 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 3 輪。新增三層測試驗證 Conv／ReLU 排列、通道、kernel 與參數大小、16 相零初始化等於 bilinear、三層權重及 decoded 梯度、encoded 不變性、checkpoint 重建；既有測試驗證舊版相容性。[測試 log](./runs/checks-v9-relu16-three-q5-all/tests.log) 已保存。run_logged／update_active_record 保存所有狀態、曲線、快照與指紋，不覆寫舊產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu16-three-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.893373 | 16.640248 | 20.059127 |
| 2 | 12.534435 | 16.407851 | 20.143319 |
| **3** | **12.446166** | **16.244480** | **20.183360** |
| 4 | 12.382772 | 16.319053 | 20.143291 |
| 5 | 12.343886 | 16.277620 | 20.220576 |

[best.pt](./runs/relu16-three-v9-q5/best.pt) 按最低驗證 MAE 選第 3 輪；第 5 輪 PSNR 較高，但 MAE 較差，不改選模規則。[last.pt](./runs/relu16-three-v9-q5/last.pt) 為第 5 輪，設定見 [config.json](./runs/relu16-three-v9-q5/config.json)。

## 完整評估與解讀

Q=5 frame 720…899、5,400 條，以五輪中最低驗證 MAE 的第 3 輪評估。本次來源為 [FSR report](./runs/eval-fsr1-v9-relu16-three-q5/report.json)；兩層 relu16 來源為 [既有 report](./runs/eval-fsr1-v9-relu16-q5/report.json)（五輪最佳第 4 輪），寬核版為 [既有 3×9 report](./runs/eval-fsr1-v9-relu16-wide-q5/report.json)（五輪最佳第 4 輪）。各自使用相同資料、快取、驗證窗口與五輪預算；FSR 及 bilinear 數值相同。舊版未重跑或覆寫。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **relu16 三層、3×3** | **3,492** | **16.244480** | **20.183361** | **14.801605** | **21.008268** |
| relu16 兩層、3×3 | 1,172 | 16.321658 | 20.152144 | 14.933389 | 20.956188 |
| relu16 兩層、3×9 | 3,476 | 16.497208 | 20.080343 | 15.047679 | 20.913139 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

三層版相較兩層版的 RAW MAE 降低 0.077177（約 **0.47%**），PSNR 增加 0.031216 dB；RGB MAE 降低 0.131785（約 0.88%）、PSNR 增加 0.052080 dB。以 frame/block 對齊兩份逐條結果，三層勝過兩層的 RAW MAE 計數為 **3,311／5,400（61.31%）**。這次加深帶來小幅平均改善，但不是每條都改善。

本次 [逐條結果](./runs/eval-fsr1-v9-relu16-three-q5/blocks.jsonl) 的 RAW MAE 嚴格低於對照：FSR＋RCAS 為 **4,582／5,400（84.85%）**，無 RCAS FSR 為 4,431／5,400（82.06%），RAW bilinear 為 4,827／5,400（89.39%）。對 FSR＋RCAS 勝率低於兩層版的 88.24%，即使平均誤差更低，也不能宣稱各區塊的改善更一致。按獨立場景分析與退化區塊分析仍待完成。

參數約為兩層版 2.98 倍，收益有限；三層可作較佳平均數值候選，但單 seed、同影片且持續以此保留集選模，不能證明穩健泛化。kernel 與 layers 預設保持原兩層 3×3，三層以 `--v9-layers 3` 明確選擇，供後續品質／效能權衡。

已查看 frame 750 比較 PNG，三層標籤與排列正常，仍有水平條帶、模糊與細節流失。離線影片已輸出 180 幀／30 FPS，未完整觀看；不宣告視覺、時間穩定性、新影片泛化或即時效能通過。不能將離線批次評估當作串流速度驗收。

## 產物與自動紀錄

- [最佳權重](./runs/relu16-three-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu16-three-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu16-three-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu16-three-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu16-three-q5-all--run-770f559ae2e1.json)、[訓練紀錄](./experiments/records/training--relu16-three-v9-q5--run-9a2af37faf4c.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu16-three-q5--run-4413f19c1780.json)
- [自動實驗索引](./實驗總表.md)，保存完整狀態、曲線、快照與資料／權重指紋。

## 重現

在儲存庫根目錄執行，重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu16-three-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 3 --v9-variant relu16 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu16-three-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu16-three-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu16-three-q5
```
