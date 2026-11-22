# v9 四層 16 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，將三層 relu16 再加一層：**4→16→16→16→4**，四層均為 3×3，前三層後各有 ReLU。共 **5,812 參數／FP32 純參數 23,248 bytes**；三層版為 3,492 參數。本次同時增加深度、ReLU 次數、容量與感受野，不能將效果完全歸因於其中一項。

卷積仍在高解析度 CFA 四平面 8×320 上執行，平面感受野 9×9，同色位置輸出 RAW 半徑 ±8 像素。只用本幀本條實際 decoded RAW＋phase，固定相位對齊 bilinear、四平面殘差加回、交錯與 clamp [0,1]，輸出 640×16 RAW8 RGGB；沒有 encoded 學習分支、歷史幀或鄰條。末層零初始化，初始等於 bilinear。

指定 `--v9-layers 4 --v9-variant relu16 --v9-kernel-width 3`。預設仍為兩層，缺省 layers／kernel 的舊 checkpoint 仍按兩層／3×3 載入；舊版結構與權重保持相容。FSR 評估名稱為 `v9-relu16-layers4`。

## 資料與比較契約

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`，SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`，快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪，不 warm-start、不截短，按最低驗證 MAE 選 best.pt。

主要比較基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、三層／兩層 relu16 與 bilinear。舊模型引用同資料、快取、窗口及五輪預算的既有 report，不假稱重跑；分別選三層最佳第 3 輪、兩層最佳第 4 輪。RAW 投影與 RGB 參考分開報告，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片保留集且反覆用於選模，僅供初步研究，不能證明新影片泛化。完整視覺、獨立影片與即時串流效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作完成，37 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 5 輪。新增四層測試涵蓋 Conv／ReLU 排列、kernel／通道與參數、16 相初始等於 bilinear、四層及 decoded 梯度、encoded 不變性與 checkpoint 重建；舊版本測試同時通過。[測試 log](./runs/checks-v9-relu16-four-q5-all/tests.log) 已保存。所有執行以 run_logged／update_active_record 保存狀態、曲線、快照與指紋，不覆寫既有產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu16-four-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.746738 | 16.469517 | 20.110207 |
| 2 | 12.368335 | 16.469513 | 20.165340 |
| 3 | 12.232585 | 16.146869 | 20.257123 |
| 4 | 12.147615 | 16.270486 | 20.266848 |
| **5** | **12.095859** | **16.028068** | **20.323953** |

[best.pt](./runs/relu16-four-v9-q5/best.pt) 與 [last.pt](./runs/relu16-four-v9-q5/last.pt) 均為第 5 輪；設定見 [config.json](./runs/relu16-four-v9-q5/config.json)。

## 完整評估與解讀

同 Q=5 frame 720…899、5,400 條，以五輪中最低驗證 MAE 的第 5 輪評估。來源為 [本次 report](./runs/eval-fsr1-v9-relu16-four-q5/report.json)；三層與兩層引用 [三層 report](./runs/eval-fsr1-v9-relu16-three-q5/report.json) 及 [兩層 report](./runs/eval-fsr1-v9-relu16-q5/report.json)。資料、快取、窗口及五輪預算一致，FSR 與 bilinear 數值相同；未重跑或覆寫舊模型。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **relu16 四層、3×3** | **5,812** | **16.028068** | **20.323953** | **14.642700** | **21.145604** |
| relu16 三層、3×3 | 3,492 | 16.244480 | 20.183361 | 14.801605 | 21.008268 |
| relu16 兩層、3×3 | 1,172 | 16.321658 | 20.152144 | 14.933389 | 20.956188 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

四層相較三層的 RAW MAE 降低 0.216412（約 **1.33%**）、PSNR 增加 0.140592 dB；RGB MAE 降低 0.158905、PSNR 增加 0.137335 dB。相較兩層的 RAW MAE 改善約 **1.80%**。參數比三層多 2,320（約 66.44%），約為兩層 4.96 倍；本次有平均數值改善，但不能據此推論持續加深必然有效。

依 frame/block 對齊 [本次逐條結果](./runs/eval-fsr1-v9-relu16-four-q5/blocks.jsonl) 與三層逐條結果，RAW MAE 嚴格勝過三層為 **3,554／5,400（65.81%）**。對主要 FSR＋RCAS 的 RAW MAE 勝率為 **4,959／5,400（91.83%）**，高於三層的 84.85%；無 RCAS FSR 為 4,900／5,400（90.74%），RAW bilinear 為 5,213／5,400（96.54%）。仍有部分區塊退化，按場景與失敗條分析尚待完成。

四層是目前這組深度比較中平均指標較佳的候選。保留兩層 CLI 預設，四層以 `--v9-layers 4` 明確選擇；效能尚未量測，不宣稱部署更合適。單 seed、同影片且反覆選模，不能證明新影片泛化。

已查看 frame 750 六欄比較 PNG，四層標籤與排列正常，仍有水平條帶、模糊與細節流失。離線影片已輸出 180 幀／30 FPS，未完整觀看；不宣告視覺、時間穩定性、新影片或即時串流驗收通過。數值初篩改善與完整驗收分開。

## 產物與自動紀錄

- [最佳權重](./runs/relu16-four-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu16-four-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu16-four-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu16-four-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu16-four-q5-all--run-cc2779f3f3ad.json)、[訓練紀錄](./experiments/records/training--relu16-four-v9-q5--run-3e7eb2c1a7f5.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu16-four-q5--run-e1548cdd3632.json)
- [自動實驗索引](./實驗總表.md)，保存曲線、快照、資料／權重指紋與完整狀態。

## 重現

在儲存庫根目錄執行；重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu16-four-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 4 --v9-variant relu16 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu16-four-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu16-four-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu16-four-q5
```
