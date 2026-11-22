# v9 四層 32 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，將四層 relu24 擴寬為 **4→32→32→32→4**，四層均為 3×3，前三層後各有 ReLU。共 **20,836 參數／FP32 純參數 83,344 bytes**；四層 24／16 通道分別為 12,172／5,812 參數。只改中間寬度，深度、kernel 與 ReLU 排列相同；不是先前兩層無 ReLU plain32。

高解析度 CFA 四平面網格為 8×320，四層感受野仍 9×9，同色位置輸出 RAW 半徑 ±8 像素；擴寬不增加視野。只用本幀本條實際 decoded RAW＋phase，維持固定相位對齊 bilinear、卷積殘差加回、交錯與 clamp [0,1]，輸出 640×16 RAW8 RGGB；無 encoded 分支、歷史幀或鄰條。末層零初始化，初始等於 bilinear。

設定 `--v9-layers 4 --v9-variant relu32 --v9-kernel-width 3`，CLI 預設仍保留兩層 relu16；舊 variants／權重保持相容。FSR 評估名稱為 `v9-relu32-layers4`。

## 資料與比較契約

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`），快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪、不截短、不 warm-start，按最低驗證 MAE 選 best.pt。

主要基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、四層 24／16 通道及 bilinear。舊模型引用同資料、快取、窗口及五輪預算的既有 report（24／16 通道均最佳第 5 輪），不假稱重跑。RAW 投影與 RGB 參考分開報告，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片反覆選模只供數值初篩，不能證明新影片泛化；完整視覺、新影片與即時效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作完成，40 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 3 輪。新增測試涵蓋四層 32 通道 Conv／ReLU 結構、kernel 與參數大小、全 16 相初始等於 bilinear、四層及 decoded 梯度、encoded 不變性與 checkpoint 重建；舊版測試同時通過。[測試 log](./runs/checks-v9-relu32-four-q5-all/tests.log) 已保存。所有執行經 run_logged／update_active_record 保存狀態、曲線、快照與指紋，不覆寫舊產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu32-four-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.621452 | 15.889006 | 20.293060 |
| 2 | 12.139281 | 16.079063 | 20.268603 |
| **3** | **11.960072** | **15.731815** | **20.350481** |
| 4 | 11.854917 | 15.866851 | 20.322786 |
| 5 | 11.796711 | 15.865766 | 20.359371 |

[best.pt](./runs/relu32-four-v9-q5/best.pt) 依最低驗證 MAE 選第 3 輪；第 5 輪 PSNR 較高但 MAE 較差，不改選模規則。[last.pt](./runs/relu32-four-v9-q5/last.pt) 為第 5 輪，設定見 [config.json](./runs/relu32-four-v9-q5/config.json)。

## 完整評估與解讀

Q=5 frame 720…899、5,400 條，以五輪最低驗證 MAE 的第 3 輪評估。來源為 [本次 report](./runs/eval-fsr1-v9-relu32-four-q5/report.json)；24／16 通道分別引用 [24 通道 report](./runs/eval-fsr1-v9-relu24-four-q5/report.json) 與 [16 通道 report](./runs/eval-fsr1-v9-relu16-four-q5/report.json)，均為五輪最佳第 5 輪。各版本資料、快取、窗口與五輪預算一致，FSR／bilinear 數值相同；未重跑或覆寫舊模型。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 四層 relu32 | 20,836 | 15.731815 | 20.350482 | 14.302455 | 21.195823 |
| 四層 relu24 | 12,172 | 15.875731 | 20.355058 | 14.553716 | 21.175977 |
| 四層 relu16 | 5,812 | 16.028068 | 20.323953 | 14.642700 | 21.145604 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

32 通道相較 24 通道的 RAW MAE 降低 0.143915（約 **0.91%**），但 RAW PSNR **降低 0.004577 dB**；RGB MAE 降低 0.251261（約 1.73%），RGB PSNR 增加 0.019846 dB。主要選模 MAE 改善，但未同時改善所有指標，不宣稱全面勝過 24 通道。相較 16 通道 RAW MAE 改善約 1.85%。

參數比 24 通道增加 8,664（約 **71.18%**、總數為 1.71 倍），比 16 通道約為 3.58 倍；品質改善幅度小，效能與部署代價須實測，不能從參數量直接當作實際延遲。

依 frame/block 對齊 [本次逐條結果](./runs/eval-fsr1-v9-relu32-four-q5/blocks.jsonl) 與 24 通道逐條結果，32 通道 RAW MAE 嚴格勝過 24 通道為 **3,645／5,400（67.50%）**。對主要 FSR＋RCAS 為 **5,116／5,400（94.74%）**，24 通道為 94.15%；無 RCAS FSR 為 5,068／5,400（93.85%），RAW bilinear 為 5,306／5,400（98.26%）。部分區塊仍退化，按獨立場景與失敗條分析尚待完成。

本次 32 通道是這組四層寬度比較中較佳的平均 MAE 候選；24 通道保有略高 RAW PSNR，不能直接淘汰。單 seed、同影片反覆選模，不代表新影片泛化；不能保證持續擴寬有益。CLI 預設維持兩層 relu16，四層 relu32 明確指定，不覆寫舊權重。

已查看 frame 750 比較 PNG，32 通道標籤與排列正常，仍有水平條帶、模糊與細節流失。離線比較影片已輸出 180 幀／30 FPS，未完整觀看；不宣告視覺、時間穩定性、新影片或即時串流驗收通過。

## 產物與自動紀錄

- [最佳權重](./runs/relu32-four-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu32-four-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu32-four-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu32-four-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu32-four-q5-all--run-5af376dcfa5c.json)、[訓練紀錄](./experiments/records/training--relu32-four-v9-q5--run-a9cf147311c7.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu32-four-q5--run-4198367e22d7.json)
- [自動實驗索引](./實驗總表.md)，保存完整狀態、曲線、快照及資料／權重指紋。

## 重現

在儲存庫根目錄執行，重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu32-four-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 4 --v9-variant relu32 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu32-four-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu32-four-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu32-four-q5
```
