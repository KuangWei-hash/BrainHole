# v9 四層 128 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，將四層 relu64 擴寬為 **4→128→128→128→4**，四層均為 3×3，前三層後各有 ReLU。共 **304,516 參數／FP32 純參數 1,218,064 bytes**；64／32 通道分別為 78,532／20,836 參數。只改中間寬度，深度、kernel 與 ReLU 排列保持相同，效能另行實測。

卷積仍在高解析度 CFA 四平面 8×320 網格上執行，感受野仍 9×9，同色位置輸出 RAW 半徑 ±8 像素。只讀本幀本條實際 decoded RAW＋phase，維持固定相位對齊 bilinear、卷積殘差加回、交錯與 clamp [0,1]，輸出 640×16 RAW8 RGGB。沒有 encoded 分支、前幀或鄰條。末層零初始化，初始等於 bilinear。

指定 `--v9-layers 4 --v9-variant relu128 --v9-kernel-width 3`；CLI 預設仍兩層 relu16，舊 variants 與權重保持相容。FSR 評估名稱為 `v9-relu128-layers4`。

## 資料與比較契約

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`），快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪，不截短、不 warm-start，按最低驗證 MAE 選 best.pt。

主要基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、四層 64／32／24／16 通道及 bilinear。舊模型引用相同資料、快取、窗口與五輪預算的既有 report，64 通道最佳第 4 輪、32 通道第 3 輪、24／16 通道第 5 輪，不假稱重跑。RAW 投影與 RGB 參考分開，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片反覆選模僅供數值初篩，不能證明新影片泛化；完整視覺、新影片與即時串流效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作完成，42 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 4 輪。新增測試驗證四層 128 通道 Conv／ReLU 結構、kernel 與參數大小、16 相初始等於 bilinear、四層及 decoded 梯度、encoded 不變性及 checkpoint 重建；舊版測試同時通過。[測試 log](./runs/checks-v9-relu128-four-q5-all/tests.log) 已保存。run_logged／update_active_record 保存所有狀態、曲線、快照與指紋，不覆寫舊產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu128-four-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.567209 | 16.111446 | 20.274429 |
| 2 | 11.962682 | 15.560370 | 20.423506 |
| 3 | 11.733475 | 15.589981 | 20.458652 |
| **4** | **11.622787** | **15.234079** | **20.551496** |
| 5 | 11.568947 | 15.360552 | 20.498996 |

[best.pt](./runs/relu128-four-v9-q5/best.pt) 為第 4 輪，[last.pt](./runs/relu128-four-v9-q5/last.pt) 為第 5 輪，設定見 [config.json](./runs/relu128-four-v9-q5/config.json)。

## 完整評估與解讀

Q=5 frame 720…899、5,400 條，以五輪最低驗證 MAE 的第 4 輪評估。來源為 [本次 report](./runs/eval-fsr1-v9-relu128-four-q5/report.json)；其他寬度引用 [64 通道 report](./runs/eval-fsr1-v9-relu64-four-q5/report.json)、[32 通道 report](./runs/eval-fsr1-v9-relu32-four-q5/report.json)、[24 通道 report](./runs/eval-fsr1-v9-relu24-four-q5/report.json)、[16 通道 report](./runs/eval-fsr1-v9-relu16-four-q5/report.json)。資料、快取、窗口及五輪預算一致，FSR／bilinear 數值相同；舊模型未重跑或覆寫。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **四層 relu128** | **304,516** | **15.234079** | **20.551496** | **13.830817** | **21.415726** |
| 四層 relu64 | 78,532 | 15.548606 | 20.430942 | 14.135203 | 21.275638 |
| 四層 relu32 | 20,836 | 15.731815 | 20.350482 | 14.302455 | 21.195823 |
| 四層 relu24 | 12,172 | 15.875731 | 20.355058 | 14.553716 | 21.175977 |
| 四層 relu16 | 5,812 | 16.028068 | 20.323953 | 14.642700 | 21.145604 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.15648785555804 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

128 通道相較 64 通道的 RAW MAE 降低 0.314527（約 **2.02%**），PSNR 增加 0.120554 dB；RGB MAE 降低 0.304386（約 2.15%），PSNR 增加 0.140087 dB。這次四項平均指標均改善，是目前這組寬度比較較佳的數值候選；不代表獨立影片或部署已通過。

參數比 64 通道增加 225,984（約 **287.76%**，總數 **3.88 倍**），純 FP32 參數約 1.22 MB。每輪離線訓練約 44 秒，數值見原始曲線；此耗時不是 batch=1 即時延遲，不能用來宣告符合串流需求。品質收益與成本分開判斷。

依 frame/block 對齊 [本次逐條結果](./runs/eval-fsr1-v9-relu128-four-q5/blocks.jsonl) 與 64 通道逐條結果，128 通道 RAW MAE 嚴格勝過 64 通道為 **4,009／5,400（74.24%）**。對主要 FSR＋RCAS 為 **5,362／5,400（99.30%）**，64 通道為 94.61%；無 RCAS FSR 為 5,351／5,400（99.09%），RAW bilinear 為 5,381／5,400（99.65%）。仍有退化條，按場景與失敗原因分析尚待完成。

單 seed、同影片反覆選模，不能證明穩健泛化，也不能保證繼續擴寬有益。CLI 預設保持兩層 relu16，四層 relu128 明確指定；所有舊權重及負面紀錄均保留。

已查看 frame 750 六欄比較 PNG，128 通道標籤與排列正常，仍有條帶、模糊及細節流失。離線比較影片已輸出 180 幀／30 FPS，未完整觀看；不宣告完整視覺、時間穩定性、新影片或即時串流驗收通過。

## 產物與自動紀錄

- [最佳權重](./runs/relu128-four-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu128-four-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu128-four-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu128-four-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu128-four-q5-all--run-49a6b2429efa.json)、[訓練紀錄](./experiments/records/training--relu128-four-v9-q5--run-94afa632311c.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu128-four-q5--run-996e94ef47fe.json)
- [自動實驗索引](./實驗總表.md)，保存完整曲線、狀態、快照與資料／權重指紋。

## 重現

在儲存庫根目錄執行，重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu128-four-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 4 --v9-variant relu128 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu128-four-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu128-four-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu128-four-q5
```
