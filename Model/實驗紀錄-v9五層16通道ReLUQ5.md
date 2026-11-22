# v9 五層 16 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，從四層再加一層：**4→16→16→16→16→4**，五層均為 3×3，前四層後各有 ReLU。共 **8,132 參數／FP32 純參數 32,528 bytes**；四層版為 5,812 參數。本次同時增加深度、ReLU 次數、參數與感受野，不能將效果完全歸因於其中一項。

卷積仍在高解析度 CFA 四平面 8×320 上執行，平面感受野 11×11，同色位置輸出 RAW 半徑 ±10 像素；不代表取得條外內容。只用本幀本條實際 decoded RAW＋phase，維持固定相位對齊 bilinear、四平面殘差加回、交錯與 clamp [0,1]，輸出 640×16 RAW8 RGGB。沒有 encoded 分支、歷史幀或鄰條；末層零初始化，初始等於 bilinear。

指定 `--v9-layers 5 --v9-variant relu16 --v9-kernel-width 3`；CLI 預設仍兩層，舊 checkpoint 保持相容。FSR 評估名稱為 `v9-relu16-layers5`。

## 資料與比較契約

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`），快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪，不截短、不 warm-start，按最低驗證 MAE 選 best.pt。

主要基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、四／三／兩層 relu16 及 bilinear。舊版本引用相同資料、快取、窗口與五輪預算的既有 report，四層最佳第 5 輪、三層第 3 輪、兩層第 4 輪，不假稱重跑。RAW 投影與 RGB 參考分開，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片且反覆用於選模，僅供初篩，不能證明新影片泛化；視覺、新影片與即時效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作完成，38 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 4 輪。新增測試涵蓋五層 Conv／ReLU 結構、通道／kernel 與參數、16 相零初始化、所有層與 decoded 梯度、encoded 不變性及 checkpoint 重建；舊版測試同時通過。[測試 log](./runs/checks-v9-relu16-five-q5-all/tests.log) 已保存。所有執行以 run_logged／update_active_record 保存狀態、曲線、快照與指紋，不覆寫舊產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu16-five-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.914091 | 16.789657 | 19.959933 |
| 2 | 12.499367 | 16.338141 | 20.100214 |
| 3 | 12.258877 | 16.454975 | 20.128430 |
| **4** | **12.147226** | **16.099018** | **20.217805** |
| 5 | 12.053839 | 16.348463 | 20.212319 |

[best.pt](./runs/relu16-five-v9-q5/best.pt) 為第 4 輪，[last.pt](./runs/relu16-five-v9-q5/last.pt) 為第 5 輪，設定見 [config.json](./runs/relu16-five-v9-q5/config.json)。

## 完整評估與解讀

Q=5 frame 720…899、5,400 條，取五輪最低驗證 MAE 的第 4 輪。來源為 [本次 report](./runs/eval-fsr1-v9-relu16-five-q5/report.json)；四層引用 [四層 report](./runs/eval-fsr1-v9-relu16-four-q5/report.json)，三層引用 [三層 report](./runs/eval-fsr1-v9-relu16-three-q5/report.json)，兩層引用 [兩層 report](./runs/eval-fsr1-v9-relu16-q5/report.json)。各版本資料、快取、窗口與五輪預算一致，FSR／bilinear 數值亦一致；未重跑或改寫舊結果。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| relu16 五層、3×3 | 8,132 | 16.099018 | 20.217805 | 14.637216 | 21.076572 |
| relu16 四層、3×3 | 5,812 | 16.028068 | 20.323953 | 14.642700 | 21.145604 |
| relu16 三層、3×3 | 3,492 | 16.244480 | 20.183361 | 14.801605 | 21.008268 |
| relu16 兩層、3×3 | 1,172 | 16.321658 | 20.152144 | 14.933389 | 20.956188 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

五層相較四層的 RAW MAE 增加 0.070950（約 **0.44%**），PSNR 降低 0.106147 dB；參數增加 2,320（約 **39.92%**）。RGB MAE 略降 0.005484（約 0.04%），但 RGB PSNR 亦降低 0.069032 dB，不能宣稱所有指標都退步或全面改善。依主要 RAW 指標，這次加深至五層沒有改善，四層仍是這組深度比較較佳的候選；不將負面結果改寫為成功，也不能單憑本次推斷已達深度上限或確定過擬合。

依 frame/block 對齊 [本次逐條結果](./runs/eval-fsr1-v9-relu16-five-q5/blocks.jsonl) 與四層逐條結果，五層 RAW MAE 嚴格勝過四層為 **2,017／5,400（37.35%）**。對主要 FSR＋RCAS 為 **4,722／5,400（87.44%）**，低於四層的 91.83%；無 RCAS FSR 為 4,610／5,400（85.37%），RAW bilinear 為 5,071／5,400（93.91%）。平均指標仍優於兩種 FSR，但部分區塊退步；按場景與退化原因分析尚待完成。

已查看 frame 750 比較 PNG，五層標籤與排列正常，仍有水平條帶、模糊與細節流失。離線比較影片輸出 180 幀／30 FPS，未完整觀看；不宣告視覺、時間穩定性、新影片泛化或即時串流通過。單 seed、同影片反覆選模的限制保留。CLI 預設仍兩層，五層明確指定 `--v9-layers 5`，不根據本次結果替換四層候選。

## 產物與自動紀錄

- [最佳權重](./runs/relu16-five-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu16-five-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu16-five-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu16-five-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu16-five-q5-all--run-634c5c6f819a.json)、[訓練紀錄](./experiments/records/training--relu16-five-v9-q5--run-85024beef6c2.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu16-five-q5--run-ace98df4f35c.json)
- [自動實驗索引](./實驗總表.md)，保存完整曲線、狀態、快照及資料／權重指紋。

## 重現

在儲存庫根目錄執行；重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu16-five-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 5 --v9-variant relu16 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu16-five-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu16-five-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu16-five-q5
```
