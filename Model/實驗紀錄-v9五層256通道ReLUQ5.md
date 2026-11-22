# v9 五層 256 通道＋ReLU，Q=5

## 目的與設定

依使用者要求，在四層 256 通道上再增加一層：**4→256→256→256→256→4**，五層均為 3×3，前四層後各有 ReLU。共 **1,788,932 參數／FP32 純參數 7,155,728 bytes**；四層 256／128 通道分別為 1,198,852／304,516 參數。同時增加深度、ReLU 次數、容量與感受野，不能將效果完全歸因於其中一項。

既有模型已支援此組合，不改前向流程，新增具體結構／梯度與 checkpoint 測試。卷積仍在高解析度 CFA 四平面 8×320 網格上執行，五層感受野為 11×11，同色位置輸出 RAW 半徑 ±10 像素；不取得條外內容。只讀本幀本條實際 decoded RAW＋phase，維持固定相位對齊 bilinear、卷積殘差加回、交錯與 clamp [0,1]，輸出 640×16 RAW8 RGGB。沒有 encoded 分支、歷史幀或鄰條。末層零初始化，初始等於 bilinear。

明確指定 `--v9-layers 5 --v9-variant relu256 --v9-kernel-width 3`，FSR 評估名稱為 `v9-relu256-layers5`；CLI 預設及舊 variants／權重保持相容。

## 資料與比較契約

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`），快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪，不截短、不 warm-start；不從已訓練的四層權重接續，依最低驗證 MAE 選 best.pt。

主要基準為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、四層 256／128／64 通道及 bilinear。舊模型引用相同資料、快取、窗口與五輪預算的既有 report，四層 256 最佳第 3 輪、128／64 最佳第 4 輪，不假稱重跑。RAW 投影與 RGB 參考分開，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片反覆選模只供數值初篩，不能證明新影片泛化；完整視覺、新影片與即時串流效能另行驗收，不沿用為 Q=4 成績。

## 狀態

實作已支援，44 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 3 輪。新增測試驗證五層 256 通道 Conv／ReLU 結構、kernel／通道與參數大小、16 相初始等於 bilinear、五層及 decoded 梯度、encoded 不變性與 checkpoint 重建；舊版測試同時通過。[測試 log](./runs/checks-v9-relu256-five-q5-all/tests.log) 已保存。所有執行經 run_logged／update_active_record 保存狀態、曲線、快照與指紋，不覆寫舊產物。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu256-five-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.810707 | 16.468959 | 20.094219 |
| 2 | 12.109965 | 15.867400 | 20.382386 |
| **3** | **11.806175** | **15.567865** | **20.395050** |
| 4 | 11.635284 | 15.846347 | 20.321274 |
| 5 | 11.504167 | 15.627807 | 20.396841 |

[best.pt](./runs/relu256-five-v9-q5/best.pt) 按最低驗證 MAE 選第 3 輪；第 5 輪 PSNR 略高但 MAE 較差，不改選模規則。[last.pt](./runs/relu256-five-v9-q5/last.pt) 為第 5 輪，設定見 [config.json](./runs/relu256-five-v9-q5/config.json)。每輪訓練約 203…206 秒、驗證約 23 秒，是離線過程耗時，不是即時串流延遲。

## 完整評估與解讀

Q=5 frame 720…899、5,400 條，以五輪最低驗證 MAE 的第 3 輪評估。來源為 [本次 report](./runs/eval-fsr1-v9-relu256-five-q5/report.json)；其他模型引用 [四層 256 report](./runs/eval-fsr1-v9-relu256-four-q5/report.json)、[四層 128 report](./runs/eval-fsr1-v9-relu128-four-q5/report.json)、[四層 64 report](./runs/eval-fsr1-v9-relu64-four-q5/report.json)。資料、快取、窗口及五輪預算一致，FSR／bilinear 數值相同；舊模型未重跑或覆寫。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 五層 relu256 | 1,788,932 | 15.567865 | 20.395050 | 14.147745 | 21.249740 |
| 四層 relu256 | 1,198,852 | 15.513671 | 20.484319 | 14.185627 | 21.320860 |
| 四層 relu128 | 304,516 | 15.234079 | 20.551496 | 13.830817 | 21.415726 |
| 四層 relu64 | 78,532 | 15.548606 | 20.430942 | 14.135203 | 21.275638 |
| FSR EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

相較四層 256，五層 RAW MAE 增加 0.054194（約 **0.35%**），RAW PSNR 降低 0.089270 dB；RGB MAE 略降 0.037881（約 0.27%），但 RGB PSNR 降低 0.071120 dB。不能稱所有指標都退步，也不能據 RGB MAE 的小改善宣稱整體成功。依主要 RAW 指標，本次在 256 通道再加一層沒有幫助。

相較四層 128，五層 256 的 RAW MAE 增加 0.333786（約 **2.19%**），RAW PSNR 降低 0.156447 dB，RGB 兩項也較差。參數比四層 256 增加 590,080（約 **49.22%**），比四層 128 約為 **5.87 倍**；每輪離線訓練由約 143 秒增至約 205 秒，但不作即時推理效能結論。本次不建議以五層 256 替換四層 128，後者仍為目前這組實驗較佳的平均數值候選。單 seed、五輪與同影片反覆選模不能證明深度／寬度已達上限，也不足以斷定退步原因為過擬合。

依 frame/block 對齊 [本次逐條結果](./runs/eval-fsr1-v9-relu256-five-q5/blocks.jsonl) 與舊模型逐條結果，五層 256 的 RAW MAE 嚴格勝過四層 256 為 **2,315／5,400（42.87%）**，勝過四層 128 為 **531／5,400（9.83%）**。對主要 FSR＋RCAS 為 **5,325／5,400（98.61%）**，四層 256 為 99.02%；無 RCAS FSR 為 5,292／5,400（98.00%），RAW bilinear 為 5,368／5,400（99.41%）。仍優於 FSR 的平均分數，不等於優於原模型；按場景及退化原因分析尚待完成。

已查看 frame 750 六欄比較 PNG，五層 256 標籤與排列正常，仍有條帶、模糊與細節流失。離線比較影片已輸出 180 幀／30 FPS，未完整觀看；不宣告完整視覺、時間穩定性、新影片泛化或即時串流驗收通過。CLI 預設與舊權重保持不變，所有負面結果與產物保留。

## 產物與自動紀錄

- [最佳權重](./runs/relu256-five-v9-q5/best.pt)、[完整 report](./runs/eval-fsr1-v9-relu256-five-q5/report.json)、[比較圖](./runs/eval-fsr1-v9-relu256-five-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu256-five-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu256-five-q5-all--run-2581c09a9cdd.json)、[訓練紀錄](./experiments/records/training--relu256-five-v9-q5--run-7190cd333b3b.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu256-five-q5--run-37a3eb4431c6.json)
- [自動實驗索引](./實驗總表.md)，保存完整狀態、曲線、快照與資料／權重指紋。

## 重現

在儲存庫根目錄執行；重跑須選新輸出目錄。

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu256-five-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 5 --v9-variant relu256 --v9-kernel-width 3 --history 1 --device cuda --epochs 5 --out Model/runs/relu256-five-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu256-five-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu256-five-q5
```
