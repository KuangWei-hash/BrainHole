# v9 16 通道＋ReLU，Q=5

## 目的與架構

依使用者提議測試 `Conv3×3 4→16 → ReLU → Conv3×3 16→4`，設定 `v9_layers=2, v9_variant=relu16`，CLI 預設 relu16。共 1,172 參數／FP32 純參數 4,688 bytes，與 plain16 相同；兩層之間增加 ReLU。舊 variants 與權重保留相容。

只輸入本幀本條實際 decoded RAW＋phase，沿用相位對齊 bilinear、四平面殘差修正與輸出 clamp [0,1]，輸出 640×16 RAW8 RGGB。沒有 encoded 學習分支、前幀、鄰條。末層零初始化，初始輸出為 bilinear。

## 資料、預算與比較契約

沿用 Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`（SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`）、快取 `runs/decoded-q5-v4.npz`。train frame 0…719、21,600 條；validation frame 720…899、5,400 條。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪，不截短、不 warm-start，依最低驗證 MAE 選 best.pt。

主要對照為官方 FSR 1 兩次 EASU 2x＋RCAS sharpness=0，另列無 RCAS、plain16、plain32 與 bilinear。plain16／plain32 引用相同資料、窗口及五輪預算的既有評估，明列來源，不假稱重跑。RAW 投影與 RGB 參考指標分開報告，RGB 參考由目標 RAW 共同顯色，不是原始完整 RGB。

單 seed、同影片切分只供初步品質研究；不能沿用為 Q=4／新影片成績。完整視覺、獨立影片泛化與即時效能分開驗收。

## 狀態

實作完成，33 項測試全數通過，完整五輪訓練與 FSR 評估均已完成，最佳第 4 輪。新增測試驗證 Conv／ReLU／Conv 結構、16 通道與參數大小、全 16 相零初始化等於 bilinear、兩層權重與 decoded 梯度、encoded 不影響輸出、checkpoint 重建；舊版與 FSR 測試同時通過。測試輸出為 `runs/checks-v9-relu16-q5-all/tests.log`。使用既有 run_logged／update_active_record 保存所有狀態、曲線、程式快照、資料及權重指紋。

## 訓練曲線

來源：[metrics.jsonl](./runs/relu16-v9-q5/metrics.jsonl)，每輪完整訓練 21,600 條、驗證 5,400 條。

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR (dB) |
| --- | ---: | ---: | ---: |
| 1 | 12.909724 | 16.364806 | 20.120257 |
| 2 | 12.767002 | 16.626620 | 20.059297 |
| 3 | 12.731196 | 16.373090 | 20.123478 |
| **4** | **12.698305** | **16.321658** | **20.152144** |
| 5 | 12.671996 | 16.369080 | 20.153926 |

best.pt 依最低驗證 MAE 選第 4 輪；第 5 輪雖有稍高 PSNR，MAE 較差，不改選模規則。最佳權重與最終權重分別保存在 [best.pt](./runs/relu16-v9-q5/best.pt) 及 [last.pt](./runs/relu16-v9-q5/last.pt)，設定見 [config.json](./runs/relu16-v9-q5/config.json)。

## 完整評估與結論

同 Q=5 frame 720…899、5,400 條，使用最佳第 4 輪權重。[本次 report](./runs/eval-fsr1-v9-relu16-q5/report.json) 分開記錄 RAW 投影與 RGB 參考誤差；plain16 來源為 [先前 FSR report](./runs/eval-fsr1-v9-q5/report.json)，plain32 來源為 [32 通道 FSR report](./runs/eval-fsr1-v9-plain32-q5/report.json)，分別選五輪中的第 5／3 輪。兩者與本次資料庫指紋、header、快取與窗口一致，FSR 及 bilinear 對照數值亦一致；沒有重跑或覆寫舊模型。

| 方法 | 參數 | RAW 投影 MAE ↓ | RAW PSNR (dB) ↑ | RGB 參考 MAE ↓ | RGB PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **v9 relu16** | **1,172** | **16.321658** | **20.152144** | **14.933389** | **20.956188** |
| v9 plain16（既有） | 1,172 | 16.626771 | 20.079337 | 15.482858 | 20.788811 |
| v9 plain32（既有） | 2,340 | 16.684918 | 20.065681 | 15.562386 | 20.765904 |
| FSR 1 EASU 2x＋2x＋RCAS 0 | — | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR 1 EASU 2x＋2x | — | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear | 0 | 17.544584 | 19.708695 | 16.156488 | 20.550097 |
| 相同低解析度 RGB 輸入的 bilinear | 0 | 17.199471 | 19.921410 | 16.121581 | 20.598105 |

相同參數量下，relu16 比 plain16 的 RAW MAE 降低 0.305114（約 **1.84%**），PSNR 增加 0.072807 dB；RGB 參考指標也改善。相較主要 FSR＋RCAS 的 RAW MAE 降低約 **5.14%**。這次 16 通道加入 ReLU 有平均數值效益，優於 plain16 與 plain32；不從先前 4 通道 ReLU 退步推論所有寬度都會退步。

依 [逐條結果](./runs/eval-fsr1-v9-relu16-q5/blocks.jsonl) 的 RAW MAE 嚴格小於對照計數：勝 FSR＋RCAS 為 4,765／5,400（**88.24%**），勝無 RCAS FSR 為 4,669／5,400（86.46%），勝 RAW bilinear 為 5,167／5,400（95.69%）。平均指標雖改善，對 FSR＋RCAS 的逐條勝率低於 plain32 的 97.22%；不能宣稱所有區塊均改善。按獨立場景分組與失敗條分析尚待完成。

已查看 frame 750 六欄比較圖，方法標籤及版面正常，仍有水平條帶、模糊及細節流失；此觀察不代表全面視覺驗收。離線比較影片已輸出 180 幀／30 FPS，未完整觀看影片；時間穩定性、新影片泛化與即時串流效能均未驗收。以上為數值初篩改善，不能宣告完整品質驗收通過。

## 產物與自動紀錄

- [完整評估](./runs/eval-fsr1-v9-relu16-q5/report.json)、[逐條結果](./runs/eval-fsr1-v9-relu16-q5/blocks.jsonl)、[六欄比較圖](./runs/eval-fsr1-v9-relu16-q5/comparison.png)、[比較影片](./runs/eval-fsr1-v9-relu16-q5/comparison.mp4)
- [測試紀錄](./experiments/records/checks--checks-v9-relu16-q5-all--run-c2fbf100009d.json)、[訓練紀錄](./experiments/records/training--relu16-v9-q5--run-2c85f2778170.json)、[評估紀錄](./experiments/records/evaluation--eval-fsr1-v9-relu16-q5--run-c13854914427.json)
- [自動實驗索引](./實驗總表.md)，保存完整曲線、程式快照、資料及權重指紋與狀態。

## 重現

在儲存庫根目錄執行，重跑須選新輸出目錄：

```powershell
.\Model\.venv\Scripts\python.exe Model/run_checks.py --out Model/runs/checks-v9-relu16-q5-all
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 2 --v9-variant relu16 --history 1 --device cuda --epochs 5 --out Model/runs/relu16-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu16-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-relu16-q5
```
