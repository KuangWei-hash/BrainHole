# v9 無 ReLU：4→16→4，Q=5

## 目的與架構

依要求把無 ReLU 的擴張層由 8 增加至 16 通道：`Conv3×3 4→16 → Conv3×3 16→4`，共 **1,172 參數／FP32 純參數 4,688 bytes**。目前 CLI 預設 `--v9-layers 2 --v9-variant plain16`，既有所有 variants 與權重保留相容。

仍只讀本幀本條的實際 decoded RAW＋phase，固定 RGGB 相位對齊 bilinear 放大、分平面，兩層卷積修正加回，交錯並 clamp [0,1]，輸出 640×16 RAW8 RGGB。末層零初始化，初始等於 bilinear。沒有中間 ReLU、encoded 分支、前幀或上一條。

## 資料與預算

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`，SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`，快取 `runs/decoded-q5-v4.npz`。Go 實際解碼與 Rust 全部 27,000 筆 byte 一致。

訓練 frame 0…719（21,600 窗口）、驗證 frame 720…899（5,400 窗口），與 4／8 通道版同集合及預算。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪，不截短步數、不 warm-start，依最低驗證 MAE 選最佳權重。主對照為無 ReLU 4→8→4，亦列無 ReLU 4→4→4 與 nearest／bilinear。

單 seed、同影片保留集，只代表這次五輪設定的數值初篩。擴寬不保證改善；視覺、泛化與串流速度分開判定。run_logged／update_active_record 保存全部曲線、來源快照、指紋與狀態，舊結果不覆寫。

28 項測試通過，涵蓋精確通道／層數、無 ReLU、1,172 參數／4,688 bytes、全 16 相零初始化等於基準、兩層及 decoded 梯度、encoded 不影響輸出、checkpoint 重建與舊版相容。正式五輪已完成，最佳第 5 輪，best.pt 與 last.pt 皆為此輪；輸出為 `runs/plain16-v9-q5/` 及 `runs/eval-v9-plain16-q5/`。

## 訓練曲線

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR dB |
| --- | ---: | ---: | ---: |
| 1 | 13.410174 | 16.763671 | 19.996626 |
| 2 | 13.318588 | 16.779464 | 20.027479 |
| 3 | 13.291662 | 16.697412 | 20.053373 |
| 4 | 13.274340 | 16.712149 | 20.047156 |
| **5** | **13.257812** | **16.626771** | **20.079337** |

## 完整評估

同一次評估、frame 720…899、5,400 窗口，全部取各自五輪內最低驗證 MAE 的最佳權重：

| 方法 | 參數 | MAE ↓ | PSNR dB ↑ |
| --- | ---: | ---: | ---: |
| nearest | 0 | 19.276165 | 18.490834 |
| bilinear | 0 | 17.544584 | 19.708695 |
| 無 ReLU 4→4→4 | 296 | 16.806699 | 19.995103 |
| 無 ReLU 4→8→4 | 588 | 16.743010 | 19.998957 |
| **無 ReLU 4→16→4** | **1,172** | **16.626771** | **20.079337** |

16 通道相對 8 通道 MAE 降低 0.116239（約 **0.69%**），PSNR 增加 0.080380 dB；相對 4 通道 MAE 降低約 1.07%，相對 bilinear 約 5.23%。數值初篩通過，5,400 條皆同時勝過 nearest 與 bilinear。這次 4→8→16 的結果支持增加通道有小幅效益，但仍是單 seed、同影片、五輪預算，不能保證繼續加寬必然提升。

已查看比較 PNG，排列正常，仍有條帶、模糊及細節流失；未完成全面視覺驗收。ffprobe 確認離線比較影片 1280×1524、180 幀、30 FPS、6 秒、11,180,834 bytes。時間誤差變化診斷 bilinear 15.175699、4 通道 13.959415、8 通道 13.990068、16 通道 13.686025，未光流對齊，不能單獨當作閃爍驗收。獨立影片泛化、即時串流速度尚未驗證。

## 產物與重現指令

- [最佳權重](./runs/plain16-v9-q5/best.pt)、[最終權重](./runs/plain16-v9-q5/last.pt)、[設定](./runs/plain16-v9-q5/config.json)、[曲線](./runs/plain16-v9-q5/metrics.jsonl)
- [完整評估](./runs/eval-v9-plain16-q5/report.json)、[逐條結果](./runs/eval-v9-plain16-q5/blocks.jsonl)、[比較圖](./runs/eval-v9-plain16-q5/comparison.png)、[影片](./runs/eval-v9-plain16-q5/comparison.mp4)
- [自動實驗索引](./實驗總表.md)

重跑時須選新輸出目錄：

```powershell
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 2 --v9-variant plain16 --history 1 --device cuda --epochs 5 --out Model/runs/plain16-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/plain4-v9-q5/best.pt Model/runs/plain8-v9-q5/best.pt Model/runs/plain16-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-v9-plain16-q5
```
