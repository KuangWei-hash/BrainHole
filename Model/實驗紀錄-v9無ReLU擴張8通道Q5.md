# v9 無 ReLU：4→8→4，Q=5

依使用者要求移除 ReLU，測試兩層 3×3 卷積的通道擴張再收縮。`v9_layers=2, v9_variant=plain8` 為目前訓練預設；中間通道由 4 擴到 8，再收回 4。588 參數，FP32 純參數 2,352 bytes。既有 plain4、relu4、relu8 與單層版本保留相容及原產物。

輸入只用本幀本條實際 decoded RAW＋phase，固定 RGGB 相位對齊 bilinear 放大、分平面，Conv3×3 4→8 → Conv3×3 8→4，加回放大四平面後交錯、clamp [0,1]。末層零初始化，初始等於 bilinear。沒有 encoded 分支、歷史幀或上一條。

資料為 Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`，SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`；快取 `runs/decoded-q5-v4.npz`，Go 實際解碼與 Rust 全 27,000 筆 byte 一致。輸出 640×16 RAW8 RGGB。

訓練 frame 0…719（21,600 窗口），驗證 frame 720…899（5,400 窗口）。Adam lr=0.001、batch=8、seed=42、FP32 RAW L1，從頭完整五輪、不截步、不 warm-start；依最低驗證 MAE 選最佳輪。與先前 plain4、relu4、relu8 五輪相同預算。

主對照為無 ReLU 4→4→4，另比較同為 588 參數的 4→8→4＋ReLU 以區分寬度與 activation。所有比較使用相同 5,400 窗口，同影片保留集與單 seed 不代表獨立影片泛化或統計穩定性。保留負面結果；數值、視覺、泛化、串流效能分開判定。

27 項模型及紀錄測試通過，新增測試檢查精確 4→8→4、無 ReLU、588 參數／2,352 bytes、全 16 相零初始化等於 bilinear、兩步訓練梯度、encoded 不影響輸出與 checkpoint 重建。五輪完整訓練完成，run_logged 與 update_active_record 保存來源快照、指紋與逐輪數值。

## 訓練曲線

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR dB |
| --- | ---: | ---: | ---: |
| 1 | 13.420496 | 16.911454 | 19.946281 |
| **2** | **13.336330** | **16.743010** | **19.998957** |
| 3 | 13.315183 | 16.817685 | 19.996592 |
| 4 | 13.300051 | 16.758526 | 20.019988 |
| 5 | 13.285275 | 16.753768 | 20.019746 |

依最低驗證 MAE 選最佳第 2 輪，best.pt 為第 2 輪，last.pt 為第 5 輪。第 4 輪 PSNR 較高，但不變更既定的 MAE 選模規則。

## 完整評估與解讀

以下為同一次評估、相同 frame 720…899、5,400 窗口，各模型均取各自五輪預算內最低驗證 MAE 的最佳權重：

| 方法 | 參數 | MAE ↓ | PSNR dB ↑ |
| --- | ---: | ---: | ---: |
| nearest | 0 | 19.276165 | 18.490834 |
| bilinear | 0 | 17.544584 | 19.708695 |
| 無 ReLU 4→4→4 | 296 | 16.806699 | 19.995103 |
| **無 ReLU 4→8→4，本次** | **588** | **16.743010** | **19.998957** |
| 原 ReLU 4→8→4 | 588 | 16.646705 | 20.014616 |

本次擴張再收縮比無 ReLU 4 通道版 MAE 降低 0.063689，約 0.38%，PSNR 增加 0.003854 dB，屬小幅改善。相對 bilinear MAE 降低約 4.57%，數值初篩通過，逐條同時勝過 nearest／bilinear 比例 99.5185%。與同寬 588 參數的原 ReLU 版相比，無 ReLU MAE 仍較差 0.096305；因此先前 4 通道加入 ReLU 退步，不能推論所有寬度下 ReLU 都無效。本次依要求預設採 plain8，保留原 ReLU 權重為對照。

無中間 activation 的卷積部分仍是仿射映射，擴寬改變參數分解及可表達能力，不能將這次結果解讀成非線性能力改善。只有單 seed、五輪且同影片保留集，小幅差距尚未證明穩定或可泛化。

已查看比較 PNG，排列與尺寸正常；條帶、模糊與細節流失仍明顯，未完成全面視覺驗收。ffprobe 確認影片 1280×1524、180 幀、30 FPS、6 秒、11,298,387 bytes。時間誤差變化診斷 bilinear 15.175699、plain4 13.959415、plain8 13.990068、relu8 13.919200，未光流對齊，不能單獨判定閃爍。新影片泛化、端到端串流速度均未驗證；離線輸出影片不代表即時速度。

## 產物與重現指令

- [最佳權重](./runs/plain8-v9-q5/best.pt)、[最終權重](./runs/plain8-v9-q5/last.pt)、[設定](./runs/plain8-v9-q5/config.json)、[曲線](./runs/plain8-v9-q5/metrics.jsonl)
- [完整評估](./runs/eval-v9-plain8-q5/report.json)、[逐條結果](./runs/eval-v9-plain8-q5/blocks.jsonl)、[對照圖](./runs/eval-v9-plain8-q5/comparison.png)、[影片](./runs/eval-v9-plain8-q5/comparison.mp4)
- [自動實驗索引](./實驗總表.md)

以下為本次實際指令；重跑時須使用新輸出目錄，避免覆寫：

```powershell
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 2 --v9-variant plain8 --history 1 --device cuda --epochs 5 --out Model/runs/plain8-v9-q5
.\Model\.venv\Scripts\python.exe Model/evaluate.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/plain4-v9-q5/best.pt Model/runs/plain8-v9-q5/best.pt Model/runs/simple-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-v9-plain8-q5
```
