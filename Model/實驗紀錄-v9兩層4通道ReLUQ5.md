# v9 兩層 4 通道加入 ReLU，Q=5

## 目的與改動

依使用者要求，維持兩層 3×3、4→4→4，在兩層卷積之間加入 ReLU：

```text
Q=5 本幀本條 decoded RAW＋phase
    → 固定 RGGB 相位對齊 bilinear 放大與分平面
    → Conv3×3 4→4 → ReLU → Conv3×3 4→4
    → 加回放大四平面 → RAW 交錯 → clamp [0,1]
```

ReLU 沒有可學習參數，因此仍為 **296 參數／FP32 純參數 1,184 bytes**。沒有擴到 8 通道，沒有 encoded 分支、歷史幀、上一條、embedding 或時間融合。末層零初始化，初始預測等於 bilinear。

新增 variant `relu4`，保存 `v9_layers=2, v9_variant=relu4, hidden_channels=4, activation=relu`。目前訓練預設為 relu4；無 ReLU 版本仍用 plain4，原 4→8→4＋ReLU 用 relu8。舊 checkpoint 的架構與權重保持相容，不跨 variant warm-start。

## 資料與預算

Q=5 schema v4、levels=1/0、Br=4，資料庫 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`，SHA-256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`。快取 decoded-q5-v4.npz 綁定同筆 Go 實際 decoded，已與 Rust 全部 27,000 筆逐 byte 核對。

- Big Buck Bunny 30 秒，單幀本條，輸出本條 640×16 RAW8 RGGB。
- 訓練 frame 0…719，共 21,600 窗口；驗證 frame 720…899，共 **5,400 窗口**。
- 與無 ReLU 的 plain4 採相同資料／切分／窗口／五輪預算：Adam lr=0.001、batch=8、seed=42、FP32、RAW L1，從頭訓練、不截短步數、不載入舊權重。
- 主對照為無 ReLU 的 296 參數版，原完整 MAE 16.806699／PSNR 19.995103 dB；本次重新載入其最佳權重於同一次評估比較。另記 Q=5 nearest／bilinear，不混入不同 Q 或 5,370 窗口的分數。
- 單一 seed、同影片保留集，不代表獨立影片或統計穩定性結論。

## 檢查與狀態

26 項模型／紀錄測試通過，包含精確 Conv→ReLU→Conv 順序、每層通道、296 參數／1,184 bytes、所有 16 相初始基準、實際兩步訓練的兩層及 decoded 梯度、encoded／length 不影響輸出、checkpoint 重建，以及所有舊模型相容。

正式五輪訓練與完整評估已完成，run_logged／update_active_record 保存曲線與來源／資料／權重指紋。新輸出 `runs/relu4-v9-q5/`，比較為 `runs/eval-v9-relu4-q5/`；舊版產物保留。

## 結果與解讀

| 輪數 | 訓練 MAE | 驗證 MAE | 驗證 PSNR dB |
| --- | ---: | ---: | ---: |
| 1 | 13.432946 | 17.376178 | 19.750389 |
| 2 | 13.325218 | 17.392072 | 19.758858 |
| 3 | 13.300188 | 17.450957 | 19.748544 |
| 4 | 13.267877 | 17.531815 | 19.734928 |
| **5** | **13.176719** | **17.290878** | **19.851339** |

最佳輪依最低驗證 MAE 選為第 5 輪，best.pt 與 last.pt 均為本輪。兩版使用各自五輪中的最佳權重，在本次同一評估程序的 frame 720…899、5,400 窗口比較：

| 方法 | MAE（越低越好） | PSNR dB（越高越好） |
| --- | ---: | ---: |
| nearest | 19.276165 | 18.490834 |
| bilinear | 17.544584 | 19.708695 |
| 兩層 4 通道、無 ReLU | **16.806699** | **19.995103** |
| 兩層 4 通道、加入 ReLU | 17.290878 | 19.851339 |

ReLU 相對無 ReLU 的 MAE 增加 0.484179（約 2.88%），PSNR 下降 0.143764 dB，本次沒有改善；相對 bilinear 仍降低 MAE 約 1.45%，數值初篩通過，逐條同時勝過兩基準的比例 86.4444%（無 ReLU 99.7778%）。不能由這次單 seed、五輪、單影片測試斷言 ReLU 普遍無效。訓練 MAE 降低但驗證較差，顯示這次設定下訓練集擬合改善沒有轉化為保留集品質；原因尚未獨立驗證。

已查看比較 PNG，輸出排列與尺寸正常，但仍可見條帶、模糊及細節流失，沒有通過完整視覺驗收。ffprobe 確認比較影片 1280×1016、180 幀、30 FPS、6 秒、8,555,591 bytes；這是離線影片，不能當作即時效能證據。時間誤差變化診斷 bilinear 15.175699、無 ReLU 13.959415、ReLU 14.590014；未作光流對齊，不能單獨判定閃爍。新影片泛化與串流效能未驗證。

## 產物

- [最佳權重](./runs/relu4-v9-q5/best.pt)、[逐輪曲線](./runs/relu4-v9-q5/metrics.jsonl)、[訓練設定](./runs/relu4-v9-q5/config.json)
- [完整評估](./runs/eval-v9-relu4-q5/report.json)、[逐條結果](./runs/eval-v9-relu4-q5/blocks.jsonl)
- [比較圖片](./runs/eval-v9-relu4-q5/comparison.png)、[比較影片](./runs/eval-v9-relu4-q5/comparison.mp4)
- [自動實驗索引](./實驗總表.md)：訓練、評估與 26 項測試皆已留存。

## 重現

```powershell
.\Model\.venv\Scripts\python.exe Model/train.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --architecture v9 --v9-layers 2 --v9-variant relu4 --history 1 --device cuda --epochs 5 --out Model/runs/relu4-v9-q5
```
