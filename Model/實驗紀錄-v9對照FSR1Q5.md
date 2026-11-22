# v9 對照 AMD FSR 1.0，Q=5

## 目的與比較契約

依使用者要求，以官方 FSR 1 EASU／RCAS 對照目前 v9 plain16。沿用 bilinear 作定位基準，沒有改訓練或替換歷史分數。

後續比較規則依使用者指示改為 **FSR 1 作主要基準**，bilinear 僅作輔助定位。此處補充比較規則與已保存 report 的解讀，不改寫原始數值或歷史判定。

官方 EASU 要求顯示編碼 RGB，標準範圍每邊 1…2 倍（面積 1…4 倍）。本資料每邊放大 4 倍，故預先固定 **兩次 EASU 2x，再 RCAS sharpness=0（最大銳化）**。亦列不加 RCAS 的兩次 EASU；不是單次官方 Performance 模式。不依驗證標籤調參。使用 AMD 官方 FidelityFX CLI v1.0.3 D3D11 shader、FP16 texture，各階段經 PNG8 儲存，因此不是無中間量化的遊戲整合。

FSR 不能直接處理 Bayer。輸入轉換先把實際 decoded 的四個 CFA 平面依採樣相位與 R/G/B 座標，獨立 bilinear 重建成 160×4 RGB，位置為高解析度 (4u+1.5,4v+1.5)，配合 FSR 中心對齊縮放，不再額外平移。兩綠平面平均。資料來自影片 RGB8 直接抽 RGGB，未線性化，故保留顯示編碼範圍 [0,1]，不額外加 gamma。整個流程逐條、無上下條或歷史幀。

同時報告 RGB 與 RAW 投影誤差：RGB 參考是固定 bilinear 去馬賽克的高解析度 RAW 標籤，並非原始完整 RGB；FSR RGB 依 RGGB 取樣後可與原 RAW 標籤比較，模型 RAW 則經共同去馬賽克後比較 RGB。RGB 分數不可冒充先前 RAW 分數。保留 RAW bilinear→RGB、共同低解析度 RGB→bilinear 與 v9，以區分輸入轉換與上採樣的影響。

批次 atlas 的每條輸入獨立複製 8 像素邊界，只取自身有效區域；不是把相鄰影像條當作額外上下文。需以單獨 tile 與 atlas 數值核對邊界隔離。

## 資料與範圍

Q=5 schema v4 `big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`，SHA256 `11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`，快取 decoded-q5-v4.npz。本次只評估 frame 720…899、5,400 條，v9 權重 `runs/plain16-v9-q5/best.pt`（五輪中最佳第 5 輪）。不重新訓練，不使用標籤生成 FSR 輸入。

## 來源與執行狀態

- 官方算法：[AMD FSR 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR)、[shader 與輸入規格](https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/ffx-fsr/ffx_fsr1.h)
- 工具：[FidelityFX CLI v1.0.3](https://github.com/GPUOpen-Effects/FidelityFX-CLI/releases/tag/v1.0.3)，ZIP、exe、MIT license 與 README 保存在 `tools/fsr1/`。雜湊隨評估 report 保存。
- 初次 sandbox 網路請求失敗，經授權連線下載成功。直接試工具 `-help` 回傳 exit 1 `ERROR: Unknown command line option.`，未執行影像測試；後續依官方 README 使用正式參數。
- 狀態：接入實作中；所有試跑、失敗、完整評估透過 run_logged 保存，不覆寫結果。完整視覺、獨立影片泛化及即時效能另行驗證。

### 完成狀態補記

上述「接入實作中」為執行前狀態；自動紀錄 [完整評估](./experiments/records/evaluation--eval-fsr1-v9-q5--run-ee6041531ea6.json) 已為 `completed`。以下數值取自 [report.json](./runs/eval-fsr1-v9-q5/report.json)，Q=5、frame 720…899、5,400 條；沿用 plain16 五輪訓練最佳第 5 輪權重，未重訓。

| 方法 | RAW 投影 MAE | RAW PSNR (dB) | RGB 參考 MAE | RGB PSNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| v9 plain16 | 16.626771 | 20.079337 | 15.482858 | 20.788811 |
| FSR 1 EASU 2x＋2x＋RCAS 0（主要基準） | 17.205235 | 19.849065 | 16.214781 | 20.492716 |
| FSR 1 EASU 2x＋2x | 17.071351 | 19.906588 | 16.056117 | 20.565029 |
| RAW bilinear（輔助） | 17.544584 | 19.708695 | 16.156488 | 20.550097 |

既有同影片保留集的平均 MAE／PSNR 優於兩種 FSR 對照；此結論僅涵蓋平均指標。逐區塊勝率與按場景分析尚待補充，完整視覺、新影片泛化及即時效能仍未驗收，不能宣告完整通過。Q=4 與獨立來源影片也不能沿用本次結果。[六欄對照影片](./runs/eval-fsr1-v9-q5/comparison.mp4) 為離線輸出。

## 檢查與重現

31 項測試通過，新增 RGB→RGGB 取樣、全 16 相實體座標的仿射場景解析解、官方 CLI 的常色保持，以及不同內容相鄰 tile 與單獨 tile 的輸出一致（容許 1/255 量化差異）。批次隔離檢查同時涵蓋 EASU 與 RCAS。原有 28 項測試繼續通過。

完整評估前執行 frame 720 單幀 30 條流程檢查，`runs/smoke-fsr1-v9-q5/` 已保存；此 smoke 不作完整品質結論。

工具 ZIP SHA256 `d280f245730c6d163c0e072a881ed4933b32e67b9de5494650119afa9649ea11`；exe SHA256 `0d0016683ab9331968b76963a8509b717d43652440f1adbf5a743aa2d950cbbd`，另見 `tools/fsr1/manifest.json`。

```powershell
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/plain16-v9-q5/best.pt --start-frame 720 --video --out Model/runs/eval-fsr1-v9-q5
```

重跑須選新輸出目錄。`evaluate_fsr.py` 不重新訓練；需要 Windows／D3D11 與官方工具。輸入／兩次 EASU／RCAS 的所有 atlas PNG 與官方 CLI log 均保存於輸出 `official-cli/`；每個 atlas 最多 128 條、16 欄、各自 8 像素複製邊界，沒有互相引用輸入內容。
