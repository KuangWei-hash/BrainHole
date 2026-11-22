# EchoSR-Lite ×4 預訓練模型直接處理 Q=5 decoded

## 目的與來源

依使用者要求，直接以訓練資料庫中的實際 decoded 測試 EchoSR-Lite。這次是官方預訓練權重的直接推論，不做專案資料訓練或微調，不是 RAW 版 EchoSR 新模型。

官方專案：[funnyWang-Echoes/EchoSR](https://github.com/funnyWang-Echoes/EchoSR)，Apache-2.0；使用 commit `09999643d2088ddbf39539244b046b9e4a951c3a`，官方 `options/EchoSR_test/test_EchoSR_lite_SRx4_DF2K.yml` 的設定：embed_dim=36、depths=[2,3,2,3]、mlp_ratio=1.5、kernel_sizes=[0,5,11,17]、RGB、pixelshuffledirect、×4。

官方權重 `experiments/EchoSR_x4_lite_DF2K/models/EchoSR_x4_lite_DF2K.pth`，SHA256 `fc5a0b7c68d2431749ddd6b8833d8a090d7d3d8315aa2e92cd67faebf7a7e272`，state key=params，嚴格載入成功，共 **258,646** 參數。架構來源 SHA256 `8dcaadc6c2b461ea8cfa4afb7687e4a7f3cdbfa5c32dad395a23b6d1f022093b`。

## 資料、介面與限制

資料庫：`training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite`，實際解碼快取：`Model/runs/decoded-q5-v4.npz`。只使用當前幀本條 decoded 與 phase，不讀上一條、前幀、未來幀或標籤產生輸入。

官方模型接受 RGB，不能把單通道 Bayer RAW 直接當 RGB。沿用既有 FSR 的 `low_rgb` 相位／CFA 座標轉換，把 160×4 RAW 轉成 160×4 RGB，兩綠平面依原規範合併，再做 PNG8 等價量化，與 FSR 的實際低解析輸入相同。

EchoSR 的 GCE 使用 `h//8,w//8` 全域池化，原 4 列輸入無法直接使用。因此每條四邊各 replicate 8 像素，得到 176×20 RGB；模型 ×4 後裁掉各邊 32 像素，保留 640×16 RGB，clamp [0,1]。這和 FSR atlas 的 guard8 邊界延伸一致，每個條帶獨立處理，不將不同條帶拼成一張圖。

RGB 指標以同一個目標 RAW 顯色的 RGB 為參考，不是原始完整 RGB；RAW 指標將 EchoSR 的 RGB 依 RGGB 投影後比較，與 FSR 一致。v9 則是原本的 RAW 輸出；兩種指標分開報告。

`echosr_adapter.py` 原樣載入官方網路運算與權重，為避免更動既有 PyTorch 環境，只在載入時去除 BasicSR registry／未使用的 tqdm 匯入，將 timm 工具替換為推論等價工具：eval DropPath 為恆等、初始化用 torch.nn.init.trunc_normal_。沒有改卷積、池化或權重；此 adapter 不支援訓練。原始架構、官方設定與 LICENSE 另複製到每次評估目錄的 `echosr-source`，report 保存 commit、來源及權重雜湊。

## 範圍與比較契約

- 流程檢查：frame 721、30 條，僅驗證 strict load、形狀、有限輸出與資料／相位／官方 FSR 介面，不作完整品質結論。[流程 report](./runs/smoke-echosr-lite-q5-20261121/report.json)。
- 完整評估：固定 frame 721…899、5,370 條，與既有雙幀窗口一致。雖來自同一訓練資料庫，評分只取保留集，EchoSR 本次沒有訓練預算。
- v9 單幀參考：四層 relu128，原五輪、train 21,600 條、validation 5,400 條、最佳第 4 輪；本次只在相同 5,370 條上重新推論。
- v9 雙幀：四層 relu128，原五輪、train 21,570 條、validation 5,370 條、最佳第 5 輪，僅作目前方案參考；輸入比 EchoSR 多前幀，不能把所有差異都歸因於架構。
- FSR：官方 EASU 2x＋2x、另加 RCAS 0，沒有訓練；當前本條低解析 RGB 相同，固定 guard8。另列 RAW bilinear 與相同 RGB 輸入 bilinear。

## 狀態

30 條流程檢查與 5,370 條完整評估均已完成，179 幀、30 FPS 的八欄比較影片已輸出。官方 strict load 成功、輸出形狀與有限值檢查均通過。兩次入口均經 run_logged，保留來源快照、資料指紋與逐條紀錄；沒有另外重跑整套既有單元測試。

## 完整結果

來源：[report.json](./runs/eval-echosr-lite-q5-20261121/report.json)、[逐區塊結果](./runs/eval-echosr-lite-q5-20261121/blocks.jsonl)。全部為 Q=5 frame 721…899、5,370 條，不能把前面的單幀流程檢查分數混進本表。

| 方法 | RAW 投影 MAE ↓ | RAW PSNR ↑ | RGB 參考 MAE ↓ | RGB PSNR ↑ | RAW 最差 10% MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| EchoSR-Lite ×4，官方 DF2K 預訓練 | 17.194935 | 19.836919 | 16.212310 | 20.476424 | 25.850174 |
| 官方 FSR EASU 2x＋2x＋RCAS 0 | 17.213109 | 19.842178 | 16.223156 | 20.485277 | 25.903681 |
| 官方 FSR EASU 2x＋2x | 17.079363 | 19.899603 | 16.064604 | 20.557476 | 25.814304 |
| 相同 RGB 輸入 bilinear | 17.208636 | 19.914219 | 16.131054 | 20.590353 | 26.693122 |
| RAW bilinear | 17.552326 | 19.701981 | 16.164984 | 20.542705 | 26.220262 |
| v9 四層 relu128 單幀 | 15.241226 | 20.544308 | 13.838357 | 21.407731 | 23.819960 |
| **v9 四層 relu128 雙幀** | **14.171117** | **21.282882** | **12.676014** | **22.261846** | **21.957133** |

EchoSR 對 FSR＋RCAS 的 RAW MAE 僅改善 **0.018173（0.106%）**，RAW PSNR 反而低 0.005259 dB；RGB 也是 MAE 略低、PSNR 略差。對無 RCAS 的 FSR，RAW／RGB 平均指標均較差。尚未證明有實質品質收益，也未滿足現有「MAE 與 PSNR 都勝主要對照」的數值初篩條件。

EchoSR 的 RAW MAE 逐區塊勝率：對 FSR＋RCAS **58.31%**、對無 RCAS FSR **10.43%**、對相同 RGB 輸入 bilinear **26.89%**、對 v9 單幀 **0.74%**、對 v9 雙幀 **0.09%**、對 RAW bilinear **98.90%**。能贏 RAW bilinear 不代表比相同 RGB 輸入的放大方法更好。

## 視覺與負面案例

已查看 frame 750 的八欄比較 PNG，EchoSR 標籤與排列正常，輸出仍有顯著條帶與塊狀紋理，未看到它補回目標中的樹葉／草地細節。這只是固定畫面的觀測；完整影片尚未逐幀觀看，不宣告視覺或時間一致性驗收通過。

對 FSR＋RCAS 的最大退步出現在多個影格的 block 10（畫面第 160…175 列）：frame 807 的 EchoSR MAE **19.685877**、FSR **19.124805**，增加 **0.561072**；frame 855 增加 0.523711、frame 839 增加 0.512184。前十筆保留於 report 的 `echosr_largest_regressions_vs_fsr`；尚未逐條確認場景原因，不猜測是特定物體或運動造成。

此測試只回答「官方 RGB 預訓練權重直接處理本專案條塊 decoded」的效果。原模型的 RGB 超解析資料與本專案相位採樣、有損編碼、僅 4 列高的輸入不同，可能存在分布與上下文不足，但本次沒有消融，不能確定各因素的影響。不能據此宣告 EchoSR 架構在專案重訓後也一定無效；本次也不是生成式或擴散模型試驗。

目前保持 v9 雙幀為較佳候選，不替換為 EchoSR。沒有量測 EchoSR batch=1／端到端延遲，不以批次離線評估耗時宣告速度或即時能力。

## 產物

- [EchoSR-Lite 單獨影片](./runs/echosr-lite-only-q5-20261121/echosr-lite.mp4)：依使用者要求，從比較影片第 8 格裁出畫面，移除標籤與其他對照。640×480、30 FPS、179 幀、5.966667 秒，FFprobe 已核對。沒有重新推論；原比較影片已經有 H264 壓縮，這次以 CRF 0 無損重編碼其解碼後的裁切畫面，避免增加編碼損失。`export_video_panel.py` 經 run_logged 保存來源／輸出雜湊、裁切座標、完整命令與探測結果，見 [匯出 report](./runs/echosr-lite-only-q5-20261121/report.json)。

- [比較影片](./runs/eval-echosr-lite-q5-20261121/comparison.mp4)、[比較圖](./runs/eval-echosr-lite-q5-20261121/comparison.png)。
- [完整 report](./runs/eval-echosr-lite-q5-20261121/report.json)、[逐條結果](./runs/eval-echosr-lite-q5-20261121/blocks.jsonl)、[來源封存](./runs/eval-echosr-lite-q5-20261121/echosr-source/EchoSR_arch.py)。
- 官方預訓練權重保存在 `Model/tools/EchoSR/experiments/EchoSR_x4_lite_DF2K/models/`；本次沒有產生新訓練權重。
- 完整評估與流程檢查分別存入 `experiments/records`，逐條 gzip 封存於 `experiments/details`；見 [實驗總表](./實驗總表.md)。

## 重現

在專案根目錄執行；重跑選新 out。

```powershell
git clone --depth 1 https://github.com/funnyWang-Echoes/EchoSR.git Model/tools/EchoSR
.\Model\.venv\Scripts\python.exe Model/evaluate_fsr.py --db training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q5-shared-v4-raw8.sqlite --decoded-cache Model/runs/decoded-q5-v4.npz --checkpoint Model/runs/relu128-four-v9-history2-q5/best.pt --single-frame-reference Model/runs/relu128-four-v9-q5/best.pt --echosr-lite --start-frame 720 --video --out Model/runs/eval-echosr-lite-q5-20261121
```

克隆後須核對／取用上列 commit，不能把未來 main 的內容當作本次程式版本。
