# OxideAV JPEG XS RGGB bridge

本轉接器將 Go 程式輸出的 **160 × 4 RAW8 RGGB（640 bytes）**送入 [OxideAV/oxideav-jpegxs](https://github.com/OxideAV/oxideav-jpegxs) 的 `encode_components`，使用 Star-Tetrix `Cpih=3`。

依賴固定在 `3484a041d6d8e893683c1920cb3f68fe6fd3b8b8`（crate 0.0.9），停用預設 framework feature，使用 MIT 授權。需要 Rust 1.95+；Windows MSVC 版本需 C++ Build Tools。回到 `training-data-preparation` 執行 `./build-jpegxs.ps1`，產出 `bin/brainhole-jpegxs.exe`。

## RGGB 與碼流尺寸

RGGB 交錯資料拆為 R、G1、G2、B 四個 **80 × 2** 平面。碼流的 component grid 為 **80 × 2、4 components**，用 CRG 表示 RGGB 位置，代表的原始 CFA 是 **160 × 4**。不是把輸入補成四個 160 × 4 平面，也沒有經過 RGB 去馬賽克。

## 目前正式設定

原生 Go 與 Rust 轉接器的 preset 皆已改為 Q=6。正式資料使用原生 Go Br=4，Big Buck Bunny 27,000 筆的 payload 平均 95.29 bytes，主體含長度與幀序號約 702.4 kbps；全 128 範本相同輸入為 1.63%。下方 Q=8 表格是歷史比較，生成品質尚未驗證。Rust 上游轉接器仍使用 Br=8，與原生 Go 正式設定的範本不同，不能混用編碼設定。

## 碼率策略

上游這一版的 `target_bytes` CBR 分支拒絕 `Cpih=3`，因此不使用該介面，也不加 COM padding 強制補長。轉接器實際編碼 Q=0..15，輸出變動長度：

1. 優先選擇能符合 `--target-bytes` 的最低 Q。
2. 若目標無法達成，改選符合 `--max-bytes` 的最低 Q；這是容量上限退讓策略，不代表達成目標壓縮倍率。
3. 沒有任何候選符合上限就報錯，不截斷碼流。

轉接器單獨執行且未指定 Q 時，使用上述自動策略，預設目標 80 bytes、上限 640 bytes、水平小波層數 1、垂直 0。**目前 Go 使用的 `encoder.oxideav.json` 已指定 `--quantization 6`**，只編碼 Q=6，不自動換 Q；若超過 640 bytes 則報錯。Q=6 是量化參數，不代表 6:1 壓縮。

可用 `--quantization N` 固定 Q（0..15）；設定會列入 Go 的編碼器參數紀錄。每個選定輸出都會由上游解碼器自解碼驗證尺寸；Q=0 額外檢查逐 byte 相同。這是同實作往返測試，尚未經另一套獨立 JPEG XS 解碼器驗證互通性。

## 命令

```powershell
.\bin\brainhole-jpegxs.exe encode --input input.rggb8 --output output.jxs --target-bytes 80 --max-bytes 640
.\bin\brainhole-jpegxs.exe decode --input output.jxs --output reconstructed.rggb8
.\bin\brainhole-jpegxs.exe sweep --input input.rggb8
```

`decode` 輸出仍是 160 × 4 Bayer，不是 RGB。`sweep` 輸出 CSV，量測水平 1..5、垂直 0..1、Q=0..15 的長度，並自解碼檢查。輸出檔若已存在會拒絕覆寫。

模型離線對照可用 `decode-batch --input records.batch --output decoded.raw`：輸入為重複的
`u32 big-endian 碼流長度 + 完整 JPEG XS 碼流`，每筆最多 640 bytes；輸出按順序串接每筆 640-byte RAW。
截短、零長度或無法解碼任一筆即失敗，不略過錯誤。這個批次格式只供離線解碼，不是 BHXS 網路封裝。

## 2026-11-20 實測

### 先前設定：固定 Q=8（比較基準）

levels=1/0、上限 640 bytes。相同 testsrc 一幀相位 1 的 30 個區塊，完整碼流為 **164..186 bytes、平均 177.50 bytes**，推算 30 FPS 為 **1.278 Mbps**。目前 SQLite v3 已改為共用範本加逐筆 payload，完整碼流可重建，見[封裝規格](../../shared-header-v1.md)。既有 Q=4 或 schema v1 資料庫須使用不同版本另存，不混用設定。

### 先前設定：固定 Q=4（比較基準）

保持 levels=1/0、完整碼流上限 640 bytes，以同樣輸入重新測試：

| 測試輸入 | 輸出 bytes |
| --- | ---: |
| 全部為 128 的平坦 Bayer | 146 |
| 人工梯度／循環圖樣 | 549 |
| 固定種子的偽隨機資料 | 594 |
| FFmpeg testsrc，一幀相位 1 的 30 個區塊 | 306..516，平均 461.03 |

30 個 testsrc 區塊全部成功編碼、自解碼並寫入 SQLite。相較先前自動策略的平均 597.63 bytes，固定 Q=4 約減少 **22.9%**。若該平均大小維持在 640 × 480、30 FPS，影像碼流推算為 **3,319,440 bps（約 3.319 Mbps）**，不含封包開銷。仍未達成 80 bytes／8:1，亦尚未評估 AI 輸出品質。

### 先前設定：自動選 Q（比較基準）

使用本機 release build，預設 levels=1/0、target=80、max=640，對 640-byte 輸入測試：

| 測試輸入 | 選定輸出 bytes | Q | 達成 80-byte 目標 |
| --- | ---: | ---: | --- |
| 全部為 128 的平坦 Bayer | 146 | 0 | 否 |
| 人工梯度／循環圖樣 | 549 | 4 | 否 |
| 固定種子的偽隨機資料 | 594 | 4 | 否 |
| FFmpeg testsrc，一幀相位 1 的 30 個區塊 | 553..625，平均 597.63 | 逐區塊選擇 | 否 |

平坦資料掃描全部上述層數與 Q 組合，觀察到最小為 **140 bytes**（levels=2/0）；這是該次設定掃描結果，不是 JPEG XS 標準的普遍最小值。

testsrc 的平均大小若維持在每秒 30 幀、每幀 30 區塊，推算影像碼流約 **4,302,960 bps**，未計封包開銷；這是人工測試圖樣的大小推算，不是部署流量保證。原先 576 kbps／80 bytes 的規劃仍是未達成的目標。

驗證包含：多種圖樣下 levels=1..5/0..1 的 Q=0 無損往返、Q=1..15 的有損碼流解碼、Go 真實影片解碼到 SQLite 的一幀整合測試。沒有測量 AI 生成品質。

上游證據：[Star-Tetrix 編碼與 target_bytes 分支](https://github.com/OxideAV/oxideav-jpegxs/blob/3484a041d6d8e893683c1920cb3f68fe6fd3b8b8/src/api.rs)、[component 與 plane 結構](https://github.com/OxideAV/oxideav-jpegxs/blob/3484a041d6d8e893683c1920cb3f68fe6fd3b8b8/src/image.rs)。

## 碼率瓶頸與共用標頭的先期量測

對相同的一幀 FFmpeg testsrc、相位 1、30 個區塊，固定 Q 掃描結果：

| levels | Q | 完整碼流 bytes（最小..最大） | 平均 bytes | 30 FPS 推算 bps |
| --- | ---: | --- | ---: | ---: |
| 1/0 | 4 | 306..516 | 461.03 | 3,319,440 |
| 1/0 | 6 | 222..337 | 305.73 | 2,201,280 |
| 1/0 | 8 | 164..186 | 177.50 | 1,278,000 |
| 1/0 | 10 | 146..146 | 146.00 | 1,051,200 |
| 1/0 | 12、15 | 146..146 | 146.00 | 1,051,200 |
| 2/0 | 4 | 260..567 | 501.23 | 3,608,880 |
| 2/0 | 6 | 192..377 | 334.60 | 2,409,120 |
| 2/0 | 8 | 156..214 | 198.63 | 1,430,160 |
| 2/0 | 10 | 140..142 | 140.13 | 1,008,960 |
| 2/0 | 12、15 | 140..140 | 140.00 | 1,008,000 |

另測人工梯度在 Q=10、levels=1/0 的解碼結果，全部 640 個像素值均為 128。提高 Q 不能只看 byte 數，可能已失去可供 AI 依據的輸入資訊；尚未評估 Q=8 的生成品質。

解析 levels=1/0 的實際碼流：main header 為 **94 bytes**，slice header 為 **6 bytes**，EOC 為 **2 bytes**。獨立完整碼流光是這些結構就有 **102 bytes**，此設定下不可能得到 80-byte 完整碼流。這是目前編碼設定的分析，不是所有 JPEG XS 設定的普遍下限。

Q=8 的共用標頭實驗已驗證 30 筆的前 100 bytes 全部相同，EOC 也固定；將兩者集中保存後，每筆變動資料為 **62..84 bytes，平均 75.5 bytes**。每筆另加 2-byte 長度欄位時，推算：

```text
(75.5 + 2) × 30 blocks/frame × 30 frames/s × 8
= 558,000 bps
```

先期實驗確認 30 筆皆能用共用前綴、變動資料與固定 EOC **逐 byte 重建原本已自解碼驗證的完整碼流**。目前此檢查已整合到 SQLite v3 與串流測試，不再需要 `BRAINHOLE_ANALYZE_SHARED_HEADERS` 開關。

同樣條件下，**Q=6** 的共用標頭實驗也通過 30 筆逐 byte 重建：完整碼流為 222..337 bytes、平均 305.73 bytes（推算 2,201,280 bps）；扣除相同的 102-byte 固定結構後，變動資料為 **120..235 bytes、平均 203.73 bytes**。每筆加 2-byte 長度，推算為 **1,481,280 bps（約 1.481 Mbps）**，仍未含網路封包與其他中繼資訊，也未驗證模型生成品質。此為實驗設定，正式設定為 Q=6。

上述 558 kbps 是先期單幀實驗，尚未計幀序號與初始設定。現已完成正式的[共用標頭封裝 v1／SQLite v3](../../shared-header-v1.md)：Q=8，17 幀、510 筆涵蓋全部 16 相及回繞，payload 平均約 75.03 bytes，加入每筆長度與每幀序號後約 **556.5 kbps**，未含初始設定與網路封包。SQLite 保存一份共用範本與每筆 payload，不再逐筆保存重複標頭。長時間影片、模型品質與實際網路表現仍待驗證。
