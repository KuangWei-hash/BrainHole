# 共用標頭封裝 v1

## 目的與範圍

在保留原始 JPEG XS 碼流可還原性的前提下，將重複標頭集中保存，AI 每筆只輸入變動資料（payload）。這是 BrainHole 自訂封裝，不是每個區塊各自完整的標準 JPEG XS 碼流；需要標準解碼器時先重建完整碼流。

目前正式設定為 Q=6、levels=1/0，原生 Go 使用 Br=4。原始輸入為 160 × 4 RGGB8，編碼器拆成四個 80 × 2 平面，Cpih=3。封裝只支援此幾何、單 slice、單 precinct 欄的 Bayer 子集。不同設定或不支援的碼流回報錯誤，不猜測格式。

## 共用內容與逐筆資料

```text
完整碼流 = prefix + payload + suffix

prefix：SOC、CAP、PIH、CDT、WGT、CTS、CRG 等標頭，加第一個 SLH
payload：SLH 之後、EOC 之前的資料，含 precinct／packet 的必要結構
suffix：EOC（FF 11）
```

程式解析 marker 長度尋找分界，**不固定刪除前 100 bytes**。目前設定實測 prefix=100 bytes、suffix=2 bytes。

PIH 的 `Lcod` 若為 0，重建時保持 0；若為完整碼流長度，範本內將此 4-byte 欄位歸零，重建時依該筆長度補回。其他任何 prefix/suffix 位元改變都視為設定變更。每次分離都重組並逐 byte 比對原始碼流。

每個資料庫／串流只接受一個不可變的共用設定。跨區塊、跨幀與續跑皆檢查同一設定 ID；若有變化就停止並記錄錯誤，需另建資料庫或開啟新串流，不靜默切換設定。

## 設定 ID

設定 ID 是 SHA-256 小寫十六進位字串，綁定封裝版本、編碼器 identity、範本與 Lcod 還原規則。所有整數使用 big-endian，長度以 byte 計算，hash 的輸入依序為：

| 欄位 | 編碼 |
| --- | --- |
| domain | ASCII `BHXH`，4 bytes |
| version | uint16，值 1 |
| identity 長度、內容 | uint16 + UTF-8 bytes，上限 4096 bytes |
| prefix 長度、內容 | uint16 + bytes；Lcod 已歸零 |
| suffix 長度、內容 | uint16 + bytes |
| Lcod 在 prefix 中的位置 | uint16 |
| 是否補回 Lcod | uint8，0 或 1 |

設定的儲存／傳輸表示使用 JSON，prefix/suffix 是標準 Base64。**只有共用設定採 JSON/Base64；逐筆 payload 直接使用二進位 bytes。** SHA-256 檢查設定一致性，不替代傳輸層認證或 payload 完整性保護。

## SQLite schema v3：RAW8 標籤

`PRAGMA user_version=3`，每部影片一個資料庫。v1 保存完整碼流與 RGB 標籤，v2 改為共用標頭與 RGB 標籤，v3 再把標籤改為 RAW8 RGGB。BHXS 串流不包含訓練標籤，因此仍維持版本 1。

```text
dataset_info
  config             來源、影片分組、尺寸、30 FPS 規則、編碼器等 JSON
  status             完成後為 complete

shared_header        每個資料庫只允許一筆
  singleton          固定 1
  id                 設定 SHA-256
  definition         SharedHeader JSON（version、id、encoder_identity、
                     prefix、suffix、length_offset、patch_length）

samples              每個區塊一筆
  frame_index / source_frame_index / source_timestamp_us
  block_index        0..29
  phase_id           (frame_index % 16) + 1
  header_id          外鍵對應 shared_header.id
  jpegxs_payload     BLOB，只保存變動資料
  valid_length       payload 的實際 byte 數
  codestream_length  可重建的完整 JPEG XS byte 數
  raw_bayer_target   降採樣前的 RAW8 RGGB，row-major [16,640]，10,240 bytes

errors               編碼／封裝失敗紀錄
```

完整碼流仍須不超過 640 bytes，payload 必須非空，並檢查 `codestream_length = len(prefix) + valid_length + len(suffix)`。第一個共用範本與整幀 30 筆樣本在同一交易提交；失敗不留下半幀或孤立範本。標頭不一致會停止，不截斷資料。

**舊版資料庫不自動遷移或覆寫**。新資料產生／續跑僅接受 schema v3 且來源／處理設定完全相同的資料庫。`dataset-convert-target` 可將已完成的 RGB v2 另存為 RAW8 v3，記錄來源資料庫雜湊、保留編碼資料與配對資訊。匯出器可讀 v2/v3。

## AI 輸入

訓練與推理皆使用相同版本的 `jpegxs_payload`，不將共用範本重複送入每筆模型輸入。

```text
L = valid_length
model_input = jpegxs_payload + 零填補至 640 bytes
valid_mask[i] = i < L
stored_target = 對應的 640 × 16 RAW8 RGGB
```

0x00 也可能是有效內容，不能依值是否為 0 判斷遮罩。phase_id 保留為中繼資料；是否額外輸入模型仍由模型設計決定。模型與資料載入器應綁定封裝版本及允許的編碼器設定，避免混用舊版完整碼流輸入。

最終輸出目標仍是 RGB。RAW8 只保存對應位置的一個色彩通道，不能完整恢復轉存前 RGB。模型訓練可考慮將預測 RGB 投影至相同 RGGB 後與標籤比較，或在載入時去馬賽克得到近似 RGB 參考；具體 loss 與生成方式尚未實作，不將近似參考當作原 RGB ground truth。

## BHXS 串流 v1

提供 `.bhxs` 檔案匯出、逐區塊 writer/reader。可放在有序可靠的 byte stream（檔案、TCP、QUIC stream）上；目前沒有實作網路連線、UDP 封包化、重傳或模型服務。

串流開始只送一次：

| 欄位 | 大小 |
| --- | ---: |
| magic | ASCII `BHXS`，4 bytes |
| 串流版本 | uint8，1 |
| 設定 JSON 長度 | uint32 big-endian，上限 65,536 |
| 設定 JSON | header 加 RGB 尺寸、區塊高度、FPS、16 相、RGGB、模型補零長度 |

設定中的固定尺寸為 RGB 640 × 480、區塊高 16、30 FPS、16 相、padding=640。讀取端驗證版本、設定 hash 與支援的幾何。

之後每幀：

```text
frame_index：uint64 big-endian，範圍 0..2^63-1
    block 0：uint16 payload_length + payload bytes
    block 1：uint16 payload_length + payload bytes
    …
    block 29：uint16 payload_length + payload bytes
```

首幀序號可非 0，之後必須連續。區塊依序 0..29，位置與 phase_id 由幀序號推得，避免逐筆重複傳送。writer 與 reader 在一個區塊到齊時立即處理，**不等待整幀**。串流截在半個欄位或半幀時回報錯誤；不支援中途切換設定，需重新建立串流。

不在 payload 中增加填零或重複範本。封裝不額外引入像素損失；有損部分來自 JPEG XS 的量化設定，目前固定 Q=6。

## 目前 Q=6 的實際影片驗證

Big Buck Bunny 30 秒、900 幀、27,000 筆、16 相、原生 Go Q=6／Br=4：payload 為 44..202 bytes，平均 95.2903 bytes；加入每筆 2-byte 長度、每幀 8-byte 幀序號，資料率約 **702,410 bps**，未含初始設定與網路封包。SQLite 完整性、逐幀筆數、相位、RAW8 標籤與碼流重建檢查通過。以編碼後且解碼為全 128 的範本比對，439 筆（1.63%）輸入相同。

## 先前 Q=8 的頻寬與驗證紀錄

第一幀相位 1 的舊實驗平均 payload=75.5 bytes。加入正式封裝的每筆 2-byte 長度、每幀 8-byte 幀序號：

```text
((75.5 + 2) × 30 + 8) × 30 × 8 = 559,920 bps
```

17 幀移動 testsrc、Q=8、全部 16 相及回繞至相位 1 的真實 Rust 編碼測試共 **510 筆**：

- 共用 prefix=100、suffix=2，在所有幀與相位皆一致。
- payload 為 56..84 bytes，平均約 75.03 bytes。
- 依實際總 byte 數推算約 **556,532 bps**（含長度與幀序號，未含一次性的設定與網路封包）。
- SQLite 與 `.bhxs` 的完整碼流重建一致，首尾幀重建資料通過上游解碼器解碼。

另外通過標頭漂移、Lcod 長度還原、未知版本、錯誤長度、資料截斷、舊版資料庫保護、交易回滾與首區塊先於整幀送達測試。這些結果不代表已驗證長時間部署、獨立解碼器互通性或 AI 生成品質。

## 使用方式

在 `training-data-preparation` 建置：

```powershell
go build -o bin/video-to-dataset.exe ./cmd/video-to-dataset
go build -o bin/dataset-export.exe ./cmd/dataset-export
go build -o bin/dataset-convert-target.exe ./cmd/dataset-convert-target
```

資料產生程式現在預設寫入 schema v3。匯出整個資料庫為封裝串流：

```powershell
.\bin\dataset-export.exe -input "拆好的訓練資料\video001.sqlite" -output "video001.bhxs"
```

還原指定樣本為完整 JPEG XS：

```powershell
.\bin\dataset-export.exe -mode codestream -input "拆好的訓練資料\video001.sqlite" -frame 0 -block 0 -output "sample.jxs"
```

匯出不覆寫既有檔案；錯誤時會報告已留下的部分輸出檔。`Store.RestoreSample`、`Store.ExportStream`、`NewStreamWriter`、`NewStreamReader` 為 Go 呼叫介面。逐區塊 reader 回傳 payload、frame/block/phase 與 header_id。
