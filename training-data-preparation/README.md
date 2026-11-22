# Video to Training Dataset

Go 影片轉訓練資料工具，依本專案的[做法.md](做法.md)產生每部影片一個 SQLite。

目前預設使用 **Q=4、SQLite schema v4／共用標頭封裝**。每筆保存 `jpegxs_payload`、同筆碼流由原生 Go decoder 產生的 160×4 `decoded`，以及 640×16 RAW8 RGGB 標籤。完整碼流仍可重建，舊資料庫不覆寫。格式見[共用標頭規格](shared-header-v1.md)。

先前 Q=8 的 Big Buck Bunny 資料出現 81.96% 輸入相同且解碼為全 128，該版已退為比較資料。實際結果見[做法.md](做法.md)，統計保存在資料庫旁的 `.summary.json`，不能只依低碼率判斷資料可用性。

這批已轉成 `big-buck-bunny-30s-q8-shared-v3-raw8.sqlite`，約 **332 MB**；原 RGB 版本約 885 MB。27,000 筆 RAW8 標籤已逐筆與原 RGB 的 RGGB 取值核對，編碼輸入與 BHXS 匯出完全相同。舊版保留作備份。

## 目前使用的資料與編碼設定

目前正式使用 **Q=4、levels=1/0、原生 Go Br=4**，且已涵蓋**全部兩支原始影片**：

| 資料庫 | split | 幀數 | 筆數 | payload 平均 | 完整碼流最大 | 封裝主體 | decoded 全 128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite` | train | 900 | 27,000 | 248.52 bytes | 515 bytes | 1.805 Mbps | 17 筆（0.063%） |
| `tears-of-steel-q4-shared-v4-raw8.sqlite` | validation | 22,024 | 660,720 | 239.07 bytes | 516 bytes | 1.737 Mbps | **0 筆** |

兩份資料庫 errors=0、完整性檢查通過，payload-only 封裝主體數字未含共用設定與網路封包；全部 Go／Rust 解碼結果逐 byte 相同（Tears of Steel 660,720 筆的 SHA-256 兩端一致）。兩者共用標頭 ID 相同，因此可直接用 `train.py --val-db` 做**不同來源影片**的驗證分組。Tears of Steel 的產生與驗證紀錄見 [Q=4 重建紀錄](../Model/實驗紀錄-Q4訓練資料重建.md)。

Tears of Steel（512×292／23.999 fps、734 秒）被拉伸到 640×480 並重採樣到 30 FPS，資料量為 Big Buck Bunny 的 24.5 倍（資料庫約 7.58 GB）。Q=5 的資料、模型與曲線保留未覆寫，見 [Q=5 資料紀錄](../Model/實驗紀錄-Q5訓練資料更新.md)。

`encoder.native.json` 與 `encoder.oxideav.json` 預設均為 Q=4；明確版本為 `encoder.native.q4.json`／`encoder.oxideav.q4.json`。Q=5 設定為 `encoder.native.q5.json`，Q=6 設定亦保留。歷史 Q=4 完成資料名稱含 fixed，payload 平均 248.52 bytes、封裝主體約 1.805 Mbps、全 128 解碼 17 筆；未完成的 Q=4 原檔不可用於訓練。

歷史 Q=6 payload 平均 95.29 bytes、封裝主體約 702.4 kbps、全 128 解碼為 439 筆（1.63%）。Q=4 與 Q=6 均為 Br=4；更早 Q=8 使用 Br=8，故該歷史碼率比較不只變更 Q。
## 已實作

- 由 FFprobe 取得來源幀時間戳，FFmpeg 解碼為 RGB8 並直接縮放至 640 × 480。
- 在離線 30 FPS 時間軸選取時間戳最近的來源幀，距離相同時選較早幀；多餘幀丟棄、不足複製。避免毫秒時間戳取整導致原本 30 FPS 的影片被誤複製。相位按輸出幀序號計算，複製幀也輪相。
- 每幀由上往下切成 30 個 640 × 16 RGB8 區塊，直接取 RGB 通道模擬 RGGB，不進行線性化或感測器 ISP 模擬。
- 以完整 RGGB 為單位做 16 相取樣，生成 160 × 4 RAW8；同幀同相。
- 呼叫 JPEG XS 編碼器，保存變動長度碼流，超過 640 bytes 就報錯，不截斷、不偽造。
- 內建純 Go 的 JPEG XS 編碼器（`internal/jpegxs`），逐 byte 等同 OxideAV Rust 編碼器；不需要外部行程、Rust toolchain 或暫存檔。
- SQLite 將共用範本與變動 payload 分開，保存降採樣前的 RAW8 RGGB 目標、來源時間與相位；支援完整碼流重建、逐區塊串流匯出、整幀原子寫入、錯誤紀錄與設定核對後續跑。

兩種編碼器都可用：`-encoder-config encoder.native.json` 走行程內 Go 編碼器，`-encoder-config encoder.oxideav.json` 走外部 Rust 轉接器。原生 Go 正式設定使用 Br=4，Rust 上游轉接器使用 Br=8，視為不同編碼設定，不在同一資料庫或共用範本中混用。

這是離線資料準備工具。行程內 Go 編碼器已可取代原本「每區塊啟動一次外部編碼器」的做法（每幀 30 個區塊、30 FPS 即每秒 900 次行程啟動），但本工具仍是離線資料準備用途，沒有宣稱能滿足部署時的即時延遲。

## 環境與建置

需要 Go 1.24+、FFmpeg、FFprobe。SQLite 與 Go 主程式本身不需要 C 編譯器或 SQLite 服務。

Rust 轉接器是選用的。用預設的 `encoder.native.json`（行程內 Go 編碼器）時，只需要 Go 1.24+ 即可，不需要 Rust toolchain、暫存檔或外部行程。若要用 `encoder.oxideav.json`（外部轉接器），編譯它需要 Rust 1.95+ 及 Windows C++ Build Tools。

在本資料夾執行：

```powershell
go mod download
go test ./...
go build -o bin/video-to-dataset.exe ./cmd/video-to-dataset
go build -o bin/dataset-export.exe ./cmd/dataset-export
go build -o bin/dataset-convert-target.exe ./cmd/dataset-convert-target
```

需要外部轉接器時再執行 `.\build-jpegxs.ps1`。

## JPEG XS 編碼器設定

### 行程內 Go 編碼器（預設建議）

`encoder.native.json`：

```json
{
  "kind": "native",
  "identity": "...",
  "quantization": 5,
  "levels_x": 1,
  "levels_y": 0,
  "star_tetrix": true,
  "cts": { "e1": 0, "e2": 0, "cf": 0, "ct": 0 },
  "br": 4,
  "max_bytes": 640
}
```

| 欄位 | 意義 |
| --- | --- |
| `kind` | `"native"` 選行程內 Go 編碼器 |
| `identity` | 編碼器名稱、版本與 profile，供資料版本追溯 |
| `quantization` | 量化參數 Q，0..15；0 為無損 |
| `levels_x` / `levels_y` | 小波分解層數 NL,x（1..5）與 NL,y（0..1） |
| `star_tetrix` | 是否使用 Star-Tetrix 色彩轉換（Cpih = 3） |
| `cts` | CTS 參數：`e1`/`e2` 色度權重、`cf` 範圍、`ct` CFA 型態（0 = RGGB、1 = GRBG） |
| `br` | raw bitplane-count 寬度。預設 4（Table A.7 唯一合法值）；8 可重現上游 Rust 編碼器的位元組 |
| `max_bytes` | 碼流長度上限，超過就報錯 |

這個編碼器是 OxideAV/oxideav-jpegxs 0.0.9（rev `3484a041`）Rust 編碼器的 Go 移植，只涵蓋本專案需要的子集（Cpih 0/1/3、8-bit、NL,x 1..5、NL,y 0..1、Q 0..15、單一 slice 與 precinct 欄、無 CWD 抑制、無 refinement、無 NLT）。超出子集的設定會直接回報錯誤，不會靜默編出不同結果。

與 Rust 轉接器的實測差距（本機量測：AMD Ryzen 7 4800H、16 執行緒、Windows、Go 1.24.0）：同一個 160 × 4 區塊，行程內 Go 編碼 **59.8 µs**，外部轉接器 **16.7 ms**（每次要開檔、寫暫存檔、啟動行程，還會自我解碼驗證），約快 **280 倍**。各 levels 組合的單區塊耗時：levels 1/0 約 52 µs、3/1 約 84 µs。

離線管線端到端（Big Buck Bunny 300 幀、9,000 個區塊）：原生 Go **8.55 秒（35.1 fps）**，Rust 轉接器 **178.4 秒（1.68 fps）**。拆解原生路徑的 8.55 秒：FFmpeg 解碼縮放約 0.43 秒、JPEG XS 編碼約 0.54 秒、**SQLite 寫入 105.7 MB 約 7.6 秒**——離線管線的瓶頸是資料庫寫入，不是編碼器。

即時部署的算術：30 FPS × 每幀 30 個區塊 = 900 blocks/s，Go 編碼需 900 × 59.8 µs ≈ **每秒 54 ms 單核**（約一個核心的 5.4%）；Rust 轉接器則需每秒 15 秒，等於要 15 個核心才剛好跑完，即時串流不可行。

### 外部 Rust 轉接器

`encoder.oxideav.json` 配合 `build-jpegxs.ps1` 建立的 `bin/brainhole-jpegxs.exe`，可直接接上 Go 流程。它使用真實 Star-Tetrix Bayer 編碼，並自解碼驗證。原始 `encoder.example.json` 仍保留為其他編碼器的介面模板。

目前兩個編碼設定均固定 **量化 Q=4**、levels=1/0。Q 是量化參數，不代表 4:1 壓縮。完整碼流超過 640 bytes 就報錯，不自動改 Q 或截斷。原生 Go Br=4 的 Big Buck Bunny 900 幀 payload 平均約 248.52 bytes，計入長度與幀序號的封裝主體約 **1.805 Mbps**，未含共用設定與網路封包。

| 欄位 | 意義 |
| --- | --- |
| `executable` | 真實編碼器執行檔，建議絕對路徑 |
| `identity` | 編碼器名稱、版本與 profile，供資料版本追溯 |
| `timeout_seconds` | 單區塊編碼逾時秒數 |
| `args` | 編碼器參數陣列，每個元素為一個參數，不經 shell |

可用參數替換：`{input}`、`{output}`、`{width}` = 160、`{height}` = 4、`{bitdepth}` = 8、`{ratio}` = 8、`{max_bytes}` = 640。

外部編碼器必須將一個完整、可獨立解碼的 **bare JPEG XS codestream** 寫到 `{output}`。目前檢查長度與 SOC/EOC 標記，不是完整碼流驗證器；正式產生資料前需用配套解碼器驗證結果。若編碼器使用其他介面，可提供一個轉接程式遵循上述檔案契約。

本機 FFmpeg 的 `libsvtjpegxs` 只列出 YUV 輸入，故目前改用 OxideAV 轉接器。160 × 4 Bayer 已完成自編碼／自解碼與 Go 整合驗證；獨立解碼器互通性與目標碼率仍待驗證。

## 使用方式

將原始影片放入「原始影片」，建置完成後，在本資料夾執行：

```powershell
.\bin\video-to-dataset.exe -input "原始影片\video001.mp4" -output "拆好的訓練資料\video001.sqlite" -encoder-config encoder.native.json -split train -max-frames 2
```

先用少量幀驗證。移除幀數限制並接續同一資料庫：

```powershell
.\bin\video-to-dataset.exe -input "原始影片\video001.mp4" -output "拆好的訓練資料\video001.sqlite" -encoder-config encoder.native.json -split train -resume
```

若要改用外部 Rust 轉接器，把 `-encoder-config` 換成 `encoder.oxideav.json` 即可。兩者的差別只有 `Br`（原生預設 4、Rust 轉接器固定 8），其餘位元組相同；詳見[Br 的標準符合性](#br-的標準符合性重要)。

- `-max-frames` 是包含既有資料在內的輸出幀總上限，0 表示處理整部影片。
- 既有輸出檔預設拒絕覆寫；`-resume` 會核對來源 SHA-256 與處理設定，從最後完整幀接續。第一版續跑仍需重新解碼前面的來源幀。
- `-ffmpeg`、`-ffprobe` 可指定執行檔路徑。
- `-split` 為 `train`、`validation` 或 `test`；`-group` 設定獨立場景群組，預設為來源影片 SHA-256。需在影片層級分配 split；同群組使用相同 split。目前不自動決定比例，也不跨多個資料庫強制檢查群組衝突。
- 第一版一次處理一部影片；不是自動掃描資料夾的批次介面。
- 編碼失敗或超限會記錄至 `errors` 並停止，當前幀不會部分寫入。調整編碼設定後應另建資料庫，避免同一資料集混用設定。

## SQLite

| 表格 | 內容 |
| --- | --- |
| `dataset_info` | `config` JSON：來源雜湊、split、group、尺寸、處理方式、FFmpeg 版本與編碼器設定；完成時另有 `status=complete` |
| `shared_header` | 唯一的共用 prefix/suffix、Lcod 還原規則、編碼器 identity 與 SHA-256 設定 ID |
| `samples` | 每個區塊一筆，以 `(frame_index, block_index)` 為主鍵 |
| `errors` | 編碼失敗的幀、區塊與錯誤訊息 |

schema v4 的 `samples` 包含 `source_frame_index`、`source_timestamp_us`、`phase_id`、`header_id`、`jpegxs_payload`、`valid_length`、`codestream_length`、`raw_bayer_target`，以及 **640 bytes 的 decoded**（實際 JPEG XS 解碼出的 160×4 RAW8）。幀序號與區塊序號從 0 起，相位從 1 起。來源時間戳以來源影片的時間軸記錄；輸出時間為 `frame_index / 30` 秒。若共用標頭變動會停止；不對變動內容盲目刪除固定 bytes。

`raw_bayer_target` 為降採樣前、未壓縮的 RAW8 RGGB，row-major `[16,640]`，每筆 **10,240 bytes**。相較原 RGB8 的 30,720 bytes，標籤資料減少 2/3。這是保留每個位置對應的單一色彩通道，不是 RGB 的無損壓縮；最終仍要生成 RGB，監督方式需在模型設計時定義。`jpegxs_payload` 只存實際有效 bytes，不補零；模型載入端才補至 640 並建立遮罩。完整碼流大小記於 `codestream_length`。本 Go 工具不包含模型資料載入器或模型訓練。

把已完成的 RGB schema v2 資料庫另存為 RAW8 schema v4，保留來源並記錄轉換 provenance：

```powershell
.\bin\dataset-convert-target.exe -input "拆好的訓練資料\video001-v2.sqlite" -output "拆好的訓練資料\video001-v4-raw8.sqlite"
```

轉換不重新解碼影片或重新 JPEG XS 編碼；會實際解碼既有 payload 填入 decoded。它對原 RGB 標籤取 RGGB 通道，保留 payload、時間、相位與資料分組。新檔若已存在會拒絕覆寫。新資料產生與續跑要求 schema v4；匯出器可讀 v2/v3/v4。schema v3 可用 dataset-migrate 另存 v4；單純遷移不會變更 Q，改 Q 必須重新編碼。

匯出有版本與共用設定的 `.bhxs` 串流，或重建某筆完整 JPEG XS：

```powershell
.\bin\dataset-export.exe -input "拆好的訓練資料\video001.sqlite" -output "video001.bhxs" -with-decoded=false
.\bin\dataset-export.exe -mode codestream -input "拆好的訓練資料\video001.sqlite" -frame 0 -block 0 -output "sample.jxs"
```

目前匯出器預設 `-with-decoded=true`，會把每筆 640-byte decoded 也放入 BHXS v2；頻寬估算用的 payload-only 匯出須明確加 `-with-decoded=false`。本次 Q=4 `.bhxs` 使用 false，約 1.805 Mbps 的主體數字不含 decoded 傳輸。

BHXS 逐區塊 writer/reader 不等待整幀到齊。現有工具提供檔案匯出與 Go 介面，網路傳輸服務尚未實作。

## 驗證範圍

測試涵蓋 16 相的空間位置、RGGB 保留與完整取樣覆蓋、補／刪幀時間對應、SQLite 整幀回滾與續跑、編碼長度與標記檢查。整合測試用 FFmpeg 生成短片，搭配明確標示的測試用編碼器替身驗證資料流；它不驗證真實 JPEG XS 壓縮效果，也不產生正式訓練資料。

`internal/jpegxs` 套件另外驗證 Go 編碼器本身，分成三個層級：

| 層級 | 測試 | 內容 |
| --- | --- | --- |
| 對**標準** | `TestForwardDWTMatchesTableE8`、`TestForwardDWTMatchesTableE8Streaming`、`TestForwardDWTBandKeyMatchesStandardBeta` | 把 ISO/IEC 21122-1:2022 Annex E（Table E.5／E.10／E.11／E.12／E.8）與 Annex B.3 的**逐字轉錄**寫成 Go（獨立於任何實作），直接比對編碼器的小波係數與濾波型別編號。這是對標準的驗證，不是對另一份實作的驗證 |
| 對**上游實作** | `TestGoldenVectors`、`TestParityAgainstRustBridge`、`TestConformantStreamsRoundTripThroughUpstreamDecoder` | 7 種測試圖樣 × 10 種 levels（NL,x 1..5 / NL,y 0..1）× Q 0..15，共 1120 組。`TestGoldenVectors` 比對釘選在 `internal/jpegxs/testdata/golden.json` 的 SHA-256，不需要 Rust toolchain；`TestParityAgainstRustBridge` 即時與 Rust 轉接器逐 byte 比對（需 `JPEgXS_BRIDGE`）；`TestConformantStreamsRoundTripThroughUpstreamDecoder` 把全部 1120 組交給上游 Rust 解碼器，Q=0 時無損往返必須完全等於原圖 |
| 對**第三方解碼器** | `TestConformantStreamsDecodeWithThirdPartyDecoder`、`TestNonConformantBrIsRejectedByThirdPartyDecoder` | 用 FFmpeg 的 libsvtjpegxs（SVT-AV1 的 JPEG XS 解碼器）解碼。第三方解碼器的重建結果必須與上游 Rust 解碼器逐位元組相同；Q=0 時必須完全等於原始樣本 |
| 對**上游解碼器** | `TestDecodeMatchesRustBridge`、`TestDecodedByUpstreamDecoder` | Go 原生解碼器把同一條碼流解出來的 640 bytes，必須與上游 Rust 解碼器**逐 byte 相同**。涵蓋 Q=0/1/4/6/15 × Br=4/8 × 3 組種子；`TestDecodedByUpstreamDecoder` 另外把整個正式資料庫的樣本交給上游解碼器重建並比對。Q=0 時必須完全等於原始 RGGB |

### 實機一致性驗證工具（`cmd/jpegxs-parity`）

`cmd/jpegxs-parity` 是獨立於 `go test` 的黑箱驗證工具：把同一個 160 × 4 RAW8 RGGB 區塊、同一組設定，分別交給行程內 Go 編碼器與 Rust 轉接器，比較兩者產生的**完整碼流是否逐 byte 相同**。工具兩邊不共用任何程式碼，比對的是實際位元組。

```powershell
go build -o bin/jpegxs-parity.exe ./cmd/jpegxs-parity

# 全量化掃描：63 個樣本 × 160 種設定 = 10,080 組
.\bin\jpegxs-parity.exe -bridge .\bin\brainhole-jpegxs.exe `
  -db "拆好的訓練資料\big-buck-bunny-30s-q6-shared-v3-raw8.sqlite" `
  -videos "原始影片\big-buck-bunny-1080p-30sec.mp4" `
  -workers 16 -out "驗證報告"

# 正式設定 Q=6 的大規模驗證：10,000 個樣本 × 10 種 levels = 100,000 組
.\bin\jpegxs-parity.exe -bridge .\bin\brainhole-jpegxs.exe -blocks 10000 -q 6 -cases 100000 `
  -db "拆好的訓練資料\big-buck-bunny-30s-q6-shared-v3-raw8.sqlite" `
  -videos "原始影片\big-buck-bunny-1080p-30sec.mp4,原始影片\tears_of_steel.mp4" `
  -workers 16 -out "驗證報告"
```

| 參數 | 意義 |
| --- | --- |
| `-bridge` | Rust 轉接器執行檔路徑（必要） |
| `-db` | 已完成的訓練資料庫，抽出真實輸入與真實出貨碼流做驗證 |
| `-videos` | 影片來源（逗號分隔），抽出真實 RGGB 區塊；多支影片會平分樣本數 |
| `-blocks` | corpus 的區塊總數，預設 63 |
| `-q` | 要測的量化值：`all`（0..15）或清單如 `6` |
| `-cases` | 最少要比對的組數，預設 10000 |
| `-limit` | 只跑前 N 組，僅供冒煙測試 |
| `-workers` | 平行工作數 |
| `-ffmpeg` / `-ffprobe` | 執行檔路徑；ffprobe 用來計算取樣步長，讓樣本均勻分布在整部影片 |
| `-out` | 報告輸出目錄（JSON 與 Markdown） |

測試樣本分三類：合成區塊（上游測試圖樣、結構化邊界案例、兩種偽隨機產生器）、自正式資料庫還原的真實輸入、自影片抽取的真實 RGGB 區塊。影片取樣先用 ffprobe 取得片長再算步長，讓樣本均勻分布在整部影片，而不是集中在開頭；多支影片會平分樣本數。

已執行過兩種規模的驗證：

| 驗證 | 樣本 | 設定 | 組數 | 結果 |
| --- | ---: | ---: | ---: | --- |
| 全量化掃描 | 63 | 160（Q 0..15） | 10,080 | 全部逐 byte 相同 |
| 正式設定 Q=6 | 10,000 | 10（NL,x 1..5 × NL,y 0..1） | **100,000** | 全部逐 byte 相同 |

10 萬組的樣本組成為 44 個合成、10 個資料庫還原、4,973 個 Big Buck Bunny、4,973 個 Tears of Steel（9,843 個有內容、157 個均勻）。每一組都另外交給上游 Rust 解碼器解碼確認可解。兩端全 corpus 的 SHA-256 摘要一致，重跑結果相同。

工具另外執行**負向對照組**：刻意製造必須失敗的比較（Br=4 對 Br=8、翻轉 1 byte、換輸入、換 Q），確認比對機制確實抓得到差異。對照組沒過，比對結果一律視為無效。

報告含逐 byte 相同組數、兩端全 corpus 的 SHA-256 摘要、負向對照結果、第三方解碼器檢查、正式資料庫重現結果、碼流長度分佈，以及「不同輸入卻產生相同碼流」的等價清單。檔名帶有測試條件（例如 `jpegxs-parity-q6-10000blocks-10cfg-*.md`），不同設定的報告不會混淆。

### Br 的標準符合性（重要）

上游 Rust 編碼器寫的 `Br`（raw bitplane-count 寬度）是 **8**，但 ISO/IEC 21122-1:2022 **Table A.7 只列一個合法值：4**（與 `Ng = 4`、`Ss = 8` 同列）。上游的解碼器不驗證這個欄位，所以自己往返沒問題；但 **libsvtjpegxs 會直接拒收**：

```
Picture header invalid Br expected 4, read=8!
```

換句話說，上游產生的碼流**不符合標準、也無法被第三方解碼器解碼**。上游自己的黑盒測試 `black_box_decoder_agrees_with_this_decoder` 會「通過」，因為它對每個被拒收的碼流都跳過，實際上一個都沒比到。

Go 版編碼器預設 `Br = 4`（唯一符合標準的值），並保留 `Br = 8` 供需要與上游逐 byte 相同時使用。`encoder.native.json` 目前用 `Br = 4`。

**兩種 Br 的取捨**

- `Br = 4`：符合標準、第三方解碼器可解。這是 `encoder.native.json` 的現行設定。
- `Br = 8`：與 Rust 轉接器逐 byte 相同，但第三方解碼器會拒收。

兩者只在 picture header 的一個位元組欄位不同，但**整個資料集的位元組都會變**（共享標頭定義也會變），不能混用同一個資料庫。

另有 `TestUpdateGolden`：需設定 `JPEgXS_UPDATE_GOLDEN=1` 與 `JPEgXS_BRIDGE`，重新產生釘選向量。

另有 `TestRealJPEGXSPipeline`：設定 `BRAINHOLE_JPEGXS_CONFIG` 為原生 Go 設定檔，或使用絕對執行檔路徑的 Rust 設定檔。`BRAINHOLE_REAL_FRAMES=17` 可測全部 16 相與回繞。兩種編碼器皆已通過 17 幀、510 筆的 SQLite／BHXS 碼流重建；Rust 路徑另用上游解碼器驗證首尾幀。測試使用暫存資料，不混入正式資料集。

相關文件：[FFmpeg rawvideo](https://ffmpeg.org/ffmpeg-formats.html#rawvideo)、[modernc SQLite](https://pkg.go.dev/modernc.org/sqlite)、[JPEG XS 碼流結構](https://www.rfc-editor.org/rfc/rfc9134.html#section-3.2)。
