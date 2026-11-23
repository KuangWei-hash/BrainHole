# v14／v12 在 demo.mp4 的驗證與切塊壓縮碼本

2026-11-23，依使用者要求，拿一支真實錄製的 `training-data-preparation/原始影片/demo.mp4`（1280×720、60 FPS、61.75 秒 H.264）取代先前只用 Big Buck Bunny（BBB）動畫測試片的評估，檢驗 v12／v14 的假設（固定鏡頭場景會重複）在真實素材上是否成立。資料集用既有 Go 原生 JPEG XS 管線以 `encoder.native.q5.json`（Q=5）轉出，輸出 `training-data-preparation/拆好的訓練資料/demo-q5-native.sqlite`，640×480、30 FPS、共 1,853 幀（0–1852）、55,590 條（每幀 30 條）。場景偵測（[scenes.py](v11/scenes.py)）確認全片無場次切換。[轉換紀錄](training-data-preparation/README.md)、[場景偵測自動紀錄](experiments/records/diagnostic--demo-scene-plan--run-338984771acd.json)。

## v12 對照：demo.mp4 遠比 BBB 吃到時間混合的紅利

用既有 [evaluate_v12.py](evaluate_v12.py)（預設相位保護、decay=2）在 frame 1748–1852（含 16 幀暖機，實際計分 1763–1852 共 90 幀、2,700 條）評估：

| 方法 | RAW MAE | RGB MAE | PSNR | 連續幀殘差 |
| --- | ---: | ---: | ---: | ---: |
| bayer-nearest（單幀基準） | 12.7612 | 12.1771 | 19.594 dB | 11.4708 |
| v12（16幀相位保護混合） | 7.5194 | 6.8105 | 26.068 dB | 1.3856 |

v12 勝率 93.04%（2,700 條中贏 2,513 條），改善 **41.1%**。對照既有 BBB 紀錄（795–899 幀）bayer-nearest 約 14.96、v12 為 13.8121，改善僅 **7.7%**。demo.mp4 的連續幀殘差被壓到 BBB 數量級以下，說明這支素材遠比 BBB 靜態，時間混合的效益被放大了近 5 倍。[自動紀錄](experiments/records/evaluation--v12-demo-1748-1852--run-26f6caa6d927.json)。

## 整條 RAW 查表：有潛力，但「展開到全部相位」會失真

### Oracle 上限（初步探測，未接入正式紀錄系統）

用一支臨時腳本（未納入 `run_logged`，僅供快速驗證，不在本紀錄系統留存）以 300 幀（0–1389 取樣）建表、對 200 幀全新晚期畫面（1653–1852）做「拿真實完整 RAW 去表裡找最像的條目」測試：oracle 最佳命中 RAW MAE **9.1418**（147/6,000 逐 byte 命中），對照單幀最近色塊填補基準 **12.7517**，oracle 贏 **28.3%**。這與 BBB 上同類測試「查表反而全面輸基準」的結論相反，顯示 demo.mp4 的畫面重複性確實存在、值得往下驗證。

### 嚴謹的零誤差驗收（整部影片）

用 [check_v14_exact_video.py](check_v14_exact_video.py)（generic，無需 scene-plan）對全片 1,853 幀、55,590 條做兩種 key 規則的往返測試：

| 模式 | 影格全對 | 條帶精確命中 | RAW MAE |
| --- | --- | --- | ---: |
| recorded_phase_keys（每條只用它實際拍到的那個相位當 key，BBB 對應的零誤差規則） | 1,166/1,853（62.9%） | 54,404/55,590（97.87%） | 0.000159 |
| all_reference_phases（展開成全部16相位當 key） | 505/1,853（27.2%） | 52,708/55,590（94.82%） | 0.000408 |

BBB 用同一套 recorded_phase_keys 規則可達**完全**零誤差（900/900 幀、27,000/27,000 條）；demo.mp4 **沒有**達到零誤差，recorded_phase_keys 仍有 1,186 條（2.13%）因「同一相位、同一位置，160×4 樣本逐 byte 相同，但完整 640×16 畫面不同」而誤判。用 [export_v14_exact_lookup_video.py](export_v14_exact_lookup_video.py) 把全片 1,853 幀的 target／查表還原／誤差放大 10 倍三支影片都輸出後確認，這 1,186 條錯誤高度集中在 **frame 1500–1570（約第 50 秒）、block 22–23** 這一小段時空窗，不是隨機雜訊，應是真實發生的局部內容變化。影片見 `runs/v14-exact-lookup-video/{target,recorded-phase-lookup,error-x10}.mp4`（此腳本目前未接入 `run_logged`，無自動紀錄 JSON；下方的 `diagnose_v14_codebook.py` 已接上）。

### 單相 160×4 索引查表：驗證為不可行路線

用 [evaluate_v14_current_phase_demo.py](evaluate_v14_current_phase_demo.py)（44 幀建表 1400–1443、105 幀測試 1748–1852）複製 BBB 既有做法：

| 方法 | RAW MAE |
| --- | ---: |
| 單相最近鄰（乾淨 RAW 取樣基準） | 8.3689 |
| 單相 160×4 索引查表 | 13.0126（**-55%，輸基準**） |
| 同相位內 oracle 上限 | 12.9462 |

與 BBB（32.80 vs 14.96）同樣輸給零成本基準，確認「只用 160×4 當 key、只在同相位 1/16 子集裡找」訊號量太弱，不是可行方向。[自動紀錄](experiments/records/evaluation--v14-demo-current-phase--run-c362141da6ee.json)（同一輸出目錄下較早一次因程式仍沿用「複製零誤差版本」的強制 raise 中斷，[失敗紀錄](experiments/records/evaluation--v14-demo-current-phase--run-34a85cf5facf.json) 保留；修正後才確認這支腳本實際算的是「展開版」而非 recorded_phase_keys 版）。

### 整條 RAW codebook 的實際去重率：真正的重用率只有 0.80%

用 [train_v14_demo_codebook.py](train_v14_demo_codebook.py) 以 recorded_phase_keys 規則建立全片去重 codebook：55,590 條裡僅 **53,962 條不重複（去重率 2.93%）**。其中真正命中「過去已存在的相同內容」且內容正確的只有 **442 條（0.80%）**，1,186 條是誤判(同上)，其餘 53,962 條（97.07%）的「精確命中」其實只是查到自己、沒有借用任何歷史資料。整條 RAW 存法天真儲存需 569.24 MB，去重後連同 key 一起存反而是 587.11 MB（因為 key 本身的存儲開銷）。[自動紀錄](experiments/records/training--v14-demo-codebook--run-228e76839b26.json)、[codebook 檔](runs/v14-demo-codebook/codebook.pt)。

結論：**整條 RAW 查表的真實可用重用率不到 1%**，跟 BBB 的失敗模式不同（BBB 是查到不相關內容；demo.mp4 是幾乎查不到真正重複內容），但一樣不構成可部署的方案。

## 切塊 + DCT 殘差壓縮碼本：目前最有價值的方向

沿用既有 [train_v14_compact_codebook.py](train_v14_compact_codebook.py) 的通用數學函式（`describe`／`dct_basis`／`phase_data`／`fit_round`／`pack`／`retrieve`／`reconstruct`），另寫 [train_v14_demo_compact_codebook.py](train_v14_demo_compact_codebook.py) 為 demo.mp4 做獨立的建表／測試流程（訓練窗 0–1389、測試窗 1748–1852，不依賴 BBB 的 REFERENCE/DEV/AUDIT 切分）。每條 640×16 切成數個小塊，每塊只存 **16 維描述子**（key，亮度與梯度統計）+ **4×RANK 個 int8 量化 DCT 殘差係數**（value，殘差基準為相位對齊 bilinear），每個相位×位置×小塊學 k 個代表原型（5 輪疊代收斂）。

### RANK 掃描（width=40，16 塊／條）

| RANK | RAW MAE（k=8 最佳） | 改善 | codebook 大小 |
| ---: | ---: | ---: | ---: |
| 基準 bilinear | 7.8828 | — | — |
| 8 | 7.1144 | +9.7% | 2.95 MB |
| 32 | 5.7204 | +27.4% | 8.85 MB |
| 64 | 4.7840 | +39.3% | 16.72 MB |
| 128 | 4.0437 | +48.7% | 32.45 MB |
| 160（滿，無截斷） | 3.9544 | +49.8% | 40.31 MB |

### 低／高頻拆解（[diagnose_v14_codebook.py](diagnose_v14_codebook.py)，已接入自動紀錄）

| RANK | 低頻 MAE | 高頻 MAE | 梯度X／Y MAE | 精確命中率 | 門檻覆蓋率 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 基準 bilinear | 5.4797 | 6.0450 | 5.89／5.45 | 50.51% | — |
| 8 | 3.6952（-32.6%） | 5.9098（-2.2%） | 6.17／5.31（水平**變差**） | 41.23%（**變差**） | — |
| 32 | 2.9843 | 5.0069（-17.3%） | 5.78／4.47 | 42.36% | — |
| 64 | 2.7638 | 4.0594（-32.9%） | 5.00／3.76 | 45.30% | 96.89% |
| 128 | 2.5520 | 3.1304（-48.2%） | 3.63／3.38 | **51.96%（首次贏基準）** | 96.89% |
| 160 | 2.5430 | 3.0384（-49.7%） | 3.39／3.22 | 53.24% | 96.89% |

RANK=8 的改善幾乎全部來自低頻修正，高頻與水平梯度反而不變或變差，精確命中率明顯低於基準——是數值上過關、視覺上不及格的假改善。RANK 提高後各項指標同步改善，128 起精確命中率開始贏過零成本基準，顯示紋理細節是真的被找回來，不再是用平滑換 MAE。

上表原先由 `probe_v14_compact_frequency.py` 產出，該腳本已與 [碼本誤差分解](實驗紀錄-v14-碼本誤差分解.md) 的 `diagnose_v14_matching_limit.py` 合併為單一入口 `diagnose_v14_codebook.py`，數值不變（一致性驗證見該紀錄）。

### width 對照（width=16／40，各自吃滿理論最大 RANK）

| 設定 | RAW MAE | 改善 | codebook | 改善/MB | 門檻覆蓋率 | 每條查詢塊數 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| width=40，RANK=160（滿） | 3.9544 | 49.8% | 40.31 MB | **1.24%/MB** | **96.89%** | 16 |
| width=16，RANK=64（滿） | 3.8908 | 50.6% | 41.78 MB | 1.21%/MB | 93.22% | 40（2.5倍） |

兩者收斂到幾乎同一天花板，差距在 1–2% 內。width=16 的 codebook 更大、覆蓋率更低（代表更多小塊因找不到夠近的原型而退回純 bilinear）、且推論時查詢量是 2.5 倍，整體評估**不划算**。目前凍結的選擇是 **width=40、RANK=160、k=8**，codebook 40.31 MB（僅為整條 RAW 查表 587 MB 的 6.9%）。

## encoded payload 當查表 key：精確比對可行，近似比對不可行

用真正的 JPEG XS encoded payload bytes（`jpegxs_payload`，平均 178.84 bytes）取代 160×4 decoded 像素樣本做 key，重跑全片零誤差驗收（[check_v14_exact_video_encoded.py](check_v14_exact_video_encoded.py)）：

| Key 來源 | 全對影格 | 精確命中條帶 | RAW MAE | 去重後 key 儲存量 |
| --- | --- | --- | ---: | ---: |
| `decoded`（實際訊號，固定 640 bytes） | 186/1853 | 50,873/55,590（91.52%） | 0.092041 | 32.52 MB |
| `jpegxs_payload`（encoded，平均179 bytes） | 190/1853 | 50,942/55,590（91.63%） | 0.091786 | **9.56 MB** |

準確率幾乎相同（符合預期：解碼是確定性函式），但 key 儲存量省 71%。**同時也發現並修正了另一個方法論漏洞**：先前 `check_v14_exact_video.py` 的「零誤差驗收」用 `sample()` 直接從乾淨原始 RAW 標籤反推理想樣本，不是接收端實際會有的 Q=5 有損訊號；改用真實訊號後，recorded_phase_keys 的精確命中率從 97.87% 降到 91.63%，RAW MAE 從 0.000159 升到 0.0918——demo.mp4 並未像先前以為的達成零誤差。真實訊號版的查表影片見 `runs/v14-exact-lookup-video-real/`（[export_v14_exact_lookup_video_real.py](export_v14_exact_lookup_video_real.py)），錯誤分布從「集中在單一時空窗」變成「貫穿全片、集中在特定位置（如 block 0）」，顯示多出來的撞車主要是 Q=5 量化本身造成，不是場景變化。

接著嘗試把 encoded bytes 當作切塊壓縮碼本的**近似搜尋 key**（取代 16 維像素描述子），結果是負面的：[train_v14_demo_encoded_key_compact.py](train_v14_demo_encoded_key_compact.py) 同樣 width=40、RANK=160、k=8 設定下，RAW MAE 僅 7.3531（+6.7%），遠不如像素描述子版的 +49.8%。原因是熵編碼後的 byte 距離跟視覺相似度脫鉤（量到的距離 mean=41.6，遠超過沿用的門檻上限 24），近似搜尋找不到好候選。**結論：encoded bytes 適合精確比對，不適合近似相似度搜尋；近似搜尋仍須用像素描述子。**

## 重大修正：compact codebook 先前也用了理想化訊號

檢查 `train_v14_compact_codebook.py` 的 `phase_data()` 才發現，`current = sample(raw, phase)` 是直接從乾淨的 `raw_bayer_target` 反推樣本，從未讀取過資料庫裡真正的 `decoded` 欄位——也就是說，前面整段 RANK 8→160、width 16 vs 40 的掃描，訓練和測試用的都是理想化、無雜訊的輸入，不是真實會部署的訊號。

用 [train_v14_demo_compact_codebook_real.py](train_v14_demo_compact_codebook_real.py)（修正版：訓練與測試全程讀取真實 `decoded` 欄位，殘差目標仍對真實 RAW 標籤）重跑 width=40、RANK=160：

| | 理想化訊號（錯誤） | 真實 decoded 訊號（修正後） |
| --- | ---: | ---: |
| bilinear 基準 | 7.8828 | **11.9130** |
| compact k=8 | 3.9544 | **6.3597** |
| 改善幅度 | +49.8% | **+46.6%** |

好消息：改善**比例**幾乎沒變（49.8%→46.6%），表示 compact codebook 方法本身是真的有效，不是在修雜訊的假象；只是絕對誤差值因為真實雜訊全面墊高了。修正後的 codebook 存於 `runs/v14-demo-compact-codebook-real-rank160/`。

## 疊加歷史幀：v12 式保護會拖累，使用者設計的簡化版才有效

**第一次嘗試**：把 compact codebook 的單幀輸出當作 v12 `phase_protected_blend()` 的「enlarged」輸入（16 幀、依年齡衰減保護），[export_v14_compact_plus_v12_blend.py](export_v14_compact_plus_v12_blend.py)。用修正後的真實訊號 codebook 重測：

| 方法 | RAW MAE |
| --- | ---: |
| compact-alone（單幀） | 6.3671 |
| v12（nearest-bayer + 16幀衰減保護） | 7.5194 |
| compact + v12 完整保護機制 | 6.7813（比單幀 compact 還差） |

v12 的衰減保護機制會把過去 15 幀「各自量到的相位位置」都保護起來，但這些舊保護位置可能已經跟當前畫面有落差（demo.mp4 已知有持續微小漂移），疊加後反而拖累已經夠準的單幀估計。

**使用者指定的簡化設計**：當前幀相位對應的位置完全不混合（直接用 compact 當前輸出），其餘位置只跟「上一幀的最終輸出」做遞迴混合（不是跟 16 幀都混，也不是依年齡衰減）。[export_v14_simple_phase_blend.py](export_v14_simple_phase_blend.py) 實作並對混合比例掃描：

| 當前幀權重（其餘給上一幀輸出） | RAW MAE |
| --- | ---: |
| 50% | 5.5921 |
| 40% | 5.6102 |
| **30%（最佳）** | **5.5386** |
| 20% | 5.5866 |
| 10% | 5.9234（明顯變差） |

20%–40% 是平緩谷底，30% 最佳；低於 20% 權重會快速惡化（歷史拖得太重，跟不上真實的漸進變化）。**目前最佳設定：compact codebook（width=40、RANK=160、k=8、真實訊號訓練）+ 當前相位保護 + 30% 當前／70% 上一幀輸出遞迴混合**，RAW MAE **5.5386**，比真實 bilinear 基準（11.9130）贏 **53.5%**，比 v12（7.5194）贏 **26.3%**——是目前整個 demo.mp4 系列裡所有查表／codebook 方法中最好的結果。影片見 `runs/v14-simple-phase-blend-w30/`（`target.mp4`／`compact-alone.mp4`／`v12.mp4`／`compact+simple-blend.mp4`）。

## 肉眼比對 v12 vs v14，以及 v15：數值最佳不等於觀感最佳

2026-11-23，使用者肉眼比對這批影片後指出：**v12 的缺點是快速移動物體因時間混合延遲而扭曲，優點是靜止時紋路最清晰；v14（compact+30/70混合）的優點是外型完整不扭曲，缺點是紋路明顯不如 v12——即使 v14 的 RAW MAE 較低**。機制上站得住腳：v12 的紋路來自真實像素的時間累積，對真正靜止的區域理論上可以無限逼近真相；v14 的紋路來自 k=8 聚類原型＋截斷 DCT 殘差，即使 RANK 開到滿，存的仍是壓縮近似值，天生有精細度上限。這與 v13 既有設計文件記錄的「v10 數值贏但 v12 材質更清楚」是同一種現象，也呼應 `驗收標準.md`：數值初篩通過不代表視覺驗收通過。

v15（[v15/README.md](v15/README.md)）：v12、v14 各自獨立計算完整畫面，再 50/50 平均。結果：RAW MAE 5.9858，介於 v12（7.5194）與 v14（5.5386）之間，比 v14 單獨差——符合「平均一準一不準，不會超越較準者」的預期。**但使用者目視後確認 v15 是肉眼最舒服的版本，即使分數沒有超越 v14**，目前是示範片段上實際採用的候選。影片見 `runs/v15-v12-v14-blend/`。

## 統一診斷入口：低／高頻拆解與誤差分解合併

原先分屬兩支腳本的兩個問題——「改善是來自高頻紋理找回，還是只是低頻平滑」與「誤差由匹配、覆蓋還是表示能力造成」——都作用在同一個數字（compact codebook 的 RAW MAE）上，因此合併為單一入口 [diagnose_v14_codebook.py](diagnose_v14_codebook.py)，一次跑完兩種分解。`--signal ideal` 用乾淨 RAW 反推的理想化樣本，`--signal real` 用接收端實際會拿到的 `decoded` 欄位。

### demo.mp4 的誤差分解：瓶頸已從「表示能力」轉為「表項覆蓋」

在 demo.mp4 上重跑同一套分解（測試窗 1748–1852，105 幀、3,150 條）：

| 設定 | 描述／匹配 | 表項覆蓋 | 殘差表示 | 合計 | DCT 保留能量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BBB，width=40／RANK=8 | 0.168 | 2.180 | **11.507** | 13.8551 | 37.6% |
| demo，width=40／RANK=160 | 0.147 | **3.648** | 0.159 | 3.9544 | 100% |
| demo，width=16／RANK=64 | 0.094 | **3.630** | 0.167 | 3.8908 | 100% |
| demo，width=40／RANK=160（真實訊號） | 0.253 | **5.500** | 0.606 | 6.3597 | 100% |

**BBB 的結論不能直接搬到 demo.mp4。** BBB 診斷時用的是 RANK=8 的碼本，每塊只留 8 個最低頻 DCT 模式，只保留 37.6% 的殘差能量，所以表示能力佔了 83.1% 的誤差。demo.mp4 的碼本已吃滿理論最大 RANK（64／160），DCT 保留能量 100%，表示能力只剩 4–9.5%——**這條路徑在 demo 上已經沒有表示能力的餘裕了**。

**剩下的誤差幾乎全是表項覆蓋。** demo 上三組設定的覆蓋項都佔 92.2%／93.3%／86.5%。這可以獨立驗證：若把查詢換成「真實殘差挑出的最佳表項」（`oracle-entry`），MAE 只改善 0.09–0.25；但若把「儲存值」換成理想值（`oracle-dct`），MAE 改善 3.7–5.8。也就是說**碼本存了什麼內容，比能不能找到對的表項重要得多**。

真實訊號版本更嚴重：覆蓋項從 3.648 升到 5.500，相對佔比 86.5%，而門檻覆蓋率仍是 97.1%——多數小塊「找得到表項」，但找到的表項內容與當下內容幾乎無關。這與先前 demo 紀錄裡「整條 RAW 查表真實重用率僅 0.80%」是同一個現象的兩種量測。

### 一致性驗證

合併後逐一比對兩支舊腳本留下的每一個數字，29 項全數通過：probe 腳本的 demo 低／高頻表 12 項（bilinear 7.8828／5.4797／6.0450／5.8934／5.4547／0.5051，compact 3.8908／2.5125／3.0089／3.3657／3.0921／0.5361，門檻覆蓋率 93.22%）、demo 兩份碼本的 bilinear／actual／覆蓋率、BBB 紀錄中的 7 個方法 MAE 與能量比例，以及用 MAE 值反推的 BBB 三項分解（0.168／2.180／11.507）。BBB 資料庫與碼本已不在本機（該輪在另一台機器執行），故 BBB 數字取自保存的實驗紀錄。

## 限制與下一步

- 本輪所有訓練／測試切分（訓練 0–1389、測試 1748–1852）都在同一支 61.75 秒的 demo.mp4 內，尚未在真正長時間、多場景的固定監視器錄影上驗證。
- `export_v12_demo_videos.py`、`export_v14_demo_video.py`、`export_v14_exact_lookup_video.py`、`export_v14_exact_lookup_video_real.py`、`export_v14_compact_plus_v12_blend.py`、`export_v14_simple_phase_blend.py` 目前未接入 `run_logged`，僅有本檔案與對話紀錄留存，不在 `experiments/records/` 自動索引內；之後若要正式留存應補上包裝。`check_v14_exact_video_encoded.py`、`train_v14_demo_encoded_key_compact.py`、`train_v14_demo_compact_codebook_real.py`、`diagnose_v14_codebook.py` 已接上。
- k 目前只測過 1／4／8；16 維描述子只是粗略的亮度＋梯度統計。兩者都還沒探過上限，是目前最可能繼續榨出改善的方向，待下一輪驗證。
- **混合比例掃描只做到 10% 一格的粗網格**（50/40/30/20/10%），30% 是目前網格內的最佳點，還沒在 25%／35% 附近細搜確認是否有更精確的甜蜜點。
- 即時串流效能（真正流式建表、查表延遲、記憶體隨時間成長）完全未測，目前都是離線、一次性建表後評估；「30/70 上一幀混合」的遞迴狀態量（每個位置都要存上一幀的最終輸出）在串流部署時的記憶體與延遲成本也未評估。
- 全片零誤差驗收找到的 1,186 條失敗集中在 frame 1500–1570、block 22–23，初步判定為真實內容變化而非雜訊，但未實際目視確認（已有 `error-x10.mp4` 可供檢視，尚待人工看過)。
- compact codebook 的訓練／測試訊號已修正為真實 `decoded`，但 `describe()` 描述子本身、DCT 殘差計算都還是沿用 BBB 既有數學，沒有針對真實雜訊特性重新設計。
- **統一診斷顯示 demo.mp4 的瓶頸是表項覆蓋（佔誤差 86.5–93.3%），不是表示能力也不是匹配。** 既然匹配只值 0.09–0.25 MAE，改描述子維度或匹配距離幾乎不會有幫助；真正要解的是「每個相位×位置×小塊只有 k=8 個原型不足以代表該場景會出現的內容」。k 只測過 1／4／8，16 維描述子只是粗略的亮度＋梯度統計，兩者都還沒探過上限，是目前最可能繼續榨出改善的方向，待下一輪驗證。
