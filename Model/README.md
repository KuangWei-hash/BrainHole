# 模型研究與操作

**2026-11-23，改用真實錄影 demo.mp4 驗證，並確立 v15 為目前目視採用候選。** 真實錄影（1280×720、60 FPS、61.75 秒，Q=5 原生 JPEG XS）上，v12 的時間混合改善達 41.1%（BBB 僅 7.7%）。v14 切塊＋DCT 殘差壓縮碼本（width=40、RANK=160、k=8）修正為用真實 `decoded` 訊號訓練後，RAW MAE 5.5386（compact＋當前相位保護＋30/70 遞迴混合），數值最佳。**但使用者目視判定 v15（v12＋v14 各 50/50 融合，RAW MAE 5.9858）最舒服**，因 v12 靠真實像素時間累積、紋理最清晰，v14 外型完整但紋路較軟。v12 仍是「堪用最低合格線」。跨片泛化、即時效能待驗。[v15 說明](v15/README.md) · [demo.mp4 完整紀錄](實驗紀錄-v14-demo影片驗證與壓縮碼本.md) · [v14 方向](v14/README.md)

**2026-11-23（BBB 階段），v14 完成 40 × 16 分塊特徵碼本。** 存放 16 維描述與 32 維 DCT 殘差係數，以單相 160 × 4 重建完整 640 × 16。BBB 完整 900 幀的碼本檔為 1.41 MiB，同片 RAW MAE 8.5646。先前 44 幀建表的未見後段中，最小分塊碼本為 0.36 MiB、RAW MAE 13.7436，單相雙線性為 13.7019。（此批 BBB 數值後來發現使用了理想化訊號，已在 demo.mp4 紀錄中修正。）塊寬改為 16 × 16 後容量增至 3.52 MiB、同片 MAE 8.4607，未建表後段仍落後雙線性，維持 40 × 16。誤差分解顯示描述與匹配只占 0.2 MAE，瓶頸是最低頻 DCT 模式的表示能力與表項覆蓋不足。實際固定監視器與即時效能待驗。[訓練與查詢入口](train_v14_compact_codebook.py) · [結果、碼本與影片](實驗紀錄-v14-分塊特徵碼本.md) · [塊尺度對照](實驗紀錄-v14-16x16分塊碼本.md) · [誤差分解](實驗紀錄-v14-碼本誤差分解.md)

## v12 視覺基準與 v13 同片結果

**2026-11-22，v13 R2 半強度的 S2 影片經使用者目視判定合格，與 v12 的差距沒有顯著拉開。** 同片保留區的 RAW MAE 下降 13.6%。跨片與即時效能待驗。[v13 文件入口](v13/README.md) · [S2 紀錄與影片](實驗紀錄-v13-S2.md)

**2026-11-23，v13 等權候選已凍結並完成同片歷史重測。** 時間趨勢與原生校正 64 通道模型以固定 v12 為底，在 BBB Q=5 795–899 共 3,150 條的 RAW MAE 為 10.5100，較原合格 v13 下降 12.0%。字面橫紋較淡，尚有草葉偏軟與標題淡出殘影。此候選需兩次前向。新組目視、跨片和即時效能待驗。[歷史重測與影片](實驗紀錄-v13-逐幀融合.md#最終凍結與歷史重測)

## v1 至 v10 歷史實驗摘要

**v10 第一版已實作，55項測試與短流程通過，Q=5正式二十輪訓練執行中**：輸入為實際decoded＋同份decoded依phase產生的bilinear放大圖，以本幀bilinear為底圖做條件式殘差擴散；沿用本幀／前幀本條，主流程不需要v9，不使用上一條或未來幀。實際280,340參數；每輪21,570條，以固定256條生成選模，完整5,370條與影片驗收另行執行。初期生成仍嚴重失真，沒有通過品質的模型；入口、指令與觀測見 [v10實作紀錄](./實驗紀錄-v10雙路殘差擴散Q5.md)，研究計畫見 [v10生成方針](./v10/生成方針.md)。

EchoSR-Lite ×4 官方 DF2K 預訓練權重已直接測試 Q=5 實際 decoded：當前本條 RAW 經既有相位轉 RGB、guard8 獨立補邊後推論，無微調、無前幀或鄰條。共同 frame 721…899、5,370 條 RAW 投影 MAE **17.194935**／PSNR **19.836919 dB**，與 FSR＋RCAS 的 **17.213109**／**19.842178 dB** 接近，仍明顯落後 v9 單幀／雙幀。沒有證明實質改善，目前保持 v9 雙幀候選；完整結果與限制見 [EchoSR-Lite 紀錄](./實驗紀錄-EchoSRLite直接decodedQ5.md)。

v7 Q=5 再次從頭完整五輪與 FSR 對照已完成，最佳第 5 輪。共同 frame 721…899、5,370 條：RAW MAE **15.103396**／PSNR **20.813891 dB**，相較先前 v7 的 MAE 15.121943 僅改善約 **0.12%**；比 FSR＋RCAS 的 MAE 改善 **12.26%**，仍落後 v9 雙幀的 **14.171117**。對 v9 逐條勝率 **3.28%**，目前保持 v9 雙幀為較佳候選。完整曲線、限制及產物見 [v7 再次重訓紀錄](./實驗紀錄-v7Q5再次重訓.md)。

**目前採用 v9 四層 relu128 雙幀版：本幀本條＋前幀本條，不使用上一條。** 使用設定 `--architecture v9 --history 2 --v9-layers 4 --v9-variant relu128`，權重為 [雙幀 best.pt](./runs/relu128-four-v9-history2-q5/best.pt)。依使用者決定，停止採用本幀上一條方案；不要加上 `--v9-upper-strip`。該選項預設關閉，僅保留歷史實驗重現能力。以下新增上一條的內容是已結束的實驗紀錄。

新增 v9 `--history 2 --v9-upper-strip` 實驗：在當前幀本條與前一幀本條之外，加入當前幀上一條，配合第一條缺席遮罩，共 13 個輸入通道。四層 relu128 為 314,884 參數，預設不開啟；設定與比較結果見 [雙幀加上一條紀錄](./實驗紀錄-v9雙幀加本幀上一條Q5.md)。

本次完整五輪及 FSR 比較完成，最佳第 5 輪。Q=5 共同 frame 721…899、5,370 條，新模型 RAW MAE **14.296203**／PSNR **21.158660 dB**，相較原雙幀 **14.171117**／**21.282882 dB**，MAE 退步 **0.88%**，僅 **49.89%** 區塊較佳。第一條明顯退步，其餘條帶平均也略差；目前保留原雙幀為較佳候選，負面紀錄與新權重保留。47 項測試通過，視覺／新影片／效能驗收仍待完成。

v9 現支援 `--history 2`：當前幀與前一幀本條 decoded，各自依 phase 放大後將 CFA 四平面串接成 8 通道，由 CNN 學習融合；輸出仍以當前幀 bilinear 為殘差底圖。預設保持 `--history 1`，既有單幀權重不變。雙幀四層 relu128 的 Q=5 實驗與相同窗口對照見 [雙幀實驗紀錄](./實驗紀錄-v9雙幀四層128通道Q5.md)。

雙幀完整五輪完成，最佳第 5 輪。在共同 frame 721…899、5,370 條上，雙幀 RAW MAE **14.171117**／PSNR **21.282882 dB**，單幀為 **15.241226**／**20.544308 dB**；MAE 改善 **7.02%**，**98.55%** 區塊較佳。參數增加約 1.51%。仍可見條帶與細節流失，完整影片視覺、新影片泛化與即時效能待驗收。

**本次測試 v9 五層 relu256**：五層 3×3 `4→256→256→256→256→4`，前四層後各有 ReLU，1,788,932 參數／FP32 7,155,728 bytes。指定 `--v9-layers 5 --v9-variant relu256 --v9-kernel-width 3`，省略選項仍預設兩層 relu16。44 項測試通過，Q=5 五輪及 FSR 結果見 [五層 256 通道紀錄](./實驗紀錄-v9五層256通道ReLUQ5.md)。以下為先前實驗。

五層 256 通道完整五輪與評估完成，最佳第 3 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **15.567865**／PSNR **20.395050 dB**，比四層 256 的 MAE 退步 **0.35%**、比四層 128 退步 **2.19%**。RGB MAE 比四層 256 略好但 PSNR 較差，主要 RAW 指標未改善；42.87% 區塊勝過四層 256，四層 128 仍較佳。完整視覺、泛化與即時效能尚未驗收。

**本次測試 v9 四層 relu256**：3×3 卷積 `4→256→256→256→4`，前三層後各有 ReLU，1,198,852 參數／FP32 4,795,408 bytes。指定 `--v9-layers 4 --v9-variant relu256 --v9-kernel-width 3`，省略選項仍預設兩層 relu16。43 項測試通過，Q=5 五輪及 FSR 結果見 [四層 256 通道紀錄](./實驗紀錄-v9四層256通道ReLUQ5.md)。以下為先前實驗。

四層 256 通道完整五輪與評估完成，最佳第 3 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **15.513671**／PSNR **20.484319 dB**，比 128 通道 MAE 退步 **1.84%**，RGB 指標也退步；只有 16.81% 區塊勝過 128 通道。參數約 3.94 倍，本次不建議替換 128 通道，所有產物保留。完整視覺、新影片與效能尚未驗收。

**本次測試 v9 四層 relu128**：3×3 卷積 `4→128→128→128→4`，前三層後各有 ReLU，304,516 參數／FP32 1,218,064 bytes。指定 `--v9-layers 4 --v9-variant relu128 --v9-kernel-width 3`，省略選項仍預設兩層 relu16。42 項測試通過，Q=5 五輪及 FSR 比較見 [四層 128 通道紀錄](./實驗紀錄-v9四層128通道ReLUQ5.md)。以下為先前實驗。

四層 128 通道完整五輪與評估完成，最佳第 4 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **15.234079**／PSNR **20.551496 dB**，比 64 通道 MAE 改善 **2.02%**，74.24% 區塊更好；RGB 指標也改善。參數約 3.88 倍，對 FSR＋RCAS 勝率 99.30%；完整視覺、新影片泛化與效能尚未驗收。

**本次測試 v9 四層 relu64**：3×3 卷積 `4→64→64→64→4`，前三層後各有 ReLU，78,532 參數／FP32 314,128 bytes。指定 `--v9-layers 4 --v9-variant relu64 --v9-kernel-width 3`，省略選項仍預設兩層 relu16。41 項測試通過，Q=5 五輪與 FSR 比較見 [四層 64 通道紀錄](./實驗紀錄-v9四層64通道ReLUQ5.md)。以下為先前實驗。

四層 64 通道完整五輪與評估完成，最佳第 4 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **15.548606**／PSNR **20.430942 dB**，比 32 通道 MAE 改善 **1.16%**，RGB 指標也改善；71.83% 區塊勝過 32 通道。參數約 3.77 倍，完整視覺、新影片泛化與即時效能尚未驗收。

**本次測試 v9 四層 relu32**：3×3 卷積 `4→32→32→32→4`，前三層後各有 ReLU，20,836 參數／FP32 83,344 bytes。指定 `--v9-layers 4 --v9-variant relu32 --v9-kernel-width 3`，省略選項仍預設兩層 relu16。40 項測試通過，Q=5 五輪及 FSR 結果見 [四層 32 通道紀錄](./實驗紀錄-v9四層32通道ReLUQ5.md)。以下為先前實驗。

四層 32 通道完整五輪與評估完成，最佳第 3 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **15.731815**／PSNR **20.350482 dB**，比 24 通道 MAE 改善 **0.91%**，RAW PSNR 略降 0.004577 dB；RGB 指標改善，逐條勝率 67.5%。參數增加約 71%，完整視覺、新影片與效能尚未驗收。

**本次測試 v9 四層 relu24**：3×3 卷積 `4→24→24→24→4`，前三層後各有 ReLU，12,172 參數／FP32 48,688 bytes。指定 `--v9-layers 4 --v9-variant relu24 --v9-kernel-width 3`，省略選項仍為兩層 relu16。39 項測試通過，Q=5 五輪與 FSR 比較見 [四層 24 通道紀錄](./實驗紀錄-v9四層24通道ReLUQ5.md)。以下保留先前實驗。

完整五輪及評估完成，最佳第 5 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **15.875731**／PSNR **20.355058 dB**，比四層 16 通道 MAE 改善 **0.95%**，逐條勝率 63.56%；RGB 指標也改善。參數約 2.09 倍，未完成視覺、新影片與即時效能驗收。

**本次 v9 新增五層 relu16 實驗**：五層 3×3 `4→16→16→16→16→4`，前四層後各有 ReLU，共 8,132 參數／FP32 32,528 bytes。指定 `--v9-layers 5 --v9-variant relu16 --v9-kernel-width 3`；省略 layers 仍預設兩層。38 項測試通過，Q=5 完整五輪與 FSR 結果見 [五層紀錄](./實驗紀錄-v9五層16通道ReLUQ5.md)。以下保留先前實驗。

五層完整五輪與評估完成，最佳第 4 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **16.099018**／PSNR **20.217805 dB**，比四層 MAE 退步 **0.44%**，37.35% 區塊勝過四層。RGB MAE 極小改善但 PSNR 下降，依主要 RAW 指標仍以四層較佳；完整視覺、泛化與效能尚未驗收。

**本次 v9 再加深為四層 relu16**：3×3 卷積 `4→16→16→16→4`，前三層後各有 ReLU，共 5,812 參數／FP32 23,248 bytes。指定 `--v9-layers 4 --v9-variant relu16 --v9-kernel-width 3`；省略 layers 仍預設兩層。37 項測試通過，Q=5 完整五輪及 FSR 比較見 [四層紀錄](./實驗紀錄-v9四層16通道ReLUQ5.md)。以下為先前實驗。

四層完整五輪與評估完成，最佳第 5 輪。同 Q=5 frame 720…899、5,400 條 RAW MAE **16.028068**／PSNR **20.323953 dB**，比三層 MAE 改善 **1.33%**、比兩層改善 **1.80%**。65.81% 區塊比三層更好，FSR＋RCAS 勝率 91.83%；完整視覺、泛化與效能未驗收。

**本次新增 v9 三層 relu16 實驗**：`Conv3×3 4→16 → ReLU → Conv3×3 16→16 → ReLU → Conv3×3 16→4`，3,492 參數／FP32 13,968 bytes。指定 `--v9-layers 3 --v9-variant relu16 --v9-kernel-width 3`，省略 layers 仍預設兩層。36 項測試通過；Q=5 完整五輪及 FSR 結果見 [三層紀錄](./實驗紀錄-v9三層16通道ReLUQ5.md)。以下保留先前版本。

三層五輪與 FSR 完整評估完成，最佳第 3 輪；Q=5 frame 720…899、5,400 條 RAW MAE **16.244480**／PSNR **20.183361 dB**，比兩層 relu16 的 MAE 改善 **0.47%**，兩層／三層逐條比較有 61.31% 改善。參數近三倍，對 FSR＋RCAS 的逐條勝率降至 84.85%；平均改善不代表每條更好。完整視覺、泛化與效能尚未驗收。

**本次測試 v9 擴大水平視野**：`Conv3×9 4→16 → ReLU → Conv3×9 16→4`，3,476 參數／FP32 13,904 bytes，平面感受野 5×17。指定 `--v9-variant relu16 --v9-kernel-width 9`；省略 kernel width 仍使用舊版 3×3。35 項測試通過，Q=5 完整五輪及 FSR 對照見 [擴大視野紀錄](./實驗紀錄-v9擴大水平視野Q5.md)。以下為先前實驗，Q=4 範例不是此次 Q=5 成績。

本次五輪及完整 FSR 評估完成，最佳第 4 輪；Q=5 frame 720…899、5,400 條 RAW MAE **16.497208**／PSNR **20.080343 dB**，仍勝 FSR，但比原 3×3 relu16 的 MAE 退步約 **1.08%**。原版仍是較佳平均數值候選；擴大版可用 kernel width 9 選擇，所有權重與結果保留。

**目前 v9 新增並預設 relu16**：`Conv3×3 4→16 → ReLU → Conv3×3 16→4`，1,172 參數／FP32 4,688 bytes。33 項測試通過，Q=5 五輪實驗與 FSR 比較見 [16 通道 ReLU 紀錄](./實驗紀錄-v9擴張16通道ReLUQ5.md)。以下 plain32 等為先前版本，舊權重保留相容。

relu16 五輪與完整評估完成，最佳第 4 輪。Q=5 frame 720…899、5,400 條 RAW MAE **16.321658**／PSNR **20.152144 dB**，比 plain16 的 MAE 改善 **1.84%**，也勝 plain32 與 FSR；FSR＋RCAS 逐條 MAE 勝率 **88.24%**，不是每條都改善。視覺、泛化與串流效能尚未驗收。

**目前 v9 預設為無 ReLU 的 4→32→4**，兩層 3×3、2,340 參數／FP32 9,360 bytes，`--v9-layers 2 --v9-variant plain32`。32 項測試通過，Q=5 完整五輪實驗與 FSR 比較見 [32 通道紀錄](./實驗紀錄-v9無ReLU擴張32通道Q5.md)。下方 16 通道及其他版本敘述保留為歷史紀錄；Q=4 尚未重訓。

32 通道完整五輪與 FSR 評估完成，最佳第 3 輪；同 Q=5 frame 720…899、5,400 條 RAW MAE **16.684918**／PSNR **20.065681 dB**。仍優於 FSR，但比 plain16 的 MAE 16.626771 略差 **0.35%**，此次擴寬沒有改善。產物、曲線與限制見上述紀錄。

**v10 設計文件已完成**：[`v10/README.md`](./v10/README.md)。核心診斷是「v9 的卷積發生在高解析度網格上，每層感受野只有 ±2 個輸出像素，在 4× 超分下不可能學到東西」；v10 把主體搬到低解析度網格（每 CFA 平面 2×80），同時得到 16 倍算力與 4 倍感受野，並以**相位感知的學習式 ×4 上採樣**（子像素通道循環位移，零參數）取代固定 bilinear。文獻依據見 [`v10/文獻對照.md`](./v10/文獻對照.md)。v10 尚未實作、尚未訓練，所有品質預期都只是設計假設。

**關鍵事實：目前正式設定 Q=4 下，模型還沒贏過 0 參數的 bilinear 對照**（v7 Q=4 MAE 14.504695 vs bilinear 14.497267），而 Q=5 的 bilinear 是 17.552326——Q 越低、基準越強。v10 的第一個實驗就是驗證「低解析度主體」這個假設能否翻轉這個結果。

**目前 v9 的主要比較基準為官方 FSR 1**：兩次 EASU 2x＋RCAS sharpness=0，另報無 RCAS 結果；bilinear 保留為輔助對照。使用 `evaluate_fsr.py` 進行 FSR 比較，操作與既有 Q=5 完整結果見 [FSR 紀錄](./實驗紀錄-v9對照FSR1Q5.md)。下方歷史數值初篩依當時標準保留；Q=4／新影片必須另行同條件評估，不沿用 Q=5 分數。

**訓練資料已重作為 Q=4，並新增第二部來源影片**：Big Buck Bunny（train，27,000 筆）與 Tears of Steel（validation，660,720 筆），兩庫共用標頭 ID 相同，第一次可以用 `--val-db` 做**不同來源影片**的獨立驗證，而不是同影片的時間切分。尚未用 Q=4 重訓模型；既有模型分數仍屬 Q=5／Q=6 條件。見 [Q=4 重建紀錄](./實驗紀錄-Q4訓練資料重建.md)。

16 通道無 ReLU 版 Q=5 五輪與完整評估完成，最佳第 5 輪 `runs/plain16-v9-q5/best.pt`；同 5,400 窗口 MAE **16.626771**／PSNR **20.079337 dB**，比 8 通道的 16.743010 改善約 **0.69%**，數值初篩通過。見 [結果與影片](./實驗紀錄-v9無ReLU擴張16通道Q5.md)。完整視覺、泛化與串流驗收未完成。

**目前 v9 擴張至 16 通道，無 ReLU**：`Conv3×3 4→16 → Conv3×3 16→4`，1,172 參數／FP32 純參數 4,688 bytes，預設 `--v9-layers 2 --v9-variant plain16`。28 項測試通過，Q=5 五輪實驗見 [16 通道紀錄](./實驗紀錄-v9無ReLU擴張16通道Q5.md)。以下保留歷史實驗。

無 ReLU **4→8→4** 的 Q=5 五輪已完成，最佳第 2 輪，權重 `runs/plain8-v9-q5/best.pt`；同 5,400 窗口 MAE **16.743010**／PSNR **19.998957 dB**。相對無 ReLU 4→4→4 的 MAE 小幅改善約 **0.38%**，仍落後原同寬 ReLU 版的 16.646705。27 項測試通過。完整結果與媒體見 [實驗紀錄](./實驗紀錄-v9無ReLU擴張8通道Q5.md)；目前訓練預設 plain8。

**目前 v9 改測無 ReLU 的 4→8→4 擴張收縮**，兩層 3×3，588 參數／FP32 2,352 bytes，訓練預設 `--v9-layers 2 --v9-variant plain8`。維持 Q=5、單幀 decoded＋phase 與五輪預算；見 [本次紀錄](./實驗紀錄-v9無ReLU擴張8通道Q5.md)。以下 ReLU 與其他版本為歷史實驗。

**目前 v9 已加入 ReLU**：`Conv3×3 4→4 → ReLU → Conv3×3 4→4`，296 參數／FP32 1,184 bytes，預設 `--v9-layers 2 --v9-variant relu4`。Q=5 五輪最佳第 5 輪；同 5,400 窗口 MAE **17.290878**／PSNR **19.851339 dB**，相較無 ReLU 的 **16.806699／19.995103 dB** 退步，仍優於 bilinear。26 項測試通過，完整視覺／泛化／串流驗收未完成。見 [ReLU 紀錄](./實驗紀錄-v9兩層4通道ReLUQ5.md) 與 [影片](./runs/eval-v9-relu4-q5/comparison.mp4)。以下版本敘述為歷史紀錄。

**所有實驗均須記錄**：完整索引見 [實驗總表](./實驗總表.md)，保存方式見 [紀錄規則](./experiments/README.md)。既有實驗已補登，後續 CLI 自動保存設定、曲線、結果、程式／資料指紋與完成／失敗／中斷狀態。

**v9 已維持 4 通道再加一層：4→4→4**。兩層皆 3×3，中間沒有 ReLU，共 **296 參數，FP32 純參數 1,184 bytes**。Q=5 完整五輪最佳第 4 輪，權重為 `runs/plain4-v9-q5/best.pt`；同 5,400 窗口 MAE **16.806699**／PSNR **19.995103 dB**，bilinear 為 17.544584／19.708695 dB，數值初篩通過。單層 MAE 16.836493，本次只小幅改善 0.029794。預設 `--v9-layers 2 --v9-variant plain4`；舊版均保留。見 [紀錄](./實驗紀錄-v9兩層維持4通道Q5.md) 與 [影片](./runs/eval-v9-plain4-q5/comparison.mp4)。

先前單層 `Conv3×3 4→4` v9 為 148 參數，完整五輪最佳第 4 輪權重 `runs/linear-v9-q5/best.pt`；同 5,400 窗口 MAE **16.836493**／PSNR **19.949886 dB**，MAE 相對 bilinear 改善 **4.04%**。重現單層用 `--v9-layers 1`，重現原 588 參數雙層用 `--v9-layers 2 --v9-variant relu8`。見 [單層實驗](./實驗紀錄-v9單層148參數Q5.md)。

**v9 最簡模型已完成 Q=5 五輪訓練**：單幀本條 decoded＋phase → 固定 RGGB 相位對齊 bilinear 放大 → 兩層 `4→8→4` CNN 修正，共 **588 參數**。沒有 byte encoder、前幀、上一條或殘差區塊堆疊。最佳第 4 輪權重 `runs/simple-v9-q5/best.pt`，全 5,400 窗口 MAE **16.646705**／PSNR **20.014616 dB**，數值初篩通過。共同 5,370 窗口的 MAE 為 **16.654964**（bilinear 17.552326、v7 15.121942）。參數比 v7 少 99.49%，品質仍落後 v7；兩組窗口分開記錄。見 [v9 紀錄](./實驗紀錄-v9最簡模型Q5.md) 與 [對照影片](./runs/eval-v9-q5-comparison/comparison.mp4)。

**Q=5／schema v4 的 v7 五輪重訓與完整評估已完成**。最佳第 4 輪權重為 `runs/hybrid-v7-history2-q5/best.pt`；同 5,370 驗證窗口 MAE **15.121942**／PSNR **20.787027 dB**，Q=5 bilinear 為 **17.552326／19.701981 dB**。MAE 降低 **13.85%**，數值初篩通過，完整視覺／泛化／串流驗收仍待完成。見 [v7 Q=5 紀錄](./實驗紀錄-v7Q5重訓.md) 與 [對照影片](./runs/eval-hybrid-v7-q5/comparison.mp4)。資料與快取仍為 Q=5 schema v4 及 decoded-q5-v4.npz；Q=4／Q=6 分數保留為不同輸入條件的歷史結果。

**Q=4／schema v4 的 v7 五輪完整重訓已完成**：資料為 `training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite`，快取為 `runs/decoded-q4-v4.npz`。最佳第 1 輪權重 `runs/hybrid-v7-history2-q4/best.pt`，5,370 驗證窗口 MAE **14.504695**／PSNR **20.471827 dB**；Q=4 bilinear 為 **14.497267／20.471392 dB**，模型 MAE 略差，數值初篩未通過。詳見 [v7 Q=4 紀錄](./實驗紀錄-v7Q4重訓.md) 與 [對照影片](./runs/eval-hybrid-v7-q4/comparison.mp4)。先前兩步 smoke 保留為短流程，沒有用其權重續訓。下方 Q=6 歷史分數屬不同編碼輸入條件。

v7 品質對照採五個兩層 24→48→24 區塊、本條雙幀 encoded＋decoded；v9 是本次新的單幀 decoded 最簡基準。Q=4、Q=6 權重與上一條實驗均保留歷史比較。見 [作法](./作法.MD)。
先前新 v8 為五個兩層 24→96→24 殘差區塊、218,748 參數、12 層影像卷積，`v8_variant=expand96_two_layer`，詳見 [新 v8 紀錄](./實驗紀錄-v8通道96.md)。訓練預設仍為 **v7 的 24→48→24**，本次使用 Q=5 完整重訓。
新 v8 已完成五輪，最佳第 5 輪 **MAE 20.16／PSNR 18.87 dB**，同 5,370 窗口仍落後原 v7 的 **19.87／18.99 dB**。本次容量增加尚未改善數值品質，最佳候選仍是原 v7。
正式時間窗口為 **2 幀：當前為主、前一幀為輔**，兩幀各自都有 encoded 與 decoded。`train.py --history` 預設已改為 2；1 幀及 16 幀留作比較。
舊四層 v8 五輪最佳 **20.94／18.56 dB**，上一條實驗版五輪最佳 **20.13／18.96 dB**，均落後原 v7 的 **19.87／18.99 dB**（相同 5,370 窗口）。這些是歷史結果，不代表本次新 v8。上一條版已停用，詳見 [實驗紀錄](./實驗紀錄-v7上一條上下文.md)。
v7 擴張收縮版已完整訓練 5 輪，同一組 5,370 個窗口的 MAE **19.87**／PSNR **18.99 dB**；相較 v5 的 **20.50／18.82 dB**，MAE 降低 **3.09%**，它是本次四層區塊實驗的品質對照。參數增加 82.65%，畫面仍有條帶與模糊，時間誤差變化診斷也未改善；詳見 [v7 擴張收縮實驗紀錄](./實驗紀錄-v7擴張收縮.md) 與 [對照影片](./runs/eval-hybrid-v7-expand2/comparison.mp4)。
v6 六區塊已完成 5 輪，完整 5,370 個驗證窗口的 MAE **21.11**／PSNR **18.51 dB**，落後五區塊 v5 的 **20.50／18.82 dB**；該次增深未改善，當時保留 v5 作為較佳候選與後續對照。詳見 [六區塊實驗紀錄](./實驗紀錄-v6六殘差區塊.md)。
先前 v5 五區塊已完整訓練 5 輪。在同一組 5,370 個窗口上，MAE **20.50**／PSNR **18.82 dB**，四區塊 v4 為 **20.60／18.75 dB**；它作為六區塊實驗的對照。完整結果見 [v5 五區塊實驗紀錄](./實驗紀錄-v5五殘差區塊.md)。
舊 v4 雙幀版已完整訓練 5 輪。在共同 4,950 個窗口上，MAE **20.81**／PSNR **18.65 dB**；單幀 **21.23／18.48 dB**、十六幀 **20.71／18.87 dB**。結果與限制見 [雙幀實驗紀錄](./實驗紀錄-v4雙幀.md)。
學習目標是以已有畫面為基礎修復壓縮失真、補出高解析度細節，不再學習近似解碼。
現階段輸出是 **640 × 16 RAW8 RGGB**，不是 RGB。

品質判定以 [驗收標準.md](./驗收標準.md) 為準，必須勝過 JPEG XS 正常解碼後簡單放大。
舊 v1／v2 純 bytes 實驗見 [歷史實驗紀錄](./實驗紀錄-2026-11-21.md)：其結果不代表新 v3；下方保留舊實驗說明供追溯。
先前淺層 v3 單幀完整訓練 3 輪後，驗證 MAE **22.14**／PSNR **18.25 dB**，優於解碼放大 **28.76**／**16.68 dB**。它作為本次加深主幹的對照，詳見 [v3 實驗紀錄](./實驗紀錄-v3雙輸入.md)。
四殘差區塊 v4 同樣訓練 3 輪後，驗證 MAE **21.02**／PSNR **18.59 dB**，較 v3 改善；單區塊同步模型呼叫約 **2.47 ms**，不含解碼，尚未通過即時驗收。完整結果見 [v4 實驗紀錄](./實驗紀錄-v4四殘差區塊.md)。

## 環境

在儲存庫根目錄執行 PowerShell：

```powershell
python -m venv Model/.venv
.\Model\.venv\Scripts\python.exe -m pip install -r Model/requirements.txt --cache-dir .cache/pip
```

基準環境固定 PyTorch 2.7.1 CUDA 12.6 wheel，支援目前 Python 3.13 與 GTX 1650。
安裝來源：[PyTorch 官方版本清單](https://pytorch.org/get-started/previous-versions/)。
不需要 torchvision 或獨立安裝 CUDA Toolkit；GPU 可用性須以實際運算驗證。
已安裝環境的完整版本記錄於 `requirements-lock.txt`，重現時可用該檔取代 `requirements.txt`。

## 開始訓練

```powershell
$datasetPath = 'training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite'
$validationPath = 'training-data-preparation/拆好的訓練資料/tears-of-steel-q4-shared-v4-raw8.sqlite'
# 首次使用先產生解碼快取；若已存在就直接使用，不能覆寫
.\Model\.venv\Scripts\python.exe Model/baseline.py --db $datasetPath --out Model/runs/decoded-q4-v4.npz
.\Model\.venv\Scripts\python.exe Model/baseline.py --db $validationPath --out Model/runs/decoded-tos-q4-v4.npz
# 小型流程檢查，不代表完整訓練或品質驗證
.\Model\.venv\Scripts\python.exe Model/train.py --db $datasetPath --architecture v9 --v9-layers 2 --v9-variant relu16 --decoded-cache Model/runs/decoded-q4-v4.npz --history 1 --device cuda --epochs 1 --max-steps 20 --val-steps 4 --out Model/runs/smoke-v9-relu16-q4
# v9 正式單幀：只用本幀本條 decoded＋phase
.\Model\.venv\Scripts\python.exe Model/train.py --db $datasetPath --architecture v9 --v9-layers 2 --v9-variant relu16 --decoded-cache Model/runs/decoded-q4-v4.npz --history 1 --device cuda --epochs 5 --out Model/runs/relu16-v9-q4
# 完整單幀基準
.\Model\.venv\Scripts\python.exe Model/train.py --db $datasetPath --architecture v4 --decoded-cache Model/runs/decoded-q4-v4.npz --history 1 --device cuda --epochs 3 --out Model/runs/hybrid-v4-res4-q4
# 16 幀固定窗口
.\Model\.venv\Scripts\python.exe Model/train.py --db $datasetPath --architecture v4 --decoded-cache Model/runs/decoded-q4-v4.npz --device cuda --history 16 --epochs 5 --out Model/runs/hybrid-v4-history16-q4
# 以另一部來源影片作獨立驗證分組（兩庫共用標頭 ID 必須相同）
.\Model\.venv\Scripts\python.exe Model/train.py --db $datasetPath --val-db $validationPath --decoded-cache Model/runs/decoded-q4-v4.npz --val-decoded-cache Model/runs/decoded-tos-q4-v4.npz --architecture v4 --history 1 --device cuda --epochs 3 --out Model/runs/hybrid-v4-res4-q4-unseen-video
```

每個輸出目錄只能用一次，避免意外覆寫。4 GB GPU 從 batch size 8 開始，記憶體不足時用 `--batch-size 2`。
v3～v8 使用 FP32、Adam 與高解析度 RAW L1 loss；沒有低解析度預測 loss、GAN 或感知 loss。
`best.pt` 是驗證 MAE 最低的權重，`last.pt` 包含最後一輪權重、optimizer 與設定；`--warm-start` 只載入同架構、來源與切分的權重，optimizer 重建。
`metrics.jsonl` 記錄 RAW 8-bit MAE、RAW PSNR 與整輪耗時；`epoch-*-raw.png` 上方為標籤、下方為預測的灰階 Bayer 馬賽克，不是顯色後品質。
`--max-steps` 和 `--val-steps` 非零時只評估部分資料，數值不能當完整資料集成績。

## 架構與輸入約定

目前 v9 兩層版為 **Conv3×3 4→4 → Conv3×3 4→4**，中間沒有 activation，296 參數。CLI 以 `v9_layers=2, v9_variant=plain4` 區分；原 588 參數舊權重沒有 variant 欄位時仍按 relu8 還原。25 項測試通過，跨層數／跨 variant 的 warm-start 會拒絕。以下單層／原 8 通道段落保留作比較。

本次單層 v9 僅一個 **3×3 4→4 卷積，148 參數**，零初始化後學放大四平面的修正，沒有隱藏層或 ReLU。輸入／固定相位放大／RAW 交錯／clamp 與原兩層相同。`v9_layers` 區分版本；舊 checkpoint 無此欄位時載入原兩層。24 項模型／紀錄測試通過，禁止跨層數 warm-start。下段為原兩層結構。

v9 的模型資訊為 **decoded `[B,1,1,4,160]`＋phase `[B,1]`**，只用當前一條。固定放大後的 RGGB 四平面進入兩層 4→8→4 卷積，預測修正並加回放大結果。588 參數；無 byte encoder、無 phase embedding、無時間融合、無殘差區塊、無 Tanh。共用 CLI 保留 tokens／lengths 位置參數做舊流程相容，v9 不使用其值。22 項模型／紀錄測試通過，真實推理的庫內 decoded／快取輸出逐 byte 相同。

以下雙幀、雙輸入及擴張收縮描述為 **v3～v8 舊架構**，不套用到 v9。

正式原 v7 將本條的當前與前一幀按 phase 放大、固定時間權重融合，與 encoded 條件特徵一起進入 CNN，只輸出本幀本條 640×16 RAW。參數為 114,828，影像主幹共 12 層卷積；每個殘差區塊通道 24→48→24，最大 48 通道。

tokens `[B,2,640]`、decoded `[B,2,1,4,160]`、phase／length `[B,2]`，索引 0 為本幀、1 為前幀。上一條 CLI 選項與融合邏輯已移除。帶 previous_block=true 的歷史權重會明確拒絕載入，重現該實驗請使用紀錄保存的程式快照。

新 v8 保持同樣輸入與融合方式，每區塊改為 24→96→24，殘差 expansion=4、layers=2、variant=expand96_two_layer，參數 218,748。舊四層 v8 的 checkpoint 已保存 residual_layers=4，會自動載入舊架構；也可用 `--v8-variant legacy_four_layer` 重現。warm-start 不允許跨 variant。

以下保留舊四層 v8 架構說明供比較：

影像主幹為 `輸入卷積＋ReLU → 5 × 擴張收縮殘差區塊 → 輸出卷積＋Tanh`。
v8 每區塊含四個 3×3 卷積，依序 `24→48→96→48→24`，前三層後有 ReLU，第四層後加回區塊輸入。空間尺寸與區塊輸出 24 通道保持一致；影像主幹共 22 層卷積。
擴張／收縮倍數指通道數，實際卷積參數量由輸入、輸出通道及核大小共同決定。v8 共 530,268 個參數，設定另存 `residual_layers=4`。舊 v7 為每區塊兩層 24→48→24，共 12 層影像卷積，保留比較。

- 正式模型呼叫為 `model(tokens, phases, lengths, decoded_raw)`；decoded 是 `[B,T,1,4,160]` float32、0…1，與 encoded 當前到最舊一一配對。
- 正式 T=2：tokens `[B,2,640]`、decoded `[B,2,1,4,160]`，本幀與前幀各有 encoded 與 decoded。
- decoded 每份原始大小 640 bytes；可由接收端解碼產生，不必增加網路傳輸內容。
- 殘差最後一層零初始化，起始輸出等於當前 decoded 的 phase-aware bilinear 放大，訓練後不保證一定優於基準，仍須驗收。
- v7 共 114,828 個參數；v6 為 73,284 個，v5 為 62,868 個，v4 為 52,452 個，淺層 v3 為 15,996 個。`architecture` 保存在 checkpoint，v7 另記錄 `residual_expansion=2`；舊權重仍能載入自己的架構，不能直接互相套用或 warm-start。

- 每份 640 bytes 為 `2-byte big-endian length + payload + padding`，長度僅計 payload，上限 638。
- 載入時 padding 轉成專用 token 256，與有效的 0x00 分開；這是模型內部表示，不改變線上封裝。
- phase 是獨立的 0-based 輔助輸入，由資料庫 1…16 轉換而來，不占用 640 bytes。
- 固定 `exp(-0.2*k)` 正規化融合，當前到最舊順序；可用 `--decay` 修改正值衰減。
- 只取相同 block_index 的兩幀，歷史缺失就略過窗口。
- 不跨資料庫／切分邊界；尚未自動偵測影片內場景切換，正式資料應先按場景分組。
- 必須固定 encoder／shared header；不同設定不可混訓。

目前只有雙幀時間融合，沒有鄰條空間上下文、運動對齊或串流特徵快取；尚未完成即時吞吐量驗收。
直接訓練與推理每次重算窗口，先驗證內容學習能力。

## 驗證限制

目前僅一部影片，預設前 80% 幀訓練、後 20% 幀驗證，兩側分別建立歷史，沒有窗口跨界。
這只供初步實驗，**不能代表新影片泛化能力**。正式驗證必須使用不同來源影片／獨立場景，另留測試集。
可傳 `--val-db 路徑` 使用另一部影片的 v3 資料庫，必須採相同 encoder／header 設定。
請勿把同影片的 Q=8 版本當作 Q=6 訓練的獨立驗證資料。

## 單區塊推理與測試

下列使用既有 **Q=6 歷史權重**，因此明確切回相符 Q=6 資料；Q=4 完整訓練後須使用自己的權重與快取。

```powershell
$datasetPath = 'training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q6-shared-v3-raw8.sqlite'
.\Model\.venv\Scripts\python.exe Model/predict.py --checkpoint Model/runs/hybrid-v7-history2-expand2/best.pt --db $datasetPath --frame 750 --block 7 --out Model/runs/hybrid-v7-restored-example
# 可選：以完全相同的離線解碼快取取代即時解碼
.\Model\.venv\Scripts\python.exe Model/predict.py --checkpoint Model/runs/hybrid-v7-history2-expand2/best.pt --db $datasetPath --decoded-cache Model/runs/decoded-q6.npz --frame 750 --block 7 --out Model/runs/hybrid-v7-restored-example-cached
.\Model\.venv\Scripts\python.exe Model/evaluate.py --db $datasetPath --decoded-cache Model/runs/decoded-q6.npz --checkpoint Model/runs/hybrid-v7-history2-expand2/best.pt Model/runs/hybrid-v8-history2-expand96/best.pt --start-frame 720 --video --out Model/runs/eval-hybrid-v8-expand96
.\Model\.venv\Scripts\python.exe -m unittest discover -s Model -p 'test_*.py' -v
```

推理輸出 `.raw` 是 row-major 10,240 bytes，另附標籤／預測對照 PNG。
目前推理 CLI 從 SQLite 取得輸入；v3～v8 預設重組碼流並呼叫解碼器產生低解析度正式輸入，尚未接 BHXS 線上串流。

單區塊模型呼叫時間可用下列命令量測（若輸出已存在請改名）：

```powershell
.\Model\.venv\Scripts\python.exe Model/benchmark.py --db $datasetPath --decoded-cache Model/runs/decoded-q6.npz --checkpoint Model/runs/hybrid-v3/best.pt Model/runs/hybrid-v4-res4/best.pt --out Model/runs/res4-latency.json
```

此工具使用已放入裝置的 batch=1 張量，包含 forward 與同步，不含解碼、I/O、資料傳輸與排隊，不能當作端到端串流效能。

## 歷史：16 幀品質實驗

已完整訓練 5 輪，最佳為第 4 輪。相同 4,950 個窗口上，16 幀 **20.71 MAE／18.87 dB**，單幀 **21.23／18.48 dB**，有小幅改善；完整結果與限制見 [十六幀實驗紀錄](./實驗紀錄-v4十六幀.md)。
連續影片已輸出至 `runs/eval-hybrid-v4-history16/comparison.mp4`，165 幀、30 FPS、5.5 秒。

維持 v4 四殘差區塊與雙輸入，使用當前及前 15 幀相同 block_index；每幀各有 encoded 與 decoded RAW。
訓練窗口不跨來源／切分邊界，有缺幀就略過，順序為當前到最舊。
目前融合只補償採樣 phase，沒有光流或物體運動對齊，須觀察移動物體殘影。
本階段先驗證品質，不以即時效能限制訓練。

以目前單影片資料，16 幀模式有 21,150 個訓練窗口、4,950 個驗證窗口。
驗證分界仍為 frame 720，但有效當前幀從 735 開始，因為歷史不能跨入訓練集。
與單幀模型比較時必須用同一組 4,950 個窗口，不能直接拿先前 5,400 區塊的數字比較。

```powershell
.\Model\.venv\Scripts\python.exe Model/evaluate.py --db $datasetPath --decoded-cache Model/runs/decoded-q6.npz --checkpoint Model/runs/hybrid-v4-res4/best.pt Model/runs/hybrid-v4-history16/best.pt --start-frame 720 --video --out Model/runs/eval-hybrid-v4-history16
```

`--video` 需要 FFmpeg，可輸出完整連續幀的 `comparison.mp4`，30 FPS、各方法相同顯色。
它是離線品質展示，不代表即時處理速度。數值照常涵蓋所有有效窗口，不只展示幀。
另記錄未經光流對齊的相鄰幀「預測誤差變化」MAE，僅為時間穩定性診斷，不能單獨判定閃爍或殘影驗收通過。

## 歷史：純 bytes 首次實跑（2026-11-21）

Windows、Python 3.13.1、GTX 1650 4 GB，CUDA 實際矩陣運算成功。
歷史窗口分界／缺幀、有效零值與 padding、輸出尺寸、梯度及遞減權重共兩項測試通過。
單幀與 16 幀短訓練、checkpoint 載入及 RAW 推理皆成功。

`runs/single-v1` 是 Q=6 單幀基準，6,632,260 個參數，batch size 8，seed 42。
完整訓練 3 輪，每輪 21,600 筆；驗證是同影片後段 5,400 筆，沒有截短評估。

| 輪次 | 訓練 RAW MAE（0–255） | 驗證 RAW MAE（0–255） | 驗證 RAW PSNR |
| --- | ---: | ---: | ---: |
| 1 | 26.20 | 37.36 | 13.49 dB |
| 2 | 17.58 | 33.63 | 13.98 dB |
| 3 | 15.01 | 31.04 | 14.72 dB |

額外從驗證窗口等距取 128 筆，正確輸入 MAE 31.02；將整批輸入（含 phase、長度）循環錯配一筆後，MAE 52.22。
這支持模型使用了輸入相關資訊，但不證明它學會通用解碼或能泛化至新影片。
對照圖仍偏平滑，尚未達成品質驗收。最佳權重為 `runs/single-v1/best.pt`，
範例為 `runs/single-v1/example-750-7.raw` 與 `.png`。`runs/` 不納入 Git。

## 歷史：純 bytes v2 與解碼對照

v1 在相同 5,400 筆驗證區塊上 MAE 31.04／PSNR 14.72 dB，
解碼後 phase-aware nearest 為 30.42／16.00 dB，bilinear 為 28.76／16.68 dB。
所以 v1 **未達標**；不能用其訓練誤差下降宣告有效。

v2 仍只輸入 encoded，不呼叫解碼器。byte embedding 與 1D CNN 先預測 160 × 4 RAW，
訓練用真實解碼結果作額外 L1 監督；按採樣相位插值後，加上小型 CNN 預測的有界殘差。
殘差頭從零初始化，避免一開始任意破壞預測低解析度的插值結果。
這不保證勝過真正解碼，必須通過實測。v2 有 1,659,028 個參數，v1 有 6,632,260 個。

若使用 16 幀，v2 各幀預測後先依 phase 放到共同座標，以遞減時間權重提供歷史上下文，
當前預測另有直接路徑。時間權重約束作用於融合分支，不代表整個非線性模型的最終像素貢獻嚴格依時間遞減。
若需求要求最終有效貢獻也嚴格遞減，這個版本尚不能視為符合；目前 v2 實驗以單幀為主。

```powershell
# 建立批次解碼器（測試與 release 編譯）
.\training-data-preparation\build-jpegxs.ps1
# 產生離線解碼快取；不從高解析度標籤產生，並保存資料庫與解碼器 SHA-256
.\Model\.venv\Scripts\python.exe Model/baseline.py --db $datasetPath --out Model/runs/decoded-q6.npz --decoder training-data-preparation/tools/jpegxs-bridge/target/release/brainhole-jpegxs.exe
# 改良模型
.\Model\.venv\Scripts\python.exe Model/train.py --db $datasetPath --architecture v2 --decoded-cache Model/runs/decoded-q6.npz --history 1 --device cuda --epochs 8 --out Model/runs/single-v2
# 相同樣本下比較 v1、v2、nearest 與 bilinear
.\Model\.venv\Scripts\python.exe Model/evaluate.py --db $datasetPath --decoded-cache Model/runs/decoded-q6.npz --checkpoint Model/runs/single-v1/best.pt Model/runs/single-v2/best.pt --start-frame 720 --out Model/runs/eval-v2
```

`--val-db` 搭配 v2 時另須提供該資料庫的 `--val-decoded-cache`。
本次既有 `bin/brainhole-jpegxs.exe` 被其他程序使用，build 腳本在複製階段無法覆寫；
Rust 測試與 release 編譯已成功，快取使用上方命令指定的 `target/release` 新執行檔，沒有停止既有程序。
解碼快取必須與資料庫 SHA-256 相符，不能混用不同來源或編碼設定。
這個「只作輔助標籤」限制僅適用舊 v2；新 v3 的 decoded 是正式輸入，推理可即時解碼或讀取同筆解碼快取。

`evaluate.py` 在相同區塊集合計算所有方法的 RAW MAE／PSNR、逐區塊同時勝過兩種對照的比例，
輸出 `report.json`、`blocks.jsonl`，並將指定幀拼回完整 RAW，再用相同的固定 bilinear Bayer 顯色輸出各方法的 RGB PNG。
這是對照用顯色，不含白平衡、色彩校正或感測器 ISP。
數值初篩通過不代表視覺、新影片或即時效能驗收通過。
