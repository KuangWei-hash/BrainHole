# v12 BBB Q=5 最近 Bayer 色塊

2026-11-22，將 v12 的單幀填補改為以完整 2×2 Bayer 色塊為單位的最近鄰。每格依本幀相位
找最近的解碼色塊，四個 byte 一起複製。等距取右或下，超界取邊界色塊。解碼資料本身的
CFA 排列用於輸出及顯色。[v12 規格](v12/README.md)

## 資料與設定

BBB Q=5 schema v4，使用 160×4 decoded RAW8，填至 640×16。資料庫 SHA-256 為
`11c05bdd430ba67fa84a986288c9e48786b92aded73fa38c51661acaac8975eb`。來源幀 721–899，
同一區塊的完整 16 幀窗口目標為 736–899、4,920 條，沿用 [場景清單](runs/v11-q5-20261122-scene-plan.json)。

相位保護仍從實際解碼量測取 byte，權重為 1。同相位重複時採較新的一筆。完整 16 幀窗口中未被
量測的部位，才將最近色塊填補的各幀以原順序做 50/50。影片暖機時直接用本幀最近色塊填補。
訓練 0 步，最佳輪不適用。

## 結果

| 方法 | RAW MAE | RAW PSNR dB | RGB MAE | 相鄰幀殘差變動 MAE |
| --- | ---: | ---: | ---: | ---: |
| 單幀最近 Bayer 色塊 | 19.2467 | 18.4236 | 17.4774 | 19.2578 |
| v12 相位保護 | 15.0084 | 21.8484 | 12.3820 | 3.1346 |

4,920 條完整窗口的相位覆蓋皆為 100%。逐一核對共同的幀號與區塊，v12 的 RAW
與 RGB MAE 均與[前一版相位保護](實驗紀錄-v12相位保護Q5.md)相同。179 幀的影片中，
前 15 幀共 450 條用最近色塊暖機。第 721 幀可見整組複製造成的方格外觀，兩支影片均已整段
解碼驗證。全套 84 項測試通過，包含相位偏移、等距、邊界及色塊四個 byte 保持排列的檢查。

[自動紀錄](experiments/records/evaluation--v12-bbb-q5-bayer-nearest-20261122--run-3e0a9e12b71e.json) ·
[逐區塊結果](experiments/details/evaluation--v12-bbb-q5-bayer-nearest-20261122--run-3e0a9e12b71e.blocks.jsonl.gz) ·
[測試紀錄](experiments/records/checks--checks-v12-bayer-nearest-20261122--run-7454f4f6bbd6.json) ·
[暖機首幀](runs/v12-bbb-q5-bayer-nearest-20261122/preview-721.png) ·
[完整成片](runs/v12-bbb-q5-bayer-nearest-20261122/generated.mp4) ·
[並排影片](runs/v12-bbb-q5-bayer-nearest-20261122/comparison.mp4)
