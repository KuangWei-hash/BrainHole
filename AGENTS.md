# BrainHole 接手指引

- 正式專案根目錄為 `W:\BrainHole`。v13 採固定 v12 底圖的 AI 殘差修復。接手時先讀 [v13 文件與最新狀態](Model/v13/README.md) 及 `Model/AGENTS.md`。
- 使用者指定 v12 現有視覺成果為堪用最低合格線；後續各方法分別輸出原解析度影片，不再產生並排縮小的 `comparison.mp4`。

- 接手 v10 前先讀 [RTX 5060 環境與恢復指令](Model/實驗紀錄-v10-RTX5060環境檢查.md#下次開啟-codex) 和 [模型實驗紀錄要求](Model/AGENTS.md)。
- 品質研究先以 Big Buck Bunny Q=5 同片可行性為範圍。v10 生成研究與 v13 修復研究分開記錄；固定場景和影片比較材質、輪廓及時序穩定，MAE 用於追蹤內容失真。v10 歷史狀態和恢復指令在環境紀錄中，環境路徑須重新確認。

- README 維護要求（2026-11-22）：根目錄 README.md 是持續更新的專案入口。每次新增實驗結果、目視判斷或研究方向決策，都必須依 Model/AGENTS.md 的同步規則檢查並更新 README.md 與 Model/README.md；首頁始終先交代最終目標，再摘要最新狀態，不能累積成歷史實驗流水帳。
