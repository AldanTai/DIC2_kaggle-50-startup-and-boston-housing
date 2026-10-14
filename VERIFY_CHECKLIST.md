# V5 最終驗收清單

## Problem 1
- [ ] 讀取老師提供的 50 Startups CSV；確認目標 `Profit`，特徵為 R&D Spend / Administration / Marketing Spend / State
- [ ] Train 與 Test 指標正常
- [ ] Baseline vs Optuna 有比較表與曲線（開啟 Optuna 後）
- [ ] Feature Selection、Learning Curve、Random Data Scatter Plot 正常
- [ ] 上傳另一個具數值 Target 的 CSV，能自行選目標／欄位，UI 標題能隨目標改變
- [ ] `.pkl.gz` 下載，並從本機可用的相容環境載入做一次預測

## Problem 2
- [ ] 使用 Yahoo Finance 讀取 2330.TW；若失敗以合法 CSV 測試
- [ ] 歷史圖、日期／收盤價欄位辨識正確
- [ ] 使用時間順序切分；Ridge 與 Naive Baseline 比較
- [ ] 展開 Next Price Forecast，顯示表格與曲線
- [ ] 開啟 Optuna 再跑一次，確認最佳 alpha 顯示
- [ ] `.pkl.gz` 下載

## 部署
- [ ] `python verify_project.py` 成功
- [ ] GitHub 不含密碼或私密資料
- [ ] Streamlit Cloud 網址在不同電腦與網路開啟成功
- [ ] 截圖首頁、兩題圖表、自訂 Dataset、公開網站 URL；補入 `REPORT_TEMPLATE.md`
