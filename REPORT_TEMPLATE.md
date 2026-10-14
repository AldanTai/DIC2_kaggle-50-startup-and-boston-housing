# CRISP-DM 雙題機器學習專案報告（待填成績版）

## 1. 題目與目的

本專案實作兩種回歸問題：50 Startups 利潤預測（表格型多元迴歸）與 2330.TW 股價預測（時間序列回歸），以 Streamlit 下拉式選單整合於同一網站。並提供 CSV URL / CSV 上傳功能。

## 2. Problem 1 — 50 Startups

**Business Understanding：** 以研發、行政、行銷支出以及州別預測 Profit。

**Data Understanding：** 經典範例有 50 筆、5 欄；正式報告請填實際使用的資料來源、檢查日期、筆數、缺失值與數值分布。

**Data Preparation：** 自動選擇數值／類別欄位；缺失值填補、One-Hot Encoding、Train/Test Split；將轉換封裝在 Pipeline 內避免前處理資料洩漏。

**Modeling：** LinearRegression 作 Baseline，Optuna 搜尋 Ridge alpha；以訓練資料內交叉驗證進行模型選擇。另用 Forward Selection 比較不同特徵數量。

**Evaluation：** MAE、MSE、RMSE、R²；Actual vs Predicted、Learning Curve、Feature Selection、Baseline vs Optuna。請貼你在 V5 實際操作的截圖及數值：

| 指標 | LinearRegression | Optuna Ridge |
|---|---:|---:|
| Test MAE | 待填 | 待填 |
| Test MSE | 待填 | 待填 |
| Test RMSE | 待填 | 待填 |
| Test R² | 待填 | 待填 |
| CV RMSE | 待填 | 待填 |

**Deployment：** 以 gzip + pickle 打包完整 Pipeline，讓使用者下載後於可信環境載入。

## 3. Problem 2 — 2330.TW

**Data Understanding：** 使用 Yahoo Finance（調整後 Close）或使用者提供的日期／收盤價 CSV；記錄來源、期間和日期範圍。

**Data Preparation：** 按時間排序，使用過去價格 lag_1/2/3/5/10、MA5/MA10 和前一天的價格差；依日期先後切分訓練／測試資料，TimeSeriesSplit 調參。

**Modeling：** Ridge Regression，選用 Optuna 搜尋 alpha。

**Evaluation：** 比較 Ridge 與「前日收盤價」Naive Baseline，查看 MAE／MSE／RMSE／R² 與時間序列的 Actual vs Predicted 圖。不應將股價 R² 解釋為漲跌方向的準確率。

| 指標 | Ridge | Naive Baseline |
|---|---:|---:|
| Test MAE | 待填 | 待填 |
| Test RMSE | 待填 | 待填 |
| Test R² | 待填 | 待填 |

**Deployment：** 下載 `.pkl.gz` 模型；提供未來 1～10 個「交易觀測點」遞迴預測（非實際市場交易日行事曆）。

## 4. 網頁功能與加分項目

- 下拉選單呈現兩個問題、切換不同分析畫面（+1 分項目）。
- CSV URL／上傳 CSV，並依問題選擇 Target / Features 或 Date / Close（+2 分項目，限定資料適用範圍）。
- 模型比較、圖表、評估、即時預測和模型下載。

## 5. 測試紀錄與限制

填寫本機環境（Python / sklearn / Streamlit 版本）、部署網址、測試日期及問題排除紀錄。V5 出版時尚未驗證 Windows 瀏覽器和雲端實際執行，繳交前必須補上執行證據。

限制作業：50 筆樣本過少，單次 Test 數值不穩定；股票高自相關，預測勝過 Naive Baseline 的幅度才較有意義；使用者自訂資料可能有型態不適用或洩漏問題；外部資料來源可能失效。

## 6. 結論

請根據 V5 最終測試結果填寫：模型是否優於 Baseline、最佳特徵組合、兩題實際操作情形、改善空間。
