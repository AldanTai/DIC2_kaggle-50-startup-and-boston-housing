# ML Regression Explorer — V5（繳交與部署準備版）

Streamlit 紫色 Dashboard，兩個 Problem：50 Startups 多元線性迴歸與 2330.TW 股價預測。

## 執行

```powershell
python -m pip install -r requirements.txt
python verify_project.py
streamlit run app.py
```

## 功能

- Problem 下拉選單；兩題切換。
- 50 Startups：CSV URL / 上傳、自選 Target / Features、One-Hot Encoding、LinearRegression / Optuna Ridge、特徵選擇、學習曲線、預測與模型下載。
- 2330.TW：Yahoo Finance / CSV、時序特徵、TimeSeriesSplit、Ridge 與 Naive 基準比較、遞迴預測、模型下載。
- `.streamlit/config.toml` 與 `runtime.txt` 為部署準備設定。

## 文件

- `DEPLOYMENT.md`：GitHub + Streamlit Community Cloud 部署逐步說明。
- `REPORT_TEMPLATE.md`：依 CRISP-DM 整理的中文作業報告範本（結果欄位須以實測填寫）。
- `VERIFY_CHECKLIST.md`：課堂加分與正式繳交前檢查清單。
- `verify_project.py`：使用**模擬資料**驗證兩題核心模型，不驗證外部 Yahoo Finance 和 Streamlit UI。

## 已知限制

本機／雲端實際互動尚需測試；任意 CSV 只限於具適當資料結構的回歸或時間序列任務。部署為公開服務前應額外限制 CSV URL 可存取網域，避免網路存取風險。pickle 只應在可信環境載入。股價預測僅供課程展示，不構成投資建議。
