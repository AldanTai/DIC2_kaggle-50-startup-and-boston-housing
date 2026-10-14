# V5 部署步驟（GitHub + Streamlit Community Cloud）

## A. 本機確認

1. 建議使用 Python 3.11 建立新的虛擬環境：Windows PowerShell 執行 `py -3.11 -m venv .venv`，再執行 `.\.venv\Scripts\Activate.ps1`。
2. 安裝 `python -m pip install -r requirements.txt`。
3. 執行 `python verify_project.py` 進行離線模型流程檢查。
4. 執行 `streamlit run app.py` 並操作兩題；特別測試 Yahoo Finance 連線、未來價格預測表格、CSV 上傳、模型下載。

## B. 上傳 GitHub

1. 登入 GitHub，建立新的 Repository（例如 `ml-regression-explorer`）。
2. 把專案內的 `app.py`、`modeling.py`、`stock_page.py`、`stock_workflow.py`、`requirements.txt`、`.streamlit/config.toml` 等一起上傳至 Repository 根目錄。**不要上傳密碼、token、`.env` 或 `.streamlit/secrets.toml`。**
3. 如果公開 Repository，請勿上傳含個資的 CSV；只有範例資料時可改用公開 URL。

## C. 部署 Streamlit

1. 到 https://share.streamlit.io/ 使用 GitHub 登入。
2. 選擇 **Create app** → 選擇 Repository / branch。
3. Main file path 填入 `app.py`，選擇可用的 app 網址並部署。
4. 開啟網站檢查兩題，特別注意免費資源下 Optuna 的執行時間。
5. 建議部署後將 URL 交給不同網路的同學測試，確認不是僅限 localhost。

## D. 風險與範圍

- 免費雲端服務可能休眠、限制記憶體、網路讀取或 CPU；Yahoo Finance 亦可能限流。
- CSV URL 只接受合理、可信來源；公開展示時須防範伺服器端任意 URL 請求風險。要正式公開給不受信任的訪客時，應進一步做網域白名單或封鎖內網位址。
- `pickle` 僅能載入信任的模型，且 sklearn 套件版本需相容。
- 自訂 CSV 僅保證支援符合回歸／股價欄位要求的小型資料，不保證任意 CSV 都可用。
- 這份程式是課程示範；股價模型不是投資建議。
