"""Second Streamlit problem: TSMC 2330.TW forecasting with a reusable workflow."""
import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stock_workflow import normalize_prices, train_forecaster, forecast_next, packed_model

PURPLE = '#5244db'
GREEN = '#0fae84'

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo(ticker, period):
    import yfinance as yf
    downloaded = yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False, threads=False)
    if downloaded.empty:
        raise ValueError('Yahoo Finance 沒有回傳資料；可改用 CSV 上傳或稍後重試。')
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)
    return downloaded.reset_index()[['Date', 'Close']]


def stock_page(report_mode=False):
    st.markdown('<div class="hero"><h2>2330.TW — Stock Forecast Workflow</h2><p>台積電歷史股價預測：以過去收盤價建立 Lag / Moving Average Features，依時間順序切分資料，使用 Ridge Regression + TimeSeriesSplit 評估與調參。</p><div class="hero-tags">◈ 2330.TW &nbsp;&nbsp; ⚡ sklearn / Optuna &nbsp;&nbsp; ✦ Time-series Split &nbsp;&nbsp; ◆ Model Deployment</div></div>', unsafe_allow_html=True)
    st.markdown('### 1. Data Understanding — 股價資料來源')
    with st.container(border=True):
        src = st.radio('資料來源', ['Yahoo Finance (2330.TW)', 'CSV URL', 'Upload CSV'], horizontal=True, key='stock_source')
        data = None
        if src.startswith('Yahoo'):
            ticker = st.text_input('股票代碼', value='2330.TW')
            period = st.selectbox('歷史區間', ['2y', '3y', '5y', '10y'], index=1)
            if st.button('↻ Load Stock Data', type='primary'):
                try:
                    st.session_state['stock_dataset'] = fetch_yahoo(ticker, period)
                    st.session_state.pop('stock_result', None)
                    st.session_state.pop('stock_clean_prices', None)
                except Exception as exc:
                    st.error(f'無法取得股價資料：{exc}')
        elif src == 'CSV URL':
            url = st.text_input('CSV URL（需包含 Date 與 Close 欄位）')
            if st.button('↻ Load Stock CSV', type='primary'):
                try:
                    if not url.startswith('https://'):
                        raise ValueError('為安全起見，只接受 HTTPS CSV URL。')
                    import requests
                    with requests.get(url, stream=True, timeout=20) as res:
                        res.raise_for_status()
                        chunks, total = [], 0
                        for chunk in res.iter_content(65536):
                            total += len(chunk)
                            if total > 10_000_000:
                                raise ValueError('CSV 不可超過 10 MB')
                            chunks.append(chunk)
                    st.session_state['stock_dataset'] = pd.read_csv(io.BytesIO(b''.join(chunks)))
                    st.session_state.pop('stock_result', None)
                except Exception as exc:
                    st.error(f'讀取失敗：{exc}')
        else:
            upload = st.file_uploader('上傳歷史股價 CSV', type='csv', key='stock_csv')
            if upload is not None:
                signature = (upload.name, upload.size, upload.getvalue()[:256])
                if st.session_state.get('stock_upload_signature') != signature:
                    try:
                        st.session_state['stock_dataset'] = pd.read_csv(upload)
                        st.session_state['stock_upload_signature'] = signature
                        st.session_state.pop('stock_result', None)
                        st.session_state.pop('stock_clean_prices', None)
                    except Exception as exc:
                        st.error(f'無法讀取 CSV：{exc}')
        data = st.session_state.get('stock_dataset')
        if data is not None:
            c1, c2 = st.columns(2)
            with c1:
                date_col = st.selectbox('Date Column', data.columns.tolist(), index=data.columns.tolist().index('Date') if 'Date' in data else 0)
            with c2:
                close_col = st.selectbox('Close Column', data.columns.tolist(), index=data.columns.tolist().index('Close') if 'Close' in data else min(1, len(data.columns)-1))
            try:
                prices = normalize_prices(data, date_col, close_col)
                st.session_state['stock_clean_prices'] = prices
                a,b,c = st.columns(3)
                a.metric('ROWS', len(prices))
                b.metric('FIRST DATE', str(prices['Date'].iloc[0].date()))
                c.metric('LAST CLOSE', f"{prices['Close'].iloc[-1]:,.2f}")
                with st.expander('檢視資料與歷史價格', expanded=report_mode):
                    st.dataframe(prices.tail(15), use_container_width=True)
                    fig = go.Figure(go.Scatter(x=prices['Date'], y=prices['Close'], mode='lines', line=dict(color=PURPLE), name='Historical Close'))
                    fig.update_layout(height=300, xaxis_title='Date', yaxis_title='Adjusted Close / Close', margin=dict(l=25,r=15,t=15,b=25))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f'資料驗證失敗：{exc}')
                st.session_state.pop('stock_clean_prices', None)
        else:
            st.info('請先載入 Yahoo Finance 股價資料或提供 CSV。')
    prices = st.session_state.get('stock_clean_prices')
    if prices is None:
        return
    st.markdown('### 2. Preprocessing & Model Setup')
    with st.container(border=True):
        st.caption('採用 lag_1、lag_2、lag_3、lag_5、lag_10、MA5、MA10 與前一天的價格變化；移動平均都只使用預測日以前的資料。')
        a,b,c = st.columns(3)
        with a: frac = st.slider('Test fraction', 0.15, 0.35, 0.2, 0.05)
        with b: optimize = st.toggle('Optuna Ridge tuning', value=False, key='stock_optimize')
        with c: trials = st.slider('Optuna trials', 5, 50, 20, 5, disabled=not optimize, key='stock_trials')
        clicked = st.button('🚀 Train Stock Forecast Model', type='primary', use_container_width=True)
        if clicked:
            try:
                with st.spinner('依照時間序列切分、交叉驗證及訓練中...'):
                    st.session_state['stock_result'] = train_forecaster(prices, frac, optimize, trials)
                    st.session_state['stock_train_sig'] = (len(prices), str(prices['Date'].iloc[-1]), float(prices['Close'].iloc[-1]), frac, optimize, trials)
                st.success('模型訓練完成！')
            except Exception as exc:
                st.error(f'模型訓練失敗：{exc}')
    res = st.session_state.get('stock_result')
    if res is None:
        return
    sig = (len(prices), str(prices['Date'].iloc[-1]), float(prices['Close'].iloc[-1]), frac, optimize, trials)
    if sig != st.session_state.get('stock_train_sig'):
        st.warning('設定或資料已變更，請重新訓練以產生新結果。')
        return
    st.markdown('### 3. Evaluation & Forecast')
    m = res['test']
    cards = st.columns(4)
    for box, label, value in zip(cards, ['TEST RMSE', 'R² SCORE', 'MAE', 'MSE'], [f"{m['RMSE']:,.2f}", f"{m['R²']:.4f}", f"{m['MAE']:,.2f}", f"{m['MSE']:,.0f}"]):
        box.metric(label,value)
    st.caption(f"Chronological training: {res['train_count']} samples｜Test: {res['test_count']} samples｜Naive last-close baseline RMSE: {res['naive_rmse']:,.2f}｜CV RMSE: {res['cv_mse']**.5:,.2f}｜Ridge alpha: {res['alpha']:.5g}")
    st.markdown('#### Model Comparison — Ridge vs Naive Baseline')
    compare = pd.DataFrame([res['naive_test'], res['test']], index=['Naive last close', 'Ridge Regression'])
    st.dataframe(compare.round(4), use_container_width=True)
    if res['test']['RMSE'] < res['naive_rmse']:
        st.success(f"Ridge 的測試 RMSE 比 Naive Baseline 低 {100*(res['naive_rmse']-res['test']['RMSE'])/res['naive_rmse']:.2f}%。")
    else:
        st.warning('目前 Ridge 未勝過前一日收盤價基準；較高的 R² 不代表一定優於簡單基準。')
    st.warning('股價資料具有不確定性，僅作為教學示範，並非投資建議。請比較模型 RMSE 與前一日收盤價基準，勿只看 R²。')
    section_labels = ['📈 Actual vs Predicted', '📋 Detailed Metrics', '🔮 Next Price Forecast']
    tabs = [st.container() for _ in section_labels] if report_mode else st.tabs(section_labels)
    with tabs[0]:
        if report_mode:
            st.markdown("#### 📈 Actual vs Predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['date_test'], y=res['actual_test'], name='Actual Close', mode='lines', line=dict(color=GREEN, width=2)))
        fig.add_trace(go.Scatter(x=res['date_test'], y=res['pred_test'], name='Ridge Prediction', mode='lines', line=dict(color=PURPLE, width=2)))
        fig.add_trace(go.Scatter(x=res['date_test'], y=res['naive_pred_test'], name='Naive (Previous Close)', mode='lines', line=dict(color='#f59e0b',width=1.8,dash='dot')))
        fig.update_layout(height=400, margin=dict(l=30,r=25,t=20,b=25), xaxis_title='Date', yaxis_title='Close Price', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        if report_mode:
            st.markdown("#### 📋 Detailed Metrics")
        st.dataframe(pd.DataFrame([res['train'], res['test']], index=['Train', 'Test']).round(4), use_container_width=True)
        st.write(f"**Ridge alpha:** {res['alpha']:.6g}　｜　**TimeSeriesSplit CV RMSE:** {res['cv_mse']**0.5:,.2f}")
        st.caption('測試集依日期排序，沒有使用隨機 train_test_split。TimeSeriesSplit 只在訓練集進行模型調參。')
    with tabs[2]:
        if report_mode:
            st.markdown("#### 🔮 Next Price Forecast")
        horizon = st.slider('預測未來幾個交易觀測點（遞迴預測）', 1, 10, 3)
        if report_mode or st.button('✨ Forecast Next Prices'):
            future = forecast_next(res, horizon)
            st.dataframe(pd.DataFrame({'Forecast Step': range(1, horizon+1), 'Predicted Close': future}), hide_index=True, use_container_width=True)
            fig_future = go.Figure()
            lookback = min(40, len(res['prices']))
            recent = res['prices'].tail(lookback)
            fig_future.add_trace(go.Scatter(x=list(range(-lookback+1,1)), y=recent['Close'],name='Historical Close', mode='lines',line=dict(color=PURPLE)))
            fig_future.add_trace(go.Scatter(x=[0]+list(range(1,horizon+1)), y=[float(recent['Close'].iloc[-1])]+future,name='Recursive Forecast',mode='lines+markers',line=dict(color=GREEN,dash='dash')))
            fig_future.update_layout(height=330, xaxis_title='Relative observation (0 = latest close)',yaxis_title='Close Price')
            st.plotly_chart(fig_future,use_container_width=True)
            st.caption('Step 是未來交易觀測點順序，未套用交易所休市行事曆；多步遞迴誤差可能快速累積。')
    st.markdown('### 4. Deployment')
    with st.container(border=True):
        st.download_button('⬇ Download TSMC Forecast Model (.pkl.gz)', packed_model(res), file_name='stock_forecast.pkl.gz', mime='application/gzip')
        st.caption('下載完成最終全資料重訓的 Ridge 模型；pickle 檔案只可從可信來源載入。')
