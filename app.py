"""Streamlit dashboard; run: streamlit run app.py"""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from modeling import prepare_xy, fit_and_evaluate, compute_learning_curve, forward_selection_cv, model_bytes
from stock_page import stock_page

st.set_page_config(page_title='50 Startups | CRISP-DM Lab', page_icon='📈', layout='wide', initial_sidebar_state='collapsed')
DEFAULT_URL = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv'
PURPLE, GREEN, BLUE = '#5244db', '#0fae84', '#576bdc'

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:Inter,system-ui,sans-serif;}
.stApp {background:#f5f7fc;color:#172044;}
.block-container {max-width:1400px;padding-top:1.1rem;padding-bottom:4rem;}
header[data-testid="stHeader"] {background:transparent;}
h1,h2,h3 {letter-spacing:-.035em;color:#162048;}
h1 {font-size:1.75rem!important;} h3 {font-size:1.2rem!important;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:14px!important;}
[data-testid="stMetric"] {background:white;border:1px solid #e3e8f3;border-radius:13px;padding:17px 20px;box-shadow:0 2px 8px #23316b0b;}
[data-testid="stMetricLabel"] {color:#697394;font-size:12px;font-weight:700;text-transform:uppercase;}
[data-testid="stMetricValue"] {color:#202e61;font-size:1.52rem;}
[data-testid="stButton"] button[kind="primary"] {background:linear-gradient(100deg,#403ab3,#6651ed);border:0;color:white;}
[data-testid="stDownloadButton"] button {border-color:#bbb7f8;color:#403ab3;}
.hero {padding:26px 30px;border-radius:16px;background:linear-gradient(115deg,#1c1b4e,#382c9b 75%,#4432b7);color:white;box-shadow:0 7px 18px #24216a20;margin:10px 0 21px;}
.hero h2 {color:white;font-size:1.75rem;margin:0 0 9px;letter-spacing:-.04em;}
.hero p {color:#d8d7ff;font-size:14px;line-height:1.7;margin:0;max-width:780px;}
.hero-tags {margin-top:16px;color:#e6e6ff;font-size:12px;font-weight:600;}
.section-caption {color:#647097;font-size:13px;margin:-4px 0 14px;}
.brand {font-weight:800;font-size:21px;color:#172044;padding:5px 0 12px;}
.small-cap {color:#6e78a0;font-size:12px;text-transform:uppercase;font-weight:800;letter-spacing:.06em;}
.tagline {background:#edebff;border-radius:6px;color:#4a40c1;padding:4px 8px;font-size:12px;font-weight:700;}
[data-testid="stPlotlyChart"] {background:white;border:1px solid #e3e9f3;border-radius:14px;overflow:hidden;}
.stTabs [data-baseweb="tab-list"] {gap:10px;}
.stTabs [data-baseweb="tab"] {font-weight:700;border-radius:8px;}
footer {visibility:hidden;}
</style>''', unsafe_allow_html=True)


def load_csv_url(url):
    if not url.startswith(('https://', 'http://')):
        raise ValueError('只支援 http:// 或 https:// CSV 網址')
    with requests.get(url, timeout=20, stream=True) as res:
        res.raise_for_status()
        chunks, total = [], 0
        for part in res.iter_content(chunk_size=65536):
            total += len(part)
            if total > 10_000_000:
                raise ValueError('檔案上限 10 MB')
            chunks.append(part)
    return pd.read_csv(io.BytesIO(b''.join(chunks)))


def chart_style(fig, height=300, x='', y=''):
    fig.update_layout(height=height, margin=dict(l=35,r=20,t=30,b=35),
        paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#5b6685',size=11),
        xaxis=dict(title=x, showgrid=True, gridcolor='#f0f2f9', zeroline=False),
        yaxis=dict(title=y, showgrid=True, gridcolor='#edf0f7', zeroline=False),
        legend=dict(orientation='h',y=1.15,x=0), hovermode='closest')
    return fig


def dataset_panel():
    st.markdown('### 1. Dataset & Data Understanding')
    st.caption('支援 50 Startups 預設資料、任意 CSV 連結或直接上傳表格型資料集。')
    with st.container(border=True):
        a, b = st.columns([1, 3])
        with a:
            source = st.radio('資料來源', ['CSV URL', 'Upload CSV'], key='src')
        with b:
            if source == 'CSV URL':
                url = st.text_input('Dataset URL', value=DEFAULT_URL)
                clicked = st.button('↻ Load Dataset', type='primary')
                if clicked:
                    try:
                        df = load_csv_url(url)
                        if len(df.columns) < 2 or df.empty:
                            raise ValueError('需要至少 2 個欄位與 1 筆資料')
                        st.session_state['dataset'] = df
                        st.session_state.pop('result',None)
                        st.session_state.pop('selection',None)
                    except Exception as exc:
                        st.error(f'讀取失敗：{exc}')
            else:
                file = st.file_uploader('上傳 .csv',type=['csv'])
                if file is not None:
                    sig = (file.name, file.size, file.getvalue()[:128])
                    if st.session_state.get('uploaded_sig') != sig:
                        try:
                            if file.size > 10_000_000:
                                raise ValueError('CSV 檔案超過 10 MB 限制')
                            new_data = pd.read_csv(file)
                            if len(new_data.columns) < 2 or new_data.empty:
                                raise ValueError('需要至少 2 個欄位與 1 筆資料')
                            st.session_state['dataset'] = new_data
                            st.session_state['uploaded_sig'] = sig
                            st.session_state.pop('result',None)
                            st.session_state.pop('selection',None)
                        except Exception as exc:
                            st.error(f'CSV 格式錯誤：{exc}')
        df = st.session_state.get('dataset')
        if df is not None:
            p, q, r = st.columns(3)
            p.metric('Rows',f'{len(df):,}')
            q.metric('Columns',len(df.columns))
            r.metric('Missing Values',int(df.isna().sum().sum()))
            with st.expander('預覽資料、欄位型態與缺失值',expanded=False):
                st.dataframe(df.head(12),use_container_width=True)
                info = pd.DataFrame({'Type':df.dtypes.astype(str),'Missing':df.isna().sum()})
                st.dataframe(info,use_container_width=True)
    return df


def visualizations(result, selection):
    result.setdefault('target','Profit')
    st.markdown('### 4. Evaluation & Parsimony Analysis')
    st.caption('以保留的 Test Set 評估表現；訓練期間使用 Cross-Validation 選擇模型。')
    m = result['test']
    cols = st.columns(4)
    for c, (label, val) in zip(cols, [('TEST RMSE',f"{m['RMSE']:,.2f}"),('R² SCORE',f"{m['R²']:.4f}"),('MAE',f"{m['MAE']:,.2f}"),('MSE',f"{m['MSE']:,.0f}")]):
        c.metric(label,val)
    report_mode = st.session_state.get('report_mode', False)
    section_labels = ['📈 Feature Selection', '🎯 Actual vs Predicted', '📉 Learning Curve', '📋 Detailed Metrics', '⚖️ Baseline vs Optuna']
    tabs = [st.container() for _ in section_labels] if report_mode else st.tabs(section_labels)
    with tabs[0]:
        if report_mode:
            st.markdown("#### 📈 Feature Selection")
        if selection is None or selection.empty:
            st.info('勾選 Forward Feature Selection 並重新訓練，即可顯示特徵數量比較。')
        else:
            selection = selection.copy()
            choice = selection['CV RMSE'].idxmin()
            best = selection.loc[choice]
            st.success(f"Cross-validation 最佳特徵組合：{best['Selected Features']}（{int(best['Feature Count'])} 個特徵，CV RMSE {best['CV RMSE']:,.2f}）")
            c1, c2 = st.columns(2)
            with c1:
                f = go.Figure(go.Scatter(x=selection['Feature Count'],y=selection['CV RMSE'],mode='lines+markers',line=dict(color=PURPLE,width=3),marker=dict(size=9)))
                f.add_trace(go.Scatter(x=[best['Feature Count']],y=[best['CV RMSE']],mode='markers',marker=dict(color=GREEN,size=14),showlegend=False))
                st.plotly_chart(chart_style(f,x='Number of Features',y='CV RMSE'),use_container_width=True)
            with c2:
                st.dataframe(selection.round(3),use_container_width=True,hide_index=True)
            st.caption('此處顯示 Training Set 內部的交叉驗證結果，不使用最終 Test Set 選特徵。')
    with tabs[1]:
        if report_mode:
            st.markdown("#### 🎯 Actual vs Predicted")
        y, pred = np.asarray(result['y_test']), np.asarray(result['pred_test'])
        low, high = min(y.min(),pred.min()),max(y.max(),pred.max())
        f = go.Figure()
        f.add_trace(go.Scatter(x=y,y=pred,mode='markers',name='Predictions',marker=dict(color=PURPLE,size=10,opacity=.82)))
        f.add_trace(go.Scatter(x=[low,high],y=[low,high],mode='lines',name='Ideal',line=dict(color=GREEN,dash='dash')))
        st.plotly_chart(chart_style(f,height=340,x=f"Actual {result['target']}",y=f"Predicted {result['target']}"),use_container_width=True)
    with tabs[2]:
        if report_mode:
            st.markdown("#### 📉 Learning Curve")
        try:
            curve = compute_learning_curve(result)
            f=go.Figure()
            for key,color in [('Training MSE',PURPLE),('Validation MSE','#f5a23d')]:
                f.add_trace(go.Scatter(x=curve['Training samples'],y=curve[key],mode='lines+markers',name=key,line=dict(color=color,width=3)))
            st.plotly_chart(chart_style(f,height=340,x='Training Set Size',y='Mean Squared Error'),use_container_width=True)
            st.caption('Validation MSE 明顯高於 Training MSE 可能是 Overfitting；兩者皆高可能是 Underfitting。小資料集曲線波動較大。')
        except ValueError as exc:
            st.warning(f'資料量不足以繪製曲線：{exc}')
    with tabs[3]:
        if report_mode:
            st.markdown("#### 📋 Detailed Metrics")
        st.dataframe(pd.DataFrame([result['train'],result['test']],index=['Train','Test']).round(4),use_container_width=True)
        st.write(f"**Chosen model:** {result['name']}　｜　**Baseline CV MSE:** {result['baseline_cv_mse']:,.2f}")
        if result['ridge_cv_mse'] is not None:
            st.write(f"**Optuna Ridge CV MSE:** {result['ridge_cv_mse']:,.2f}　｜　**Best alpha:** {result['alpha']:.6g}")
        st.caption('MAE = 平均絕對誤差；MSE = 平均平方誤差；RMSE = 原單位的均方根誤差；R² = 解釋變異程度。')
    with tabs[4]:
        if report_mode:
            st.markdown("#### ⚖️ Baseline vs Optuna")
        st.subheader('Baseline vs Optuna Ridge — Train-only Cross Validation')
        entries = [{'Model':'LinearRegression', 'CV RMSE':np.sqrt(result['baseline_cv_mse'])}]
        if result['ridge_cv_mse'] is not None:
            entries.append({'Model':'Optuna Ridge', 'CV RMSE':np.sqrt(result['ridge_cv_mse'])})
        comparison = pd.DataFrame(entries)
        st.dataframe(comparison.round(3),hide_index=True,use_container_width=True)
        f = go.Figure(go.Bar(x=comparison['Model'],y=comparison['CV RMSE'],marker_color=[PURPLE,GREEN][:len(comparison)]))
        st.plotly_chart(chart_style(f,height=290,x='Model',y='Cross-validation RMSE'),use_container_width=True)
        st.write(f"**Selected model:** {result['name']}")
        if result['alpha'] is not None:
            st.write(f"**Best alpha:** {result['alpha']:.6g}")
        st.caption('只用訓練集內部 CV 選擇模型；不能用 Test Set 反覆調參。')
        st.subheader('Holdout Test comparison (for reporting only)')
        st.dataframe(pd.DataFrame([result['baseline_test'],result['test']],index=['LinearRegression','Selected model']).round(3),use_container_width=True)
    st.markdown('### 5. Deployment — Live Predictor & Model Download')
    left,right=st.columns([1.15,1])
    with left:
        with st.container(border=True):
            st.markdown(f"**🧮 Live {result['target']} Predictor**")
            st.caption('輸入一筆新資料，即可使用目前已訓練的完整 Pipeline 預測。')
            row={}
            num=result['numeric']; cats=result['categorical']
            dataset=st.session_state.get('dataset')
            with st.form('predict_form'):
                for col in num:
                    median=pd.to_numeric(dataset[col],errors='coerce').median() if col in dataset else 0.0
                    row[col]=st.number_input(col,value=float(median) if pd.notna(median) else 0.0,format='%.2f')
                for col in cats:
                    choices=dataset[col].dropna().astype(str).unique().tolist()[:100]
                    row[col]=st.selectbox(col,choices) if choices else st.text_input(col)
                predict=st.form_submit_button('✨ Predict Now',type='primary')
            if predict:
                try:
                    value=result['model'].predict(pd.DataFrame([row]))[0]
                    st.metric('PREDICTED VALUE',f'{value:,.2f}')
                except Exception as exc:
                    st.error(f'預測失敗：{exc}')
    with right:
        with st.container(border=True):
            st.markdown('**📦 Model Artifact**')
            st.write('下載完整前處理 + 訓練模型，以 gzip 壓縮 pickle 格式封裝。')
            st.download_button('⬇ Download .pkl.gz',data=model_bytes(result['model']),file_name='startup_model.pkl.gz',mime='application/gzip',use_container_width=True)
            st.caption('只應在可信任的環境載入 pickle 檔案。')
            st.divider()
            st.markdown('**實驗摘要**')
            st.write(f"模型：**{result['name']}**")
            st.write(f"Test R²：**{result['test']['R²']:.4f}**")
            st.write(f"Test RMSE：**{result['test']['RMSE']:,.2f}**")


def startup():
    st.markdown('<div class="hero"><h2>Feature Selection & Parsimony Analysis</h2><p>50 Startups Multiple Linear Regression — Explore model performance, compare feature subsets, inspect training behavior, and predict profits in real time. Built around the CRISP-DM workflow.</p><div class="hero-tags">◈ 50 Startups &nbsp;&nbsp; ⚡ sklearn / Optuna &nbsp;&nbsp; ✦ Forward Selection &nbsp;&nbsp; ◆ Model Deployment</div></div>',unsafe_allow_html=True)
    df=dataset_panel()
    if df is None:
        st.info('請先按「Load Dataset」載入預設 50 Startups CSV，或上傳自訂資料。')
        return
    st.markdown('### 2. Preprocessing & Model Setup')
    st.caption('CRISP-DM：選擇數值型 Target、挑選 Features，系統將自動填補缺失值、One-Hot Encoding 並切分 Train / Test。')
    with st.container(border=True):
        possible=[c for c in df.columns if pd.to_numeric(df[c],errors='coerce').notna().sum()>=15]
        if not possible:
            st.warning('此資料集目前沒有符合條件的數值型 Target。')
            return
        c1,c2,c3=st.columns([1,2,1])
        with c1:
            target=st.selectbox('Target',possible,index=possible.index('Profit') if 'Profit' in possible else 0)
        with c2:
            options=[c for c in df.columns if c != target]
            features=st.multiselect('Feature Columns',options,default=options)
        with c3:
            test_size=st.slider('Test size',0.15,0.4,0.2,step=.05)
        a,b,c=st.columns(3)
        with a: optimize=st.toggle('Optuna Ridge tuning',value=True)
        with b: feature_select=st.toggle('Forward feature selection',value=True)
        with c: trials=st.slider('Optuna trials',5,60,20,step=5,disabled=not optimize)
        train_clicked=st.button('🚀 Train & Evaluate Model',type='primary',use_container_width=True)
        if train_clicked:
            try:
                X,y,nums,cats=prepare_xy(df,target,features)
                with st.spinner('Model training and cross-validation in progress...'):
                    result=fit_and_evaluate(X,y,nums,cats,test_size=test_size,n_trials=trials,optimize=optimize)
                    selection=forward_selection_cv(result,nums,cats) if feature_select else None
                st.session_state['result']=result
                st.session_state['selection']=selection
                st.session_state['train_sig']=(target,tuple(features),test_size, optimize,trials,feature_select)
                result['target'] = target
                st.success('模型訓練完成！')
            except Exception as exc:
                st.error(f'無法訓練：{exc}')
    result=st.session_state.get('result')
    if result is not None:
        if st.session_state.get('train_sig')!=(target,tuple(features),test_size,optimize,trials,feature_select):
            st.warning('設定已改變，請重新訓練後再預測或下載模型。')
        else:
            visualizations(result,st.session_state.get('selection'))
    else:
        st.markdown('### 3. Modeling')
        st.info('點選 **Train & Evaluate Model**，啟動 LinearRegression 與選用的 Optuna / Feature Selection。')
    with st.expander('🧪 Step 1 Extra — Random Data Scatter Plot', expanded=st.session_state.get('report_mode', False)):
        rng=np.random.default_rng(42)
        x=rng.uniform(0,10,100); y=3*x+10+rng.normal(0,5,100)
        f=go.Figure(go.Scatter(x=x,y=y,mode='markers',marker=dict(color=PURPLE,opacity=.7)))
        st.plotly_chart(chart_style(f,x='Random X',y='Synthetic y'),use_container_width=True)


def stock():
    stock_page(report_mode=st.session_state.get('report_mode', False))


# Screenshot-friendly presentation mode: shows all analysis panels vertically.
with st.sidebar:
    st.header('📸 Export / Screenshot')
    st.toggle('報告／長截圖模式', key='report_mode', help='將所有分析頁籤改成垂直顯示，展開附加圖表，方便截圖或列印 PDF。')
    if st.session_state.get('report_mode'):
        st.info('請先訓練模型；圖表載入完畢後，可用 Chrome F12 → Ctrl+Shift+P → Capture full size screenshot，或 Ctrl+P 儲存 PDF。')

if st.session_state.get('report_mode'):
    st.markdown('''<style>
    /* Only applied in report mode; avoid touching normal application scrolling. */
    [data-testid="stMain"] {overflow: visible !important; height: auto !important; max-height: none !important;}
    [data-testid="stMainBlockContainer"] {max-width: 1050px !important; overflow: visible !important;}
    [data-testid="stAppViewContainer"] {overflow: visible !important;}
    [data-testid="stPlotlyChart"] {overflow: visible !important; break-inside: avoid; page-break-inside: avoid;}
    [data-testid="stMetric"] {break-inside: avoid; page-break-inside: avoid;}
    @media print {
        @page { size: A4 portrait; margin: 11mm; }
        html, body, #root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            height: auto !important; max-height: none !important; overflow: visible !important;
        }
        [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"],
        [data-testid="stBottom"], footer {display: none !important;}
        .block-container {max-width: 100% !important; padding: 0 !important;}
        .hero, [data-testid="stPlotlyChart"], [data-testid="stMetric"] {
            break-inside: avoid; page-break-inside: avoid;
        }
    }
    </style>''', unsafe_allow_html=True)
    st.info('📸 報告模式已啟用：分析圖表改為依序展示。列印前請等待所有圖表渲染完成。')

st.markdown('<div class="brand">📈 &nbsp; ML Regression Explorer <span class="tagline">CRISP-DM</span></div>',unsafe_allow_html=True)
top1,top2=st.columns([3,1])
with top1:
    problem=st.selectbox('CHOOSE PROBLEM',('50 Startups — Multiple Linear Regression','2330.tw — Stock Forecast'),label_visibility='visible')
with top2:
    st.caption('Two-problem navigation  ·  Dataset upload bonus')
if problem.startswith('50 Startups'):
    startup()
else:
    stock()
st.divider()
st.caption('50 Startups CRISP-DM Machine Learning Project  ·  sklearn + Optuna + Streamlit  ·  Educational use')
