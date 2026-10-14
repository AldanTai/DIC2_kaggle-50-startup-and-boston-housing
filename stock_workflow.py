"""Leakage-conscious lagged-price regression workflow for educational stock forecasting."""
import io
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LAGS = (1, 2, 3, 5, 10)
WINDOWS = (5, 10)


def normalize_prices(data, date_col='Date', close_col='Close'):
    if date_col not in data or close_col not in data:
        raise ValueError('資料需包含日期與收盤價欄位，或在網頁選擇對應的欄位。')
    out = pd.DataFrame({'Date': pd.to_datetime(data[date_col], errors='coerce'),
                        'Close': pd.to_numeric(data[close_col], errors='coerce')})
    out = out.dropna().sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True)
    out = out.loc[out['Close'] > 0].reset_index(drop=True)
    if len(out) < 65:
        raise ValueError('至少需要 65 筆有效、依日期排序的收盤價資料，才能建立 Lag Features 與測試集。')
    return out


def make_supervised(prices):
    close = prices['Close']
    features = pd.DataFrame(index=prices.index)
    for lag in LAGS:
        features[f'lag_{lag}'] = close.shift(lag)
    for window in WINDOWS:
        features[f'ma_{window}'] = close.shift(1).rolling(window).mean()
    features['yesterday_change'] = close.shift(1) - close.shift(2)
    features['Date'] = prices['Date']
    features['Target'] = close
    return features.dropna().reset_index(drop=True)


def train_forecaster(prices, test_fraction=0.2, optimize=False, trials=20):
    supervised = make_supervised(prices)
    fields = [x for x in supervised.columns if x not in ('Date', 'Target')]
    n = len(supervised)
    cut = int(n * (1 - test_fraction))
    if cut < 35 or n-cut < 8:
        raise ValueError('訓練或測試資料量不足。請增加歷史資料或調整測試比例。')
    xtrain, xtest = supervised[fields].iloc[:cut], supervised[fields].iloc[cut:]
    ytrain, ytest = supervised['Target'].iloc[:cut], supervised['Target'].iloc[cut:]
    cv = TimeSeriesSplit(n_splits=3)

    def cv_error(alpha):
        errors = []
        for idx1, idx2 in cv.split(xtrain):
            estimator = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha))])
            estimator.fit(xtrain.iloc[idx1], ytrain.iloc[idx1])
            prediction = estimator.predict(xtrain.iloc[idx2])
            errors.append(mean_squared_error(ytrain.iloc[idx2], prediction))
        return float(np.mean(errors))

    alpha = 1.0
    if optimize:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: cv_error(trial.suggest_float('alpha', 1e-4, 1e4, log=True)), n_trials=trials)
        alpha = study.best_params['alpha']
    cv_mse = cv_error(alpha)
    model = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha))])
    model.fit(xtrain, ytrain)
    pred_train = model.predict(xtrain)
    pred_test = model.predict(xtest)
    # Final deployment model is refitted on all available observations AFTER holdout evaluation.
    deployment_model = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha))])
    deployment_model.fit(supervised[fields], supervised['Target'])
    naive = xtest['lag_1'].to_numpy()
    baseline_rmse = float(np.sqrt(mean_squared_error(ytest, naive)))

    def summary(y, prediction):
        return {'MAE': float(mean_absolute_error(y, prediction)),
                'MSE': float(mean_squared_error(y, prediction)),
                'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
                'R²': float(r2_score(y, prediction))}
    return {'model': model, 'deployment_model': deployment_model, 'features': fields, 'alpha': alpha, 'cv_mse': cv_mse,
            'train': summary(ytrain, pred_train), 'test': summary(ytest, pred_test),
            'naive_rmse': baseline_rmse, 'naive_test': summary(ytest, naive), 'naive_pred_test': naive, 'date_test': supervised['Date'].iloc[cut:].reset_index(drop=True),
            'pred_test': pred_test, 'actual_test': ytest.to_numpy(), 'supervised_rows': n,
            'prices': prices.copy(), 'test_count': len(ytest), 'train_count': len(ytrain)}


def forecast_next(model_result, steps=1):
    """Recursive next-trading-observation forecast (no holiday calendar)."""
    history = model_result['prices']['Close'].astype(float).tolist()
    predicted = []
    for _ in range(steps):
        row = {f'lag_{n}': history[-n] for n in LAGS}
        for n in WINDOWS:
            row[f'ma_{n}'] = float(np.mean(history[-n:]))
        row['yesterday_change'] = history[-1] - history[-2]
        x = pd.DataFrame([row])[model_result['features']]
        next_price = float(model_result['deployment_model'].predict(x)[0])
        history.append(next_price)
        predicted.append(next_price)
    return predicted


def packed_model(model_result):
    obj = {'model': model_result['deployment_model'], 'features': model_result['features'],
           'ticker': '2330.TW', 'lag_features': LAGS, 'rolling_windows': WINDOWS,
           'note': 'Educational one-step close model; recursive estimates are not guaranteed.'}
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()
