"""Reusable leakage-safe regression workflow for tabular datasets."""
import gzip
import io
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_xy(data: pd.DataFrame, target: str, features: list[str]):
    if target in features or not features:
        raise ValueError('請選擇至少一個特徵，且目標欄不可作為特徵。')
    if len(data) < 15:
        raise ValueError('資料至少需要 15 筆才能執行這份教學流程。')
    frame = data[features + [target]].copy()
    frame[target] = pd.to_numeric(frame[target], errors='coerce')
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    if len(frame) < 15:
        raise ValueError('移除無效目標值後，資料少於 15 筆。')
    numeric, categorical = [], []
    for name in features:
        if pd.api.types.is_numeric_dtype(frame[name]):
            frame[name] = pd.to_numeric(frame[name], errors='coerce').astype(float)
            numeric.append(name)
        else:
            # Text columns are nominal unless the user explicitly converts them before loading.
            frame[name] = frame[name].astype('string').astype(object)
            frame[name] = frame[name].where(pd.notna(frame[name]), np.nan)
            categorical.append(name)
    if any(frame[c].notna().sum() == 0 for c in features):
        raise ValueError('所選特徵包含全為空值的欄位，請移除。')
    return frame[features], frame[target].to_numpy(dtype=float).reshape(-1), numeric, categorical


def make_pipeline(numeric, categorical, kind='linear', alpha=1.0):
    transformers = []
    if numeric:
        transformers.append(('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
        ]), categorical))
    preprocess = ColumnTransformer(transformers)
    regressor = LinearRegression() if kind == 'linear' else Ridge(alpha=alpha)
    return Pipeline([('preprocessor', preprocess), ('regressor', regressor)])


def metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {'MSE': float(mse), 'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mse)), 'R²': float(r2_score(y, pred))}


def fit_and_evaluate(X, y, numeric, categorical, test_size=0.2, seed=42,
                     n_trials=20, optimize=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    n_splits = min(5, max(2, len(X_train) // 5))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    baseline = make_pipeline(numeric, categorical)
    baseline_cv = -cross_val_score(baseline, X_train, y_train,
                                   scoring='neg_mean_squared_error', cv=cv).mean()
    best_alpha = None
    ridge_cv = None
    if optimize:
        import optuna
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-4, 1e4, log=True)
            mdl = make_pipeline(numeric, categorical, 'ridge', alpha)
            score = cross_val_score(mdl, X_train, y_train,
                                    scoring='neg_mean_squared_error', cv=cv)
            return float(-score.mean())
        study = optuna.create_study(direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_alpha, ridge_cv = study.best_params['alpha'], float(study.best_value)
    if ridge_cv is not None and ridge_cv < baseline_cv:
        selected_name = 'Optuna Ridge'
        chosen = make_pipeline(numeric, categorical, 'ridge', best_alpha)
    else:
        selected_name = 'LinearRegression'
        chosen = baseline
    baseline.fit(X_train, y_train)
    baseline_test_pred = baseline.predict(X_test)
    chosen.fit(X_train, y_train)
    pred_train = chosen.predict(X_train)
    pred_test = chosen.predict(X_test)
    return {
        'model': chosen, 'name': selected_name, 'alpha': best_alpha,
        'baseline_cv_mse': float(baseline_cv), 'ridge_cv_mse': ridge_cv,
        'baseline_test': metrics(y_test, baseline_test_pred), 'baseline_pred_test': baseline_test_pred,
        'train': metrics(y_train, pred_train), 'test': metrics(y_test, pred_test),
        'y_test': y_test, 'pred_test': pred_test,
        'X_train': X_train, 'y_train': y_train, 'cv': cv,
        'numeric': numeric, 'categorical': categorical,
    }


def compute_learning_curve(result):
    sizes, train, valid = learning_curve(
        result['model'], result['X_train'], result['y_train'],
        cv=result['cv'], train_sizes=np.linspace(0.5, 1.0, 5),
        scoring='neg_mean_squared_error', shuffle=True, random_state=42)
    return pd.DataFrame({'Training samples': sizes,
                         'Training MSE': -train.mean(axis=1),
                         'Validation MSE': -valid.mean(axis=1)})


def forward_selection_cv(result, numeric, categorical):
    """Rank original input columns using train-only CV RMSE; never test data."""
    X, y, cv = result['X_train'], result['y_train'], result['cv']
    remaining, chosen, rows = list(X.columns), [], []
    while remaining:
        candidates = []
        for col in remaining:
            subset = chosen + [col]
            mdl = make_pipeline([x for x in subset if x in numeric],
                                [x for x in subset if x in categorical])
            cv_mse = -cross_val_score(mdl, X[subset], y,
                scoring='neg_mean_squared_error', cv=cv).mean()
            candidates.append((float(cv_mse), col))
        mse, best = min(candidates)
        chosen.append(best)
        remaining.remove(best)
        rows.append({'Feature Count': len(chosen), 'Selected Features': ', '.join(chosen),
                     'CV RMSE': np.sqrt(mse)})
    return pd.DataFrame(rows)


def model_bytes(model):
    buff = io.BytesIO()
    with gzip.GzipFile(fileobj=buff, mode='wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return buff.getvalue()
