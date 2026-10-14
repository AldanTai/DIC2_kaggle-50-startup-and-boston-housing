"""Offline smoke test; run from the project root: python verify_project.py"""
import gzip
import pickle
import numpy as np
import pandas as pd
from modeling import prepare_xy, fit_and_evaluate, compute_learning_curve, forward_selection_cv, model_bytes
from stock_workflow import normalize_prices, train_forecaster, forecast_next, packed_model

rng = np.random.default_rng(42)
n = 90
rd = rng.uniform(10000, 140000, n)
marketing = rng.uniform(10000, 260000, n)
administration = rng.uniform(40000, 160000, n)
frame = pd.DataFrame({'R&D Spend':rd, 'Administration':administration, 'Marketing Spend':marketing,
                      'State':rng.choice(['California','Florida','New York'], size=n),
                      'Profit':40000+0.7*rd+0.05*marketing+rng.normal(0,9000,n)})
X,y,nums,cats = prepare_xy(frame, 'Profit', [x for x in frame.columns if x!='Profit'])
result = fit_and_evaluate(X,y,nums,cats,test_size=.2,n_trials=2,optimize=True)
assert np.isfinite(result['test']['RMSE'])
curve = compute_learning_curve(result)
assert len(curve)>0
selected = forward_selection_cv(result,nums,cats)
assert len(selected)>0
restored = pickle.loads(gzip.decompress(model_bytes(result['model'])))
assert np.isfinite(restored.predict(X.iloc[:1])[0])
print('PASS Problem 1: train, Optuna, learning curve, feature selection, pickle')

price = 700 + np.cumsum(rng.normal(.4,7,210))
dates = pd.bdate_range('2024-01-01',periods=210)
prices = normalize_prices(pd.DataFrame({'Date':dates,'Close':price}))
stock_result = train_forecaster(prices,test_fraction=.2,optimize=True,trials=2)
next_values = forecast_next(stock_result,3)
assert len(next_values)==3 and np.isfinite(next_values).all()
stock_bundle = pickle.loads(gzip.decompress(packed_model(stock_result)))
assert 'model' in stock_bundle
print('PASS Problem 2: time-split, Optuna, recursive forecast, pickle')
