from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')


DATA_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIRECTORY / 'avocado_dataset.csv'
if not DATASET_PATH.is_file():
    DATASET_PATH = DATA_DIRECTORY / 'avocado.csv' / 'avocado.csv'
if not DATASET_PATH.is_file():
    DATASET_PATH = DATA_DIRECTORY / 'avocado.csv'
if not DATASET_PATH.is_file():
    raise FileNotFoundError(f'Avocado CSV not found in {DATA_DIRECTORY}')


def load_series():
    df = pd.read_csv(DATASET_PATH)
    if 'Date' not in df.columns:
        raise ValueError('The dataset does not contain a Date column. Please confirm the CSV structure.')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').copy()

    # Use one region and one type to create a clean monthly time series.
    if 'region' in df.columns:
        df = df[df['region'].eq('Albany')]
    if 'type' in df.columns:
        df = df[df['type'].eq('conventional')]

    monthly = df.resample('MS', on='Date')['Total Volume'].sum().asfreq('MS').fillna(0)
    return monthly


def evaluate_forecast(model_name, forecast, actual):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return {
        'Model': model_name,
        'RMSE': round(float(rmse), 3),
        'MAPE (%)': round(float(mape), 3),
    }


def fit_arma(train_series):
    model = ARIMA(train_series, order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False)
    return model.fit()


def fit_arima(train_series):
    model = ARIMA(train_series, order=(2, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    return model.fit()


def fit_sarima(train_series):
    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12),
                   enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


series = load_series()
print('Monthly data sample:')
print(series.head())

train_size = int(len(series) * 0.8)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

# AutoReg
lags = 12
ar_model = AutoReg(train, lags=lags, trend='t')
ar_fit = ar_model.fit()
ar_forecast = ar_fit.forecast(steps=len(test))

# ARMA
arma_result = fit_arma(train)
arma_forecast = arma_result.forecast(steps=len(test))

# ARIMA
arima_result = fit_arima(train)
arima_forecast = arima_result.forecast(steps=len(test))

# SARIMA
sarima_result = fit_sarima(train)
sarima_forecast = sarima_result.forecast(steps=len(test))

results = [
    evaluate_forecast('Auto Regression', ar_forecast.values, test.values),
    evaluate_forecast('ARMA (1, 0, 1)', arma_forecast.values, test.values),
    evaluate_forecast('ARIMA (2, 1, 1)', arima_forecast.values, test.values),
    evaluate_forecast('SARIMA (1, 1, 1) x (1, 0, 1, 12)', sarima_forecast.values, test.values),
]

print('\nModel comparison:')
for result in results:
    print(result)

# ACF plot for diagnosing seasonal dependence
plot_acf(series, lags=24)
plt.title('Autocorrelation of Monthly Total Volume')
plt.close('all')

# Forecast plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index, train.values, label='Train')
ax.plot(test.index, test.values, label='Actual Test')
ax.plot(test.index, ar_forecast.values, label='AutoReg Forecast', linestyle='--')
ax.set_title('Auto Regression Forecast vs Actual')
ax.legend()
plt.close(fig)
