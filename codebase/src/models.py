import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

def train_prophet(df, forecast_days=90):
    """
    Trains a Prophet model on the provided dataframe.
    Expects df with 'ds' and 'y' columns.
    """
    # Adding holidays (Generic US holidays as a proxy for Global Superstore if country not filtered)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='US') 
    model.fit(df)
    
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    return model, forecast

def train_ml_model(df, model_type='rf'):
    """
    Trains a Machine Learning model (Random Forest or XGBoost).
    """
    # Features: date parts + lag features for better time-series performance
    df = df.copy()
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    df['rolling_mean_7'] = df['y'].shift(1).rolling(window=7).mean()
    df = df.dropna()

    features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']
    X = df[features]
    y = df['y']
    
    # Time Series Split for valid evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    
    # Simple walk-forward validation (concept for FYP complexity)
    maes = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
    
    # Final fit on all data
    model.fit(X, y)
    
    return model, np.mean(maes), features

def evaluate_models(y_true, y_pred):
    """
    Returns a dictionary of standard regression metrics.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan,
        "R2": r2_score(y_true, y_pred)
    }

def prepare_time_series_data(df, target_col='sales'):
    """
    Aggregates data by date for time-series models.
    """
    ts_df = df.groupby('order_date')[target_col].sum().reset_index()
    ts_df.columns = ['ds', 'y']
    ts_df = ts_df.sort_values('ds')
    
    # Add features for ML models
    ts_df['year'] = ts_df['ds'].dt.year
    ts_df['month'] = ts_df['ds'].dt.month
    ts_df['day'] = ts_df['ds'].dt.day
    ts_df['day_of_week'] = ts_df['ds'].dt.dayofweek
    ts_df['is_weekend'] = ts_df['day_of_week'].isin([5, 6]).astype(int)
    
    return ts_df

def decompose_series(ts_data):
    """
    Decomposes the time series into Trend, Seasonal, and Residual components.
    """
    df = ts_data.set_index('ds')
    # Use weekly resampling for clearer decomposition signals
    df_weekly = df['y'].resample('W').sum().fillna(0)
    
    # Period 52 for weekly seasonal (1 year)
    if len(df_weekly) > 104: # Need at least 2 cycles
        result = seasonal_decompose(df_weekly, model='additive', period=52)
        return result
    return None
