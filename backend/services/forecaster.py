import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

class StockForecaster:
    @staticmethod
    def predict_prophet(df: pd.DataFrame, days: int = 30):
        """ 使用 Meta Prophet 進行預測 """
        try:
            # 準備 Prophet 格式：ds (日期), y (目標數值)
            data = df.reset_index()
            data = data[['index', 'close']].rename(columns={'index': 'ds', 'close': 'y'})
            data['ds'] = data['ds'].dt.tz_localize(None) # 移除時區資訊

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(data)

            # 建立未來日期並預測
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # 只取未來的段落
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
            return result.to_dict(orient="records")
        except Exception as e:
            print(f"Prophet Error: {e}")
            return []

    @staticmethod
    def predict_arima(df: pd.DataFrame, days: int = 30):
        """ 使用 ARIMA 進行預測 """
        try:
            series = df['close'].values
            # 使用簡單的 ARIMA 模型參數 (5,1,0)
            model = ARIMA(series, order=(5, 1, 0))
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=days)
            
            # 產生未來日期
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
            
            result = []
            for date, val in zip(future_dates, forecast):
                result.append({
                    "ds": date.strftime('%Y-%m-%d'),
                    "yhat": float(val)
                })
            return result
        except Exception as e:
            print(f"ARIMA Error: {e}")
            return []