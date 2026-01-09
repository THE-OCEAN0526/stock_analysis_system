import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

class StockForecaster:
    @staticmethod
    def predict_prophet(df: pd.DataFrame, days: int = 30):
        """ 使用 Meta Prophet 進行預測 """
        try:
            data = df.reset_index()
            data = data[['index', 'close']].rename(columns={'index': 'ds', 'close': 'y'})
            # 關鍵：移除時區資訊，與前端對齊
            data['ds'] = data['ds'].dt.tz_localize(None) 

            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(data)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
            return result.to_dict(orient="records")
        except Exception as e:
            print(f"Prophet Error: {e}")
            return []

    @staticmethod
    def predict_arima(df: pd.DataFrame, days: int = 30):
        """ 使用優化後的 SARIMAX 進行預測 """
        try:
            series = df['close'].values
            # 使用 (1,1,1) 參數通常比 (5,1,0) 更能反應趨勢，避免水平線
            model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False)
            model_fit = model.fit(disp=False)
            
            forecast_obj = model_fit.get_forecast(steps=days)
            forecast = forecast_obj.predicted_mean
            
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
            
            result = []
            for date, val in zip(future_dates, forecast):
                result.append({"ds": date.strftime('%Y-%m-%d'), "yhat": float(val)})
            return result
        except Exception as e:
            print(f"ARIMA Error: {e}")
            return []