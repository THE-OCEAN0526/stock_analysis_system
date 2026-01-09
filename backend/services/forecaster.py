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
            # 準備 Prophet 格式
            data = df.reset_index()
            # 強健性修正：動態獲取第一欄(日期)與 close 欄
            date_col = data.columns[0] 
            data = data[[date_col, 'close']].rename(columns={date_col: 'ds', 'close': 'y'})
            
            # 移除時區資訊，確保與前端及 Plotly 對齊
            data['ds'] = data['ds'].dt.tz_localize(None) 

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
            print(f"Prophet Error: {str(e)}") # 建議列印出具體錯誤
            return []

    @staticmethod
    def predict_arima(df: pd.DataFrame, days: int = 30):
        """ 使用優化後的 SARIMAX 進行預測 """
        try:
            series = df['close'].values
            # 關鍵修正：增加 trend='c' 以捕捉長期趨勢，避免水平線
            # order=(1,1,1) 或 (5,1,0) 皆可，重點在於 trend 參數
            model = SARIMAX(series, order=(1, 1, 1), trend='c', enforce_stationarity=False)
            model_fit = model.fit(disp=False)
            
            forecast_obj = model_fit.get_forecast(steps=days)
            forecast = forecast_obj.predicted_mean
            
            # 產生未來日期
            last_date = df.index[-1]
            if hasattr(last_date, 'tz_localize'):
                last_date = last_date.tz_localize(None)
                
            future_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
            
            result = []
            for date, val in zip(future_dates, forecast):
                result.append({
                    "ds": date.strftime('%Y-%m-%d'),
                    "yhat": float(val)
                })
            return result
        except Exception as e:
            print(f"ARIMA Error: {str(e)}")
            return []