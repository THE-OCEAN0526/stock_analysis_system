# backend/core/indicators.py
import pandas as pd
import pandas_ta as ta

class IndicatorCalculator:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        # 確保欄位名稱符合 pandas_ta 格式 (小寫)
        df.columns = [col.lower() for col in df.columns]
        
        # 1. 移動平均線 (SMA & EMA)
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["ema_20"] = ta.ema(df["close"], length=20)
        
        # 2. 相對強弱指數 (RSI)
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        # 3. MACD (會產生 MACD, Signal, Histogram 欄位)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # 4. 波動率 (Volatility) - 使用 20 日收益率的標準差
        df["log_return"] = ta.log_return(df["close"])
        df["volatility"] = df["log_return"].rolling(window=20).std() * (252**0.5) # 年化波動率
        
        return df
