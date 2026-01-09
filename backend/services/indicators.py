import pandas as pd
import pandas_ta as ta
import numpy as np

class IndicatorCalculator:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, short_p: int = 10, long_p: int = 50) -> pd.DataFrame:
        if df.empty or len(df) < long_p:
            return df
        
        df.columns = [col.lower() for col in df.columns]

        # 1. 均線與策略
        df["sma_s"] = ta.sma(df["close"], length=short_p)
        df["sma_l"] = ta.sma(df["close"], length=long_p)
        df["ema_s"] = ta.ema(df["close"], length=short_p)
        df["ema_l"] = ta.ema(df["close"], length=long_p)

        # 訊號判定 (0/1)
        df['sma_sig'] = np.where(df['sma_s'] > df['sma_l'], 1, 0)
        df['sma_pos'] = pd.Series(df['sma_sig']).diff()
        df['ema_sig'] = np.where(df['ema_s'] > df['ema_l'], 1, 0)
        df['ema_pos'] = pd.Series(df['ema_sig']).diff()

        # 2. 其他指標
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        # 3. 波動率
        log_ret = ta.log_return(df["close"])
        df["volatility"] = log_ret.rolling(window=20).std() * (252**0.5)

        # 核心修正：強制所有欄位小寫，方便前端讀取
        df.columns = [col.lower() for col in df.columns]
        return df
