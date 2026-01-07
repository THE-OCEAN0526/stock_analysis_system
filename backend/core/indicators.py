import pandas as pd
import pandas_ta as ta
import numpy as np

class IndicatorCalculator:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, short_p: int = 10, long_p: int = 50) -> pd.DataFrame:
        if df.empty: return df
        df.columns = [col.lower() for col in df.columns]

        # 1. 移動平均線 (動態週期)
        df["sma_s"] = ta.sma(df["close"], length=short_p)
        df["sma_l"] = ta.sma(df["close"], length=long_p)
        df["ema_s"] = ta.ema(df["close"], length=short_p)
        df["ema_l"] = ta.ema(df["close"], length=long_p)

        # 2. 策略訊號判定
        df['sma_sig'] = np.where(df['sma_s'] > df['sma_l'], 1, 0)
        df['sma_pos'] = pd.Series(df['sma_sig']).diff()
        df['ema_sig'] = np.where(df['ema_s'] > df['ema_l'], 1, 0)
        df['ema_pos'] = pd.Series(df['ema_sig']).diff()

        # 3. 其他指標
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        # 4. 波動率
        log_ret = ta.log_return(df["close"])
        df["volatility"] = log_ret.rolling(window=20).std() * (252**0.5)

        return df
