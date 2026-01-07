import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(df: pd.DataFrame):
        # 數據不足或年初數據極少時的保護機制
        if df.empty or len(df) < 2:
            return {"cum_ret": 0.0, "sharpe": 0.0, "mdd": 0.0}
        
        close = df['close']
        cum_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
        
        returns = close.pct_change().dropna()
        # 核心修正：防止標準差為 0 導致的 NaN (JSON 格式不支援 NaN)
        if len(returns) > 1 and returns.std() != 0 and np.isfinite(returns.std()):
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        rolling_max = close.cummax()
        drawdown = (close - rolling_max) / rolling_max
        mdd = drawdown.min() * 100
        
        # 安全數值轉換函式
        def safe_float(val):
            return round(float(val), 2) if np.isfinite(val) else 0.0

        return {
            "cum_ret": safe_float(cum_ret),
            "sharpe": safe_float(sharpe),
            "mdd": safe_float(mdd)
        }
