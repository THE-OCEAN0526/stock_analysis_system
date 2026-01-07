import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(df: pd.DataFrame):
        if df.empty or len(df) < 2:
            return {"cum_ret": 0, "sharpe": 0, "mdd": 0}
        
        close = df['close']
        # 1. 累積報酬 (區間最後一筆 vs 第一筆)
        cum_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
        
        # 2. 夏普值 (Sharpe Ratio)
        returns = close.pct_change().dropna()
        if returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
            
        # 3. 最大回撤 (MDD)
        rolling_max = close.cummax()
        drawdown = (close - rolling_max) / rolling_max
        mdd = drawdown.min() * 100
        
        return {
            "cum_ret": round(float(cum_ret), 2),
            "sharpe": round(float(sharpe), 2),
            "mdd": round(float(mdd), 2)
        }
