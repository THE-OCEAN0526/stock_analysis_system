# backend/core/downloader.py
import yfinance as yf
import pandas as pd
from typing import Optional

class StockDownloader:
    def __init__(self):
        # 預設追蹤的四檔主動式 ETF
        self.active_etfs = ["00981A.TW", "00982A.TW", "00990A.TW", "00991A.TW"]

    def fetch_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        抓取股票歷史資料
        ticker: 股票代號 (例如 00981A.TW)
        period: 時間範圍 (1d, 5d, 1mo, 1y, max)
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval, prepost=True)
            if df.empty:
                print(f"警告：找不到 {ticker} 的資料")
            return df
        except Exception as e:
            print(f"抓取 {ticker} 時發生錯誤: {e}")
            return pd.DataFrame()
