from fastapi import FastAPI, Query
from core.downloader import StockDownloader
from core.indicators import IndicatorCalculator
from core.analyzer import PerformanceAnalyzer
import pandas as pd
import logging

app = FastAPI(title="專業股票分析系統 API")
downloader = StockDownloader()
calculator = IndicatorCalculator()
analyzer = PerformanceAnalyzer()

@app.get("/api/v1/stock/{ticker}")
async def get_stock_data(ticker: str, period: str = Query("1y"), interval: str = Query("1d"), short_p: int = 10, long_p: int = 50):
    try:
        df = downloader.fetch_data(ticker, period=period, interval=interval)
        if df is None or df.empty:
            return {"status": "error", "message": "No data found"}

        # 統一欄位為小寫
        df.columns = [col.lower() for col in df.columns]
        
        # 1. 價格類欄位優先填補 (防止 Y 軸縮放失真)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill().bfill()

        # 2. 計算技術指標
        df = calculator.add_all_indicators(df, short_p, long_p)
        
        # 3. 計算量化績效 (已加入 NaN 檢查)
        perf = analyzer.calculate_metrics(df)
        
        # 4. 判定昨收基準 (精確對齊 Yahoo)
        df.sort_index(inplace=True)
        today_date = df.index[-1].date()
        yesterday_df = df[df.index.date < today_date]
        ref_p = yesterday_df.iloc[-1]['close'] if not yesterday_df.empty else df.iloc[0]['close']

        # 5. 剩餘欄位填補 0
        df = df.fillna(0)
        return {
            "status": "success",
            "ticker": ticker,
            "reference_price": float(ref_p),
            "performance": perf,
            "data": df.to_dict(orient="index")
        }
    except Exception as e:
        logging.error(f"Backend Error: {str(e)}")
        return {"status": "error", "message": str(e)}
