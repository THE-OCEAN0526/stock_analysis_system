from fastapi import FastAPI, HTTPException, Query
from core.downloader import StockDownloader
from core.indicators import IndicatorCalculator
from core.analyzer import PerformanceAnalyzer
import logging

app = FastAPI(title="專業股票分析系統 API")
downloader = StockDownloader()
calculator = IndicatorCalculator()
analyzer = PerformanceAnalyzer()

@app.get("/api/v1/stock/{ticker}")
async def get_stock_data(
    ticker: str,
    period: str = Query("1y"),
    interval: str = Query("1d"),
    short_p: int = Query(10),
    long_p: int = Query(50)
):
    try:
        df = downloader.fetch_data(ticker, period=period, interval=interval)
        if df is None or df.empty:
            return {"status": "error", "message": "No data found"}

        # 1. 計算所有指標
        df = calculator.add_all_indicators(df, short_p, long_p)
        
        # 2. 計算量化績效
        perf = analyzer.calculate_metrics(df)
        
        # 3. 判定市場昨收價 (解決 00981A 同步問題)
        today_date = df.index[-1].date()
        yesterday_df = df[df.index.date < today_date]
        ref_p = yesterday_df.iloc[-1]['close'] if not yesterday_df.empty else df.iloc[0]['close']

        # 4. 轉換為字典
        df = df.fillna(0)
        return {
            "status": "success",
            "ticker": ticker,
            "reference_price": float(ref_p),
            "performance": perf,
            "data": df.to_dict(orient="index")
        }
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
