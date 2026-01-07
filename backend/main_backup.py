from fastapi import FastAPI, HTTPException, Query
from core.downloader import StockDownloader
from core.indicators import IndicatorCalculator
import logging

app = FastAPI(title="專業股票分析系統 API")
downloader = StockDownloader()
calculator = IndicatorCalculator()

@app.get("/api/v1/stock/{ticker}")
async def get_stock_data(
    ticker: str, 
    period: str = Query("1y", description="時間範圍"), 
    interval: str = Query("1d", description="資料密度")
):
    try:
        # 1. 下載資料
        df = downloader.fetch_data(ticker, period=period, interval=interval)
        
        if df is None or df.empty:
            # 如果是空資料，回傳 404 而非報錯，讓前端能處理
            return {"ticker": ticker, "data": {}, "status": "no_data"}
        
        # 2. 計算指標
        df_with_indicators = calculator.add_all_indicators(df)
        
        # 3. 處理 NaN 值 (JSON 不支援 NaN)
        df_with_indicators = df_with_indicators.fillna(0)
        
        # 4. 轉換為字典 (使用 isoformat 確保時間格式正確)
        result = {}
        for timestamp, row in df_with_indicators.iterrows():
            result[timestamp.isoformat()] = row.to_dict()
            
        return {
            "ticker": ticker, 
            "period": period, 
            "interval": interval,
            "data": result,
            "status": "success"
        }
    except Exception as e:
        logging.error(f"後端錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
