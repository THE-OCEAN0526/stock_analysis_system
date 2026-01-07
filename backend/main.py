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
        import yfinance as yf
        stock = yf.Ticker(ticker)
        # --- 步驟 A：獲取官方日線數據 (最準確的數據源) ---
        # 抓取 5 天日線來取得「昨日」與「今日」的官方統計
        df_daily = downloader.fetch_data(ticker, period="5d", interval="1d")
        if df_daily.empty:
            return {"status": "error", "message": "No data found"}
        
        df_daily.columns = [col.lower() for col in df_daily.columns]
        df_daily.sort_index(inplace=True)
        
       # 優先嘗試從 info 獲取官方昨收，若失敗則從日線 iloc[-2] 獲取
        try:
            # 備援：有些 Active ETF 的日線數據不穩，info 通常較準
            ref_p = float(stock.info.get('regularMarketPreviousClose', 0))
        except:
            ref_p = 0
            
        if ref_p == 0: # 如果 info 拿不到，才回頭用日線
            ref_p = float(df_daily.iloc[-2]['close']) if len(df_daily) > 1 else float(df_daily.iloc[0]['close'])
        yesterday_vol = int(df_daily.iloc[-2]['volume']) if len(df_daily) > 1 else 0
        
        # 2. 取得今日官方彙總 (倒數第一筆)
        today_row = df_daily.iloc[-1]
        t_open = float(today_row['open'])
        t_high = float(today_row['high'])
        t_low = float(today_row['low'])
        t_close = float(today_row['close'])
        t_vol = int(today_row['volume'])

        # --- 步驟 B：獲取分鐘線 (僅為了計算均價與成交金額) ---
        # 分鐘線雖然總量不準，但其「價格分佈」可以用來估算均價
        summary_df = downloader.fetch_data(ticker, period="1d", interval="1m")
        if not summary_df.empty:
            summary_df.columns = [col.lower() for col in summary_df.columns]
            # 估算均價 (VWAP)
            avg_p = float((summary_df['close'] * summary_df['volume']).sum() / summary_df['volume'].sum()) if summary_df['volume'].sum() > 0 else t_close
        else:
            avg_p = (t_high + t_low + t_close) / 3

        today_stats = {
            "open": t_open,
            "high": t_high,
            "low": t_low,
            "close": t_close,
            "volume": t_vol,
            "yesterday_volume": yesterday_vol,
            # 金額(億) = 官方總量 * 估計均價
            "amount_100m": round(float(t_vol * avg_p / 1e8), 2),
            "avg_price": round(avg_p, 2)
        }

        # --- 步驟 C：獲取圖表數據 ---
        chart_df = downloader.fetch_data(ticker, period=period, interval=interval)
        chart_df.columns = [col.lower() for col in chart_df.columns]
        chart_df.sort_index(inplace=True)

        chart_df = calculator.add_all_indicators(chart_df, short_p, long_p)
        perf = analyzer.calculate_metrics(chart_df)
        # 計算每日報酬率
        returns = chart_df['close'].pct_change()

        # 1. 計算累積報酬率序列 (Cumulative Returns)
        chart_df['cum_ret_series'] = (1 + returns.fillna(0)).cumprod() - 1

        # 2. 計算回撤序列 (Drawdown)
        cum_max = chart_df['close'].cummax()
        chart_df['drawdown_series'] = (chart_df['close'] - cum_max) / cum_max

        # 補上 NaN (第一筆通常是 NaN)
        chart_df = chart_df.fillna(0)

        

        return {
            "status": "success",
            "ticker": ticker,
            "reference_price": ref_p,
            "today_stats": today_stats,
            "performance": perf,
            "data": chart_df.to_dict(orient="index")
        }
    except Exception as e:
        logging.error(f"Backend Error: {str(e)}")
        return {"status": "error", "message": str(e)}