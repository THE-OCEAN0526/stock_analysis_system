import pandas as pd
import numpy as np
import logging
import yfinance as yf
from typing import Dict, Any, Optional

# 導入核心組件
from services.downloader import StockDownloader
from services.indicators import IndicatorCalculator
from services.analyzer import PerformanceAnalyzer
from services.forecaster import StockForecaster

class StockService:
    """
    股票分析服務類別 (Service Layer)
    負責協調數據抓取、指標計算與績效分析的業務流程。
    """
    def __init__(self):
        self.downloader = StockDownloader()
        self.calculator = IndicatorCalculator()
        self.analyzer = PerformanceAnalyzer()

    def get_full_analysis(
        self, 
        ticker: str, 
        period: str = "1y", 
        interval: str = "1d", 
        short_p: int = 10, 
        long_p: int = 50,
        predict_modes=[]
    ) -> Dict[str, Any]:
        
        """
        執行完整的股票分析流程
        """
        try:
            
            # --- 步驟 A：獲取官方日線數據 (用於統計今日與昨收) ---
            df_daily = self.downloader.fetch_data(ticker, period="5d", interval="1d")
            if df_daily.empty:
                return {"status": "error", "message": f"找不到 {ticker} 的日線資料"}
            
            df_daily.columns = [col.lower() for col in df_daily.columns]
            df_daily.sort_index(inplace=True)
            
            # 獲取參考價 (優先從 info 拿，失敗則用歷史資料)
            ref_p = self._get_reference_price(ticker, df_daily)
            yesterday_vol = int(df_daily.iloc[-2]['volume']) if len(df_daily) > 1 else 0
            
            # 取得今日官方彙總
            today_row = df_daily.iloc[-1]
            t_close = float(today_row['close'])
            t_vol = int(today_row['volume'])

            # --- 步驟 B：獲取分鐘線計算均價 ---
            avg_p = self._calculate_vwap(ticker, t_close)

            today_stats = {
                "open": float(today_row['open']),
                "high": float(today_row['high']),
                "low": float(today_row['low']),
                "close": t_close,
                "volume": t_vol,
                "yesterday_volume": yesterday_vol,
                "amount_100m": round(float(t_vol * avg_p / 1e8), 2), # 成交金額(億)
                "avg_price": round(avg_p, 2)
            }

            # --- 步驟 C：獲取圖表數據與計算指標 ---
            chart_df = self.downloader.fetch_data(ticker, period=period, interval=interval)
            if chart_df.empty:
                 return {"status": "error", "message": "無法取得圖表歷史數據"}
                 
            chart_df.columns = [col.lower() for col in chart_df.columns]
            chart_df.sort_index(inplace=True)

            # 計算技術指標
            chart_df = self.calculator.add_all_indicators(chart_df, short_p, long_p)
            
            # 計算量化績效
            perf = self.analyzer.calculate_metrics(chart_df)
            
            # 計算累積報酬與回撤序列 (用於前端繪圖)
            chart_df = self._add_series_data(chart_df)


            forecast_results = {}
            # 2. 只有在用戶真的想看預測時才進行計算
            if interval == "1d" and len(chart_df) > 30:
                if "Prophet 預測" in predict_modes:
                    forecast_results["prophet"] = StockForecaster.predict_prophet(chart_df)
                if "ARIMA 預測" in predict_modes:
                    forecast_results["arima"] = StockForecaster.predict_arima(chart_df)


            # 處理 JSON 不支援的 NaN 值
            chart_df = chart_df.fillna(0)

            return {
                "status": "success",
                "ticker": ticker,
                "reference_price": ref_p,
                "today_stats": today_stats,
                "performance": perf,
                "data": chart_df.to_dict(orient="index"),
                "forecast": forecast_results # 回傳預測結果
            }

        except Exception as e:
            logging.error(f"StockService Error: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_reference_price(self, ticker: str, df_daily: pd.DataFrame) -> float:
        """嘗試獲取官方昨收價"""
        try:
            stock = yf.Ticker(ticker)
            ref_p = float(stock.info.get('regularMarketPreviousClose', 0))
            if ref_p != 0:
                return ref_p
        except:
            pass
        # 備援方案：從日線資料拿倒數第二筆
        return float(df_daily.iloc[-2]['close']) if len(df_daily) > 1 else float(df_daily.iloc[0]['close'])

    def _calculate_vwap(self, ticker: str, fallback_price: float) -> float:
        """估算當日成交均價 (VWAP)"""
        summary_df = self.downloader.fetch_data(ticker, period="1d", interval="1m")
        if not summary_df.empty:
            summary_df.columns = [col.lower() for col in summary_df.columns]
            total_vol = summary_df['volume'].sum()
            if total_vol > 0:
                return float((summary_df['close'] * summary_df['volume']).sum() / total_vol)
        return fallback_price

    def _add_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算累積報酬率與回撤序列"""
        returns = df['close'].pct_change()
        # 1. 累積報酬序列
        df['cum_ret_series'] = (1 + returns.fillna(0)).cumprod() - 1
        # 2. 回撤序列
        cum_max = df['close'].cummax()
        df['drawdown_series'] = (df['close'] - cum_max) / cum_max
        return df
