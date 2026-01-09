import yfinance as yf
import requests
import pandas as pd
from io import StringIO
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
            # 關鍵：設定 auto_adjust=False 以獲取原始市價，而非調整後的淨值
            df = stock.history(period=period, interval=interval, prepost=True, auto_adjust=False)
            if df.empty:
                print(f"警告：找不到 {ticker} 的資料")
            return df
        except Exception as e:
            print(f"抓取 {ticker} 時發生錯誤: {e}")
            return pd.DataFrame()

    def get_taiwan_stock_list(self) -> list:
        """
        抓取台股清單：不使用任何中文欄位名稱，完全依賴欄位位置 (Index)
        """
        try:
            import requests
            from io import StringIO
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
            url_tpex = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
            
            stock_list = []
            
            for url, suffix in [(url_twse, ".TW"), (url_tpex, ".TWO")]:
                res = requests.get(url, headers=headers)
                # 抓取表格，不設定 header，讓 pandas 預設使用數字索引
                dfs = pd.read_html(StringIO(res.text))
                if not dfs: continue
                df = dfs[0]
                
                # --- 核心修正：使用 .iloc[:, 0] 抓取第一欄 ---
                # 這樣就不會發生 KeyError: '有價證券代號及名稱'
                raw_data = df.iloc[:, 0].tolist() 
                
                for item in raw_data:
                    if isinstance(item, str) and "　" in item:
                        parts = item.split("　")
                        if len(parts) >= 2:
                            ticker = parts[0].strip()
                            name = parts[1].strip()
                            # 允許 4-6 位數，且包含字母 (為了支援主動式 ETF)
                            if 4 <= len(ticker) <= 6:
                                stock_list.append(f"{ticker}{suffix} - {name}")
            
            final_result = sorted(list(set(stock_list)))
            print(f"✅ 成功抓取 {len(final_result)} 檔股票")
            return final_result

        except Exception as e:
            # 這裡列印出的 e 就是讓你報錯的原因，現在改用 iloc 後應該就不會出現中文 key 錯誤
            print(f"❌ 抓取股票清單失敗: {e}")
            return ["2330.TW - 台積電", "2317.TW - 鴻海", "2454.TW - 聯發科"]