import requests
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional

class DashboardController:
    """
    前端控制器 (Controller)：
    負責處理 API 請求、狀態管理與資料準備。
    """
    def __init__(self, backend_url: str):
        self.backend_url = backend_url

    def fetch_stock_data(
        self, 
        ticker: str, 
        period: str, 
        interval: str, 
        short_p: int, 
        long_p: int,
        predict_modes=[]
    ) -> Optional[Dict[str, Any]]:
        """
        向後端 API 請求分析數據。
        """
        try:
            params = {
                "period": period,
                "interval": interval,
                "short_p": short_p,
                "long_p": long_p,
                "predict_modes": predict_modes
            }
            response = requests.get(
                f"{self.backend_url}/api/v1/stock/{ticker}", 
                params=params, 
                timeout=15
            )
            response.raise_for_status()
            res_json = response.json()
            
            if res_json.get("status") == "success":
                # 將原始 dict 轉換為 DataFrame 方便後續 View 使用
                df = pd.DataFrame.from_dict(res_json["data"], orient="index")
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                res_json["df"] = df
                return res_json
            else:
                st.error(f"後端錯誤: {res_json.get('message')}")
                return None
        except Exception as e:
            st.error(f"📡 無法連接後端或發生通訊錯誤: {str(e)}")
            return None

    def update_session_period(self, label: str):
        """管理 Streamlit 的 Session State"""
        st.session_state.current_period = label
    
    @st.cache_data(ttl=86400) # 快取 24 小時，因為股票清單不會頻繁變動
    def get_all_stock_options(_self):
        """
        向後端獲取完整股票清單
        """
        try:
            res = requests.get(f"{_self.backend_url}/api/v1/stocks/list", timeout=10)
            if res.status_code == 200:
                data = res.json().get("data", [])
                if data:
                    return data
            return ["2330.TW - 台積電", "2317.TW - 鴻海"] # 備援
        except Exception as e:
            print(f"Fetch stock list error: {e}")
            return ["2330.TW - 台積電"]
