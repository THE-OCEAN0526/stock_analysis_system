import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
PERIOD_OPTIONS = {
    "1天": ("1d", "1m"), "5天": ("5d", "2m"), "1個月": ("1mo", "15m"),
    "6個月": ("6mo", "1d"), "今年": ("ytd", "1d"), "1年": ("1y", "1d"),
    "5年": ("5y", "1d")
}

st.set_page_config(page_title="iPhone 股市分析系統", layout="wide")
if 'current_period' not in st.session_state: st.session_state.current_period = "1年"

# --- 側邊欄 ---
st.sidebar.header("📊 設定中心")
target_ticker = st.sidebar.selectbox("選擇股票", ["00981A.TW", "2330.TW"])
chart_mode = st.sidebar.radio("樣式", ["走勢圖", "K線圖"])
show_indicators = st.sidebar.multiselect("顯示指標", ["均線", "SMA 交叉策略", "EMA 交叉策略", "RSI", "MACD", "波動率"], default=["均線"])

short_p = st.sidebar.slider("短期均線", 5, 50, 10)
long_p = st.sidebar.slider("長期均線", 20, 100, 50)

# --- API 請求 ---
@st.fragment
def render_ui(ticker):
    # 週期按鈕 (省略部分代碼，邏輯同前...)
    p_val, i_val = PERIOD_OPTIONS[st.session_state.current_period]
    
    params = {"period": p_val, "interval": i_val, "short_p": short_p, "long_p": long_p}
    res = requests.get(f"{BACKEND_URL}/api/v1/stock/{ticker}", params=params).json()

    if res.get("status") == "success":
        df = pd.DataFrame.from_dict(res["data"], orient="index")
        df.index = pd.to_datetime(df.index)
        perf = res["performance"]
        ref_p = res["reference_price"]
        latest_p = df['close'].iloc[-1]

        col_l, col_r = st.columns([2.2, 1])
        
        with col_l:
            # 這裡直接利用 df["sma_s"], df["rsi"] 等後端算好的欄位繪圖
            # 繪圖邏輯與之前相同，但代碼變得很短
            st.write("📈 圖表渲染區域 (已接收後端數據)")
            # 
            
        with col_r:
            diff = latest_p - ref_p
            pct = (diff / ref_p) * 100
            cls = "red" if diff > 0 else "green"
            st.markdown(f'<h1 style="color:{"#eb0f29" if diff>0 else "#008d41"}">{latest_p:.2f}</h1>', unsafe_allow_html=True)
            st.write(f"今日漲跌: {diff:.2f} ({pct:.2f}%)")
            
            # 顯示後端給的績效
            st.metric("累積報酬 (CR)", f"{perf['cum_ret']}%")
            st.metric("夏普值", perf['sharpe'])
            st.metric("最大回撤", f"{perf['mdd']}%")
    else:
        st.error("後端數據格式不正確")

render_ui(target_ticker)
