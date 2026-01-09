# frontend/app.py
import streamlit as st
import os
from controllers.dashboard_controller import DashboardController
from views.chart_view import ChartView
from views.sidebar_view import SidebarView

# --- 1. 初始化與配置 ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# 時間範圍對應表
PERIOD_OPTIONS = {
    "1天": ("1d", "1m"), "5天": ("5d", "2m"), "1個月": ("1mo", "15m"),
    "6個月": ("6mo", "1d"), "今年": ("ytd", "1d"), "1年": ("1y", "1d"),
    "5年": ("5y", "1d"), "最長": ("max", "1d")
}

# 欄位對照 (用於明細表格)
COLUMN_MAP = {
    "open": "開盤", "high": "最高", "low": "最低", "close": "收盤",
    "volume": "成交量", "sma_s": "短期均線", "ema_s": "短期EMA", "rsi": "RSI",
    "macdh_12_26_9": "柱狀圖", "volatility": "波動率"
}

# 初始化 OOP 物件
controller = DashboardController(BACKEND_URL)
chart_view = ChartView()
sidebar_view = SidebarView()

# 設定頁面
st.set_page_config(page_title="專業股票分析系統", layout="wide")
chart_view.inject_css()

if 'current_period' not in st.session_state:
    st.session_state.current_period = "1年"

# --- 2. 數據獲取 (清單) ---
# 獲取全台股清單 (controller 會自動處理 cache)
stock_options = controller.get_all_stock_options()

# --- 3. UI 渲染 ---
# A. 側邊欄渲染
params = sidebar_view.render_sidebar(stock_options)

# B. 主頁面標題
st.title(f"📈 {params['ticker']} 分析報表")

# C. 時間範圍按鈕列
btn_cols = st.columns(len(PERIOD_OPTIONS))
for i, label in enumerate(PERIOD_OPTIONS.keys()):
    if btn_cols[i].button(
        label, 
        use_container_width=True, 
        type="primary" if st.session_state.current_period == label else "secondary"
    ):
        controller.update_session_period(label)
        st.rerun()

# D. 獲取核心數據
p_val, i_val = PERIOD_OPTIONS[st.session_state.current_period]
res = controller.fetch_stock_data(
    ticker=params['ticker'],
    period=p_val,
    interval=i_val,
    short_p=params['short_p'],
    long_p=params['long_p']
)

# E. 繪製內容區塊
if res:
    col_left, col_right = st.columns([2.2, 1])
    
    with col_left:
        # 繪製主圖表與所有子圖
        chart_view.render_main_chart(
            df=res["df"],
            ref_p=res["reference_price"],
            chart_mode=params["chart_mode"],
            show_indicators=params["indicators"],
            show_perf_indicators=params["perf_indicators"],
            period_label=st.session_state.current_period,
            interval_code=i_val
        )

    with col_right:
        # 渲染右側數據面板與績效卡片
        chart_view.render_statistics_panel(
            stats=res["today_stats"],
            ref_p=res["reference_price"],
            perf=res["performance"],
            df=res["df"],
            show_indicators=params["indicators"]
        )

    # F. 歷史數據表格
    with st.expander("📋 歷史數據明細"):
        st.dataframe(
            res["df"].rename(columns=COLUMN_MAP).sort_index(ascending=False), 
            use_container_width=True
        )
else:
    st.warning("⚠️ 無法取得該股票之數據，請檢查代號是否正確。")