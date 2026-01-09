# 修改後的 sidebar_view.py
import streamlit as st

class SidebarView:
    @staticmethod
    def render_sidebar(stock_options: list):
        # --- 核心修改：移除 with st.sidebar.form(key='...'): ---
        st.sidebar.header("📊 設定中心")
        
        # 1. 搜尋框
        default_index = 0
        default_stock = "2330.TW - 台積電"
        if default_stock in stock_options:
            default_index = stock_options.index(default_stock)
        
        selected_option = st.sidebar.selectbox(
            "搜尋股票 (代號/名稱)", 
            options=stock_options,
            index=default_index
        )
        ticker = selected_option.split(" - ")[0] if " - " in selected_option else selected_option

        st.sidebar.markdown("---")
        
        # 2. 圖表樣式
        chart_mode = st.sidebar.radio("圖表樣式", ["走勢圖", "K線圖"])
        
        # 3. 技術指標
        indicators = st.sidebar.multiselect(
            "顯示指標", 
            ["均線", "EMA", "SMA 交叉策略", "EMA 交叉策略", "RSI", "MACD", "波動率"], 
            default=["均線"]
        )
        
        perf_indicators = st.sidebar.multiselect(
            "顯示績效圖", 
            ["累積報酬", "水下回撤圖"], 
            default=[]
        )

        # 4. 週期設定
        short_p = st.sidebar.slider("短期均線週期", 5, 50, 10)
        long_p = st.sidebar.slider("長期均線週期", 20, 100, 50)

        predict_modes = st.sidebar.multiselect(
            "時間序列預測",
            ["Prophet 預測", "ARIMA 預測"],
            default=[]
        )
            
        return {
            "ticker": ticker,
            "chart_mode": chart_mode,
            "indicators": indicators,
            "perf_indicators": perf_indicators,
            "short_p": short_p,
            "long_p": long_p,
            "predict_modes": predict_modes
        }