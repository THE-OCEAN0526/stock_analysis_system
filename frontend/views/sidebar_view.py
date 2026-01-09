import streamlit as st

class SidebarView:
    """
    前端視圖 (View)：
    負責渲染側邊欄控制項並回傳使用者選擇的參數。
    """
    @staticmethod
    def render_sidebar(stock_options: list):
        with st.sidebar.form(key='stock_analysis_form'):
            st.header("📊 設定中心")
            
            # 1. 搜尋框
            default_index = 0
            default_stock = "2330.TW - 台積電"
            if default_stock in stock_options:
                default_index = stock_options.index(default_stock)
            
            selected_option = st.selectbox(
                "搜尋股票 (代號/名稱)", 
                options=stock_options,
                index=default_index
            )
            ticker = selected_option.split(" - ")[0] if " - " in selected_option else selected_option

            st.markdown("---")
            
            # 2. 圖表樣式
            chart_mode = st.radio("圖表樣式", ["走勢圖", "K線圖"])
            
            # 3. 技術指標
            indicators = st.multiselect(
                "顯示指標", 
                ["均線", "EMA", "SMA 交叉策略", "EMA 交叉策略", "RSI", "MACD", "波動率"], 
                default=["均線"]
            )
            
            perf_indicators = st.multiselect(
                "顯示績效圖", 
                ["累積報酬", "水下回撤圖"], 
                default=[]
            )

            # 4. 週期設定
            short_p = st.slider("短期均線週期", 5, 50, 10)
            long_p = st.slider("長期均線週期", 20, 100, 50)

            # 5. 送出按鈕 (表單必須有這個按鈕才能送出)
            # 在表單內按下 Enter 也會觸發這個按鈕
            submit_button = st.form_submit_button(label='🚀 開始分析', use_container_width=True)
            
        return {
            "ticker": ticker,
            "chart_mode": chart_mode,
            "indicators": indicators,
            "perf_indicators": perf_indicators,
            "short_p": short_p,
            "long_p": long_p,
            "submitted": submit_button # 回傳是否按下送出
        }
