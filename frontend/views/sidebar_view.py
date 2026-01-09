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
        
        st.sidebar.write("**獨立分析子圖**")
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

        with st.sidebar.expander("機器學習分析子圖"):
            ml_reg_sub = st.multiselect("回歸分析 (殘差/誤差圖)", ["線性回歸誤差", "隨機森林誤差"])
            ml_cls_sub = st.multiselect("分類預測 (漲跌機率)", ["明日看漲機率"])
            ml_un_sub = st.multiselect("模式識別 (分群/降維)", ["K-Means 分群狀態", "PCA 特徵成分"])

        ml_subcharts = ml_reg_sub + ml_cls_sub + ml_un_sub

        st.sidebar.markdown("---")

        # 3. 核心預測與分析分類
        # A. 傳統時間序列分析
        ts_modes = st.sidebar.multiselect(
            "傳統時間序列分析",
            ["Prophet 預測", "ARIMA 預測"],
            help="基於數據本身的季節性與週期規律進行統計預測"
        )

        # B. 機器學習模組 (分類收納)
        with st.sidebar.expander("機器學習分析模組"):
            st.write("**監督式學習 (回歸)**")
            ml_reg_modes = st.multiselect(
                "選擇回歸模型 (預測價格)",
                ["線性回歸", "決策樹回歸", "隨機森林回歸"]
            )

            st.write("**監督式學習 (分類)**")
            ml_cls_modes = st.multiselect(
                "選擇分類模型 (預測漲跌)",
                ["邏輯回歸", "SVM 分類"]
            )
            
            st.write("**非監督式學習**")
            ml_un_modes = st.multiselect(
                "模式識別/特徵優化",
                ["K-Means 聚類", "PCA 降維分析"]
            )
        
        # 彙整所有選中的模式
        all_predict_modes = ts_modes + ml_reg_modes + ml_cls_modes + ml_un_modes

            
        return {
            "ticker": ticker,
            "chart_mode": chart_mode,
            "indicators": indicators,
            "perf_indicators": perf_indicators,
            "short_p": st.sidebar.slider("短期均線週期", 5, 50, 10),
            "long_p": st.sidebar.slider("長期均線週期", 20, 100, 50),
            "predict_modes": all_predict_modes,
            "ml_subcharts": ml_subcharts,
            "ml_details": {
                "regression": ml_reg_modes,
                "classification": ml_cls_modes,
                "unsupervised": ml_un_modes
            }
        }