import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 1. 基礎配置 ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

COLUMN_MAP = {
    "open": "開盤", "high": "最高", "low": "最低", "close": "收盤",
    "volume": "成交量", "sma_s": "短期均線", "ema_s": "短期EMA", "rsi": "RSI",
    "macdh_12_26_9": "柱狀圖", "volatility": "波動率"
}

PERIOD_OPTIONS = {
    "1天": ("1d", "1m"), "5天": ("5d", "2m"), "1個月": ("1mo", "15m"),
    "6個月": ("6mo", "1d"), "今年": ("ytd", "1d"), "1年": ("1y", "1d"),
    "5年": ("5y", "1d"), "最長": ("max", "1d")
}

st.set_page_config(page_title="iPhone 股市分析系統", layout="wide")
if 'current_period' not in st.session_state: st.session_state.current_period = "1年"

def update_period(label): st.session_state.current_period = label

# --- 樣式注入 ---
st.markdown("""
<style>
    .yahoo-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(128, 128, 128, 0.2); font-size: 15px; }
    .label { font-weight: 500; opacity: 0.8; }
    .val { font-weight: 700; text-align: right; }
    .red { color: #eb0f29 !important; }
    .green { color: #008d41 !important; }
    .price-large { font-size: 40px; font-weight: 800; line-height: 1.2; }
    .perf-box { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px; margin-bottom: 8px; }
    .perf-label { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 2px;}
    .perf-value { font-size: 18px; font-weight: 700; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- 2. 側邊欄控制 ---
st.sidebar.header("📊 設定中心")
target_ticker = st.sidebar.selectbox("選擇股票", ["00981A.TW", "00982A.TW", "00990A.TW", "00991A.TW", "2330.TW"])
chart_mode = st.sidebar.radio("圖表樣式", ["走勢圖", "K線圖"])
show_indicators = st.sidebar.multiselect("顯示指標", ["均線", "EMA", "SMA 交叉策略", "EMA 交叉策略", "RSI", "MACD", "波動率"], default=["均線"])
show_perf_indicators = st.sidebar.multiselect("顯示績效圖", ["累積報酬", "水下回撤圖"], default=[])

short_p, long_p = 10, 50
if any(x in show_indicators for x in ["均線", "EMA", "SMA 交叉策略", "EMA 交叉策略"]):
    short_p = st.sidebar.slider("短期週期", 5, 50, 10)
    if any(x in show_indicators for x in ["SMA 交叉策略", "EMA 交叉策略"]):
        long_p = st.sidebar.slider("長期週期", 20, 100, 50)

# --- 3. 渲染組件 ---
@st.fragment
def render_stock_ui(ticker):
    btn_cols = st.columns(len(PERIOD_OPTIONS))
    for i, label in enumerate(PERIOD_OPTIONS.keys()):
        btn_cols[i].button(label, use_container_width=True, 
                          type="primary" if st.session_state.current_period == label else "secondary",
                          on_click=update_period, args=(label,))

    p_val, i_val = PERIOD_OPTIONS[st.session_state.current_period]
    try:
        res = requests.get(f"{BACKEND_URL}/api/v1/stock/{ticker}", 
                           params={"period": p_val, "interval": i_val, "short_p": short_p, "long_p": long_p}, timeout=15).json()
    except: st.error("📡 無法連接後端"); return

    if res.get("status") == "success":
        df = pd.DataFrame.from_dict(res["data"], orient="index")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # 2. 獲取核心統計 (此部分不受時間按紐切換影響)
        stats = res.get("today_stats", {})
        ref_p = res.get("reference_price", 0)
        perf = res.get("performance", {})

        # 3. 計算即時漲跌 (以 today_stats 為主，確保與 Yahoo 一致)
        latest_p = stats.get('close', 0)
        diff = latest_p - ref_p
        pct = (diff / ref_p * 100) if ref_p != 0 else 0
        cls = "red" if diff > 0 else "green"

        col_left, col_right = st.columns([2.2, 1])

        with col_left:
            # 1. 子圖配置
            h_rsi, h_macd, h_vol = "RSI" in show_indicators, "MACD" in show_indicators, "波動率" in show_indicators
            h_cum, h_dd = "累積報酬" in show_perf_indicators, "水下回撤圖" in show_perf_indicators
            
            rows = 1 + h_rsi + h_macd + h_vol + h_cum + h_dd
            # 動態分配高度：主圖佔 40%-50%，其餘平分
            main_h = 0.4
            sub_h = (1 - main_h) / (rows - 1) if rows > 1 else 0
            row_heights = [main_h] + [sub_h] * (rows - 1)
            
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)

            # --- 初始化子圖行數計數器 ---
            curr_r = 2

            # 2. 動態計算 Y 軸與 X 軸範圍
            if st.session_state.current_period == "1天":
                # 1天模式：對稱縮放
                max_diff = max(abs(df['close'].max() - ref_p), abs(df['close'].min() - ref_p))
                padding = max(max_diff, ref_p * 0.005) 
                y_range = [ref_p - padding, ref_p + padding]
                
                # 設定當日交易時間範圍
                start_t = df.index[-1].replace(hour=9, minute=0, second=0)
                end_t = df.index[-1].replace(hour=13, minute=30, second=0)
                fig.update_xaxes(range=[start_t, end_t], tickformat="%H:%M")
            else:
                # 其他模式：動態縮放
                y_margin = (df['close'].max() - df['close'].min()) * 0.1
                y_range = [df['close'].min() - y_margin, df['close'].max() + y_margin]
                fig.update_xaxes(tickformat="%Y/%m/%d")
            
            # 3. 新增累積報酬圖
            if h_cum:
                fig.add_trace(go.Scatter(x=df.index, y=df['cum_ret_series'], 
                                        line=dict(color='#00FF00', width=2), 
                                        fill='tozeroy', name="累積報酬"), row=curr_r, col=1)
                fig.update_yaxes(title_text="報酬%", tickformat=".1%", row=curr_r, col=1)
                curr_r += 1

            # 4. 新增水下回撤圖 (專業投資者最看重)
            if h_dd:
                fig.add_trace(go.Scatter(x=df.index, y=df['drawdown_series'], 
                                        fill='tozeroy', line=dict(color='red', width=1), 
                                        name="回撤"), row=curr_r, col=1)
                fig.update_yaxes(title_text="回撤%", tickformat=".1%", row=curr_r, col=1)
                curr_r += 1

            # 3. 套用 Y 軸與基礎網格
            fig.update_yaxes(range=y_range, tickformat=".2f", row=1, col=1, gridcolor='rgba(128,128,128,0.1)')
            fig.add_hline(y=ref_p, line_dash="dash", line_color="gray", line_width=1, opacity=0.5, row=1, col=1)

            # 4. 【關鍵修復】動態處理 Rangebreaks
            # 永遠隱藏週末
            breaks = [dict(bounds=["sat", "mon"])]
            
            # 只有在「非日線」資料（分鐘級資料）時，才隱藏非交易時段
            # 判斷方式：檢查 i_val 是否包含 'm' (分鐘)
            if "m" in i_val:
                breaks.append(dict(bounds=[13.5, 9], pattern="hour"))
            
            fig.update_xaxes(rangebreaks=breaks)
            

            # 主圖表渲染
            if chart_mode == "走勢圖":
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], fill='tozeroy',
                    fillgradient=dict(type="vertical", colorscale=[[0, 'rgba(217, 48, 37, 0)'], [1, 'rgba(217, 48, 37, 0.4)']]),
                    line=dict(color='#d93025', width=2.5), name="價格", customdata=df['volume'],
                    hovertemplate="時間: %{x|%H:%M}<br>價格: %{y:.2f} TWD<br>成交量: %{customdata:,.0f}<extra></extra>"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)

            # 指標線條與策略訊號
            if "均線" in show_indicators: fig.add_trace(go.Scatter(x=df.index, y=df['sma_s'], line=dict(color='#FF9500', width=1.5), name="SMA"), row=1, col=1)
            if "EMA" in show_indicators: fig.add_trace(go.Scatter(x=df.index, y=df['ema_s'], line=dict(color='#007AFF', width=1.5, dash='dot'), name="EMA"), row=1, col=1)
            
            if "SMA 交叉策略" in show_indicators:
                b = df[df['sma_pos'] == 1]; s = df[df['sma_pos'] == -1]
                fig.add_trace(go.Scatter(x=b.index, y=b['close'], mode='markers', marker=dict(symbol='triangle-up', size=11, color='#00FF00'), name='SMA買'), row=1, col=1)
                fig.add_trace(go.Scatter(x=s.index, y=s['close'], mode='markers', marker=dict(symbol='triangle-down', size=11, color='#FF3333'), name='SMA賣'), row=1, col=1)
            
            if "EMA 交叉策略" in show_indicators:
                eb = df[df['ema_pos'] == 1]; es = df[df['ema_pos'] == -1]
                fig.add_trace(go.Scatter(x=eb.index, y=eb['close'], mode='markers', marker=dict(symbol='star', size=10, color='#00FFCC'), name='EMA買'), row=1, col=1)
                fig.add_trace(go.Scatter(x=es.index, y=es['close'], mode='markers', marker=dict(symbol='star-triangle-down', size=10, color='#FFCC00'), name='EMA賣'), row=1, col=1)

            # 子圖處理
            
            if h_rsi:
                fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#AF52DE', width=1.5), name="RSI", hovertemplate="RSI: %{y:.2f}<extra></extra>"), row=curr_r, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=curr_r, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=curr_r, col=1)
                curr_r += 1

            if h_macd:
                # A. 畫 MACD 線 (通常是藍色)
                m_hist = df['macdh_12_26_9']
                m_colors = ['#eb0f29' if v >= 0 else '#008d41' for v in m_hist]
                
                # 畫 MACD 線與訊號線
                fig.add_trace(go.Scatter(x=df.index, y=df['macd_12_26_9'], line=dict(color='#007AFF', width=1), name="MACD"), row=curr_r, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['macds_12_26_9'], line=dict(color='#FF9500', width=1), name="Signal"), row=curr_r, col=1)
                # 畫柱狀圖
                fig.add_trace(go.Bar(x=df.index, y=m_hist, marker_color=m_colors, name="MACD柱", opacity=0.7), row=curr_r, col=1)
                fig.add_hline(y=0, line_color="gray", line_width=0.5, row=curr_r, col=1)
                curr_r += 1

            if h_vol:
                # 波動率繪圖...
                fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], line=dict(color='#FFD700'), name="波動率"), row=curr_r, col=1)
                curr_r += 1

            fig.update_layout(height=500+(rows*100), template="plotly_white", hovermode='x unified', showlegend=False, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col_right:
            # 頂部價格顯示
            st.markdown(f'<div class="price-large {cls}">{latest_p:.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 18px; margin-bottom: 15px;" class="{cls}">{"▲" if diff > 0 else "▼"} {abs(diff):.2f} ({pct:.2f}%) 今日</div>', unsafe_allow_html=True)

            # 區間量化績效
            st.write("🔍 **區間量化績效**")
            p1, p2 = st.columns(2)
            cr_c = "red" if perf['cum_ret'] > 0 else "green"
            p1.markdown(f'<div class="perf-box"><div class="perf-label">累積報酬</div><div class="perf-value {cr_c}">{perf["cum_ret"]}%</div></div>', unsafe_allow_html=True)
            p1.markdown(f'<div class="perf-box"><div class="perf-label">夏普值</div><div class="perf-value">{perf["sharpe"]}</div></div>', unsafe_allow_html=True)
            p2.markdown(f'<div class="perf-box"><div class="perf-label">最大回撤</div><div class="perf-value green">{perf["mdd"]}%</div></div>', unsafe_allow_html=True)
            
            s_count = int(abs(df["sma_pos"]).sum() if "SMA 交叉策略" in show_indicators else 0)
            s_count += int(abs(df["ema_pos"]).sum() if "EMA 交叉策略" in show_indicators else 0)
            p2.markdown(f'<div class="perf-box"><div class="perf-label">信號次數</div><div class="perf-value">{s_count}</div></div>', unsafe_allow_html=True)

            # --- 完整的 12 項指標 (還原自 app_backup.py) ---
            h_data = [
                ("成交", f"{latest_p:.2f}", cls), 
                ("昨收", f"{ref_p:.2f}", ""),
                ("開盤", f"{stats.get('open', 0):.2f}", ""), 
                ("漲跌幅", f"{pct:.2f}%", cls),
                ("最高", f"{stats.get('high', 0):.2f}", "red"), 
                ("漲跌", f"{diff:.2f}", cls),
                ("最低", f"{stats.get('low', 0):.2f}", "green"), 
                ("總量(張)", f"{int(stats.get('volume', 0)/1000):,}", ""), 
                ("均價", f"{stats.get('avg_price', 0):.2f}", ""), 
                ("金額(億)", f"{stats.get('amount_100m', 0):.2f}", ""),
                ("昨量(張)", f"{int(stats.get('yesterday_volume', 0)/1000):,}", ""), 
                ("振幅", f"{((stats.get('high', 0) - stats.get('low', 0)) / ref_p * 100 if ref_p != 0 else 0):.2f}%", "")
            ]
            s1, s2 = st.columns(2)
            for i, (label, val, c) in enumerate(h_data):
                (s1 if i % 2 == 0 else s2).markdown(f'<div class="yahoo-row"><span class="label">{label}</span><span class="val {c}">{val}</span></div>', unsafe_allow_html=True)

        # 歷史數據明細
        with st.expander("📋 歷史數據明細"):
            st.dataframe(df.rename(columns=COLUMN_MAP).sort_index(ascending=False), use_container_width=True)

    else: st.error(f"❌ 後端報錯: {res.get('message')}")

render_stock_ui(target_ticker)