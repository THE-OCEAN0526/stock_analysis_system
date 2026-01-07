import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

# --- 1. 基礎配置 ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

COLUMN_MAP = {
    "open": "開盤", "high": "最高", "low": "最低", "close": "收盤",
    "volume": "成交量", "sma_20": "20日均線", "rsi": "RSI",
    "MACD_12_26_9": "MACD", "MACDh_12_26_9": "柱狀圖", "MACDs_12_26_9": "訊號線"
}

PERIOD_OPTIONS = {
    "1天": ("1d", "1m"), "5天": ("5d", "2m"), "1個月": ("1mo", "15m"),
    "6個月": ("6mo", "1d"), "今年": ("ytd", "1d"), "1年": ("1y", "1d"),
    "5年": ("5y", "1d"), "最長": ("max", "1d")
}

st.set_page_config(page_title="iPhone 股市分析系統", layout="wide")

if 'current_period' not in st.session_state:
    st.session_state.current_period = "1年"

def update_period(label):
    st.session_state.current_period = label

# --- 樣式注入 ---
st.markdown("""
<style>
    .yahoo-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        font-size: 15px;
    }
    .label { font-weight: 500; opacity: 0.8; }
    .val { font-weight: 700; text-align: right; }
    .red { color: #eb0f29 !important; }
    .green { color: #008d41 !important; }
    .price-large { font-size: 40px; font-weight: 800; line-height: 1.2; }

    .perf-box {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    .perf-label { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 2px;}
    .perf-value { font-size: 18px; font-weight: 700; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def fetch_chart_data(ticker, period, interval):
    try:
        url = f"{BACKEND_URL}/api/v1/stock/{ticker}?period={period}&interval={interval}"
        res = requests.get(url, timeout=10)
        return res.json()
    except: return None

@st.cache_data(ttl=30)
def fetch_latest_summary(ticker):
    try:
        url = f"{BACKEND_URL}/api/v1/stock/{ticker}?period=5d&interval=1d"
        res = requests.get(url, timeout=10)
        return res.json()
    except: return None

# --- 2. 側邊欄 ---
st.sidebar.header("📊 設定中心")
target_ticker_input = st.sidebar.selectbox("選擇股票", ["00981A.TW", "00982A.TW", "00990A.TW", "00991A.TW", "2330.TW"])

chart_mode = st.sidebar.radio("圖表樣式", ["走勢圖", "K線圖"])

show_indicators = st.sidebar.multiselect(
    "顯示指標", 
    ["均線", "EMA", "SMA 交叉策略", "EMA 交叉策略", "RSI", "MACD", "波動率"], 
    default=["均線"]
)

short_p, long_p = 10, 50
any_ma_selected = any(x in show_indicators for x in ["均線", "EMA", "SMA 交叉策略", "EMA 交叉策略"])
any_strat_selected = any(x in show_indicators for x in ["SMA 交叉策略", "EMA 交叉策略"])

if any_ma_selected:
    st.sidebar.subheader("均線週期設定")
    short_p = st.sidebar.slider("短期均線週期", 5, 50, 10)
    if any_strat_selected:
        long_p = st.sidebar.slider("長期均線週期", 20, 100, 50)

st.title(f"📈 {target_ticker_input} 分析報表")

# --- 3. 繪圖組件 ---
@st.fragment
def render_stock_ui(ticker):
    btn_cols = st.columns(len(PERIOD_OPTIONS))
    for i, label in enumerate(PERIOD_OPTIONS.keys()):
        btn_cols[i].button(label, use_container_width=True, 
                          type="primary" if st.session_state.current_period == label else "secondary",
                          on_click=update_period, args=(label,))

    p_val, i_val = PERIOD_OPTIONS[st.session_state.current_period]
    chart_res = fetch_chart_data(ticker, p_val, i_val)
    summary_res = fetch_latest_summary(ticker)

    if chart_res and summary_res and chart_res.get("status") == "success":
        df_full = pd.DataFrame.from_dict(chart_res["data"], orient="index")
        if df_full.empty: return
        df_full.index = pd.to_datetime(df_full.index)
        df_full.sort_index(inplace=True)

        returns = df_full['close'].pct_change()
        cum_ret = (df_full['close'].iloc[-1] / df_full['close'].iloc[0] - 1) * 100
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        mdd = ((df_full['close'] - df_full['close'].cummax()) / df_full['close'].cummax()).min() * 100
        
        df_chart = df_full.iloc[::2, :].copy() if len(df_full) > 1500 else df_full.copy()
        
        df_chart['sma_s'] = df_chart['close'].rolling(short_p).mean()
        df_chart['sma_l'] = df_chart['close'].rolling(long_p).mean()
        df_chart['sma_sig'] = np.where(df_chart['sma_s'] > df_chart['sma_l'], 1, 0)
        df_chart['sma_pos'] = df_chart['sma_sig'].diff()

        df_chart['ema_s'] = df_chart['close'].ewm(span=short_p, adjust=False).mean()
        df_chart['ema_l'] = df_chart['close'].ewm(span=long_p, adjust=False).mean()
        df_chart['ema_sig'] = np.where(df_chart['ema_s'] > df_chart['ema_l'], 1, 0)
        df_chart['ema_pos'] = df_chart['ema_sig'].diff()

        df_chart['volatility'] = df_chart['close'].pct_change().rolling(20).std() * (252**0.5)

        df_sum = pd.DataFrame.from_dict(summary_res["data"], orient="index")
        df_sum.index = pd.to_datetime(df_sum.index)

        col_left, col_right = st.columns([2.2, 1])

        with col_left:
            has_rsi, has_macd, has_vol = "RSI" in show_indicators, "MACD" in show_indicators, "波動率" in show_indicators
            rows = 1 + has_rsi + has_macd + has_vol
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6]+[0.2]*(rows-1))

            rb = [dict(bounds=["sat", "mon"])]
            if i_val in ["1m", "2m", "5m", "15m", "60m"]: rb.append(dict(bounds=[13.5, 9], pattern="hour"))

            if chart_mode == "走勢圖":
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['close'], fill='tozeroy',
                    fillgradient=dict(type="vertical", colorscale=[[0, 'rgba(217, 48, 37, 0)'], [1, 'rgba(217, 48, 37, 0.4)']]),
                    line=dict(color='#d93025', width=2.5), name="價格",
                    customdata=df_chart['volume'],
                    hovertemplate="<b>%{x|%Y/%m/%d %H:%M}</b><br>價格: %{y:.2f} TWD<br>成交量: %{customdata:,.0f}<extra></extra>"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name="K線"), row=1, col=1)

            if "均線" in show_indicators:
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['sma_s'], line=dict(color='#FF9500', width=1), name=f"{short_p}日均線", hovertemplate=f"{short_p}日均線: %{{y:.2f}}<extra></extra>"), row=1, col=1)

            if "EMA" in show_indicators:
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ema_s'], line=dict(color='#007AFF', width=1, dash='dot'), name=f"{short_p}日EMA", hovertemplate=f"{short_p}日EMA: %{{y:.2f}}<extra></extra>"), row=1, col=1)

            if "SMA 交叉策略" in show_indicators:
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['sma_s'], line=dict(color='#00CCFF', width=1, dash='dash'), name="SMA短線", hovertemplate="SMA短線: %{y:.2f}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['sma_l'], line=dict(color='#FF00FF', width=1, dash='dash'), name="SMA長線", hovertemplate="SMA長線: %{y:.2f}<extra></extra>"), row=1, col=1)
                buy = df_chart[df_chart['sma_pos'] == 1]; sell = df_chart[df_chart['sma_pos'] == -1]
                fig.add_trace(go.Scatter(x=buy.index, y=buy['close'], mode='markers', marker=dict(symbol='triangle-up', size=11, color='#00FF00'), name='SMA買入'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sell.index, y=sell['close'], mode='markers', marker=dict(symbol='triangle-down', size=11, color='#FF3333'), name='SMA賣出'), row=1, col=1)

            if "EMA 交叉策略" in show_indicators:
                buy_e = df_chart[df_chart['ema_pos'] == 1]; sell_e = df_chart[df_chart['ema_pos'] == -1]
                fig.add_trace(go.Scatter(x=buy_e.index, y=buy_e['close'], mode='markers', marker=dict(symbol='star', size=10, color='#00FFCC'), name='EMA買入'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sell_e.index, y=sell_e['close'], mode='markers', marker=dict(symbol='star-triangle-down', size=10, color='#FFCC00'), name='EMA賣出'), row=1, col=1)

            curr_r = 2
            if has_rsi:
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['rsi'], line=dict(color='#AF52DE', width=1.5), name="RSI", hovertemplate="RSI: %{y:.2f}<extra></extra>"), row=curr_r, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=curr_r, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=curr_r, col=1)
                curr_r += 1
                
            if has_macd:
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['MACDh_12_26_9'], name="MACD 柱", marker_color="gray", hovertemplate="柱狀圖: %{y:.4f}<extra></extra>"), row=curr_r, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD_12_26_9'], name="MACD 線", line=dict(color='#1f77b4', width=1.5), hovertemplate="MACD 線: %{y:.4f}<extra></extra>"), row=curr_r, col=1)
                if 'MACDs_12_26_9' in df_chart.columns:
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACDs_12_26_9'], name="訊號線", line=dict(color='#ff7f0e', width=1.5, dash='dot'), hovertemplate="訊號線: %{y:.4f}<extra></extra>"), row=curr_r, col=1)
                curr_r += 1
                
            if has_vol:
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['volatility'], line=dict(color='#FFD700', width=1.5), name="歷史波動率", hovertemplate="年化波動率: %{y:.2%}<extra></extra>"), row=curr_r, col=1)

            y_min, y_max = df_chart['close'].min() - 0.05, df_chart['close'].max() + 0.05
            fig.update_yaxes(range=[y_min, y_max], tickformat=".2f", row=1, col=1, gridcolor='rgba(128,128,128,0.1)')
            fig.update_xaxes(rangebreaks=rb, showspikes=True, spikemode='across', spikedash='dash')
            fig.update_layout(height=450 + (rows*120), template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col_right:
            if len(df_sum) >= 2:
                # 修正：將「1天」模式的基準改回「昨收價」，以符合 Yahoo 市場標準
                if st.session_state.current_period == "1天":
                    latest_p = df_sum.iloc[-1]['close'] # 今日最新價 (1670)
                    ref_p = df_sum.iloc[-2]['close']    # 昨日收盤價 (1705)
                    label_suffix = "今日"
                else:
                    latest_p = df_full.iloc[-1]['close'] 
                    ref_p = df_full.iloc[0]['close']     
                    label_suffix = f"過去{st.session_state.current_period}"

                diff = latest_p - ref_p
                pct = (diff / ref_p) * 100
                cls = "red" if diff > 0 else "green"
                icon = "▲" if diff > 0 else "▼"

                st.markdown(f'<div class="price-large {cls}">{df_full.iloc[-1]["close"]:.2f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 18px; margin-bottom: 15px;" class="{cls}">{icon} {abs(diff):.2f} ({pct:.2f}%) {label_suffix}</div>', unsafe_allow_html=True)

                st.write("🔍 **區間量化績效**")
                p1, p2 = st.columns(2)
                cr_cls = "red" if cum_ret > 0 else "green"
                p1.markdown(f'<div class="perf-box"><div class="perf-label">累計報酬 (CR)</div><div class="perf-value {cr_cls}">{cum_ret:.1f}%</div></div>', unsafe_allow_html=True)
                p1.markdown(f'<div class="perf-box"><div class="perf-label">夏普值 (Sharpe)</div><div class="perf-value">{sharpe:.2f}</div></div>', unsafe_allow_html=True)
                p2.markdown(f'<div class="perf-box"><div class="perf-label">最大回撤 (MDD)</div><div class="perf-value green">{mdd:.1f}%</div></div>', unsafe_allow_html=True)
                
                s_count = int(abs(df_chart["sma_pos"]).sum() if "SMA 交叉策略" in show_indicators else 0)
                s_count += int(abs(df_chart["ema_pos"]).sum() if "EMA 交叉策略" in show_indicators else 0)
                p2.markdown(f'<div class="perf-box"><div class="perf-label">信號次數</div><div class="perf-value">{s_count}</div></div>', unsafe_allow_html=True)

                curr_latest, curr_prev = df_sum.iloc[-1], df_sum.iloc[-2]
                trad_diff = curr_latest['close'] - curr_prev['close']
                trad_cls = "red" if trad_diff > 0 else "green"

                h_data = [
                    ("成交", f"{curr_latest['close']:.2f}", trad_cls), ("昨收", f"{curr_prev['close']:.2f}", ""),
                    ("開盤", f"{curr_latest['open']:.2f}", ""), ("漲跌幅", f"{(trad_diff/curr_prev['close']*100):.2f}%", trad_cls),
                    ("最高", f"{curr_latest['high']:.2f}", "red"), ("漲跌", f"{trad_diff:.2f}", trad_cls),
                    ("最低", f"{curr_latest['low']:.2f}", "green"), ("總量", f"{curr_latest['volume']:,.0f}", ""),
                    ("均價", f"{(curr_latest['open']+curr_latest['high']+curr_latest['low']+curr_latest['close'])/4:.2f}", ""), ("昨量", f"{curr_prev['volume']:,.0f}", ""),
                    ("金額(億)", f"{(curr_latest['volume']*curr_latest['close']/1e8):.2f}", ""), ("振幅", f"{((curr_latest['high']-curr_latest['low'])/curr_prev['close']*100):.2f}%", "")
                ]
                s1, s2 = st.columns(2)
                for i, (label, val, c) in enumerate(h_data):
                    (s1 if i % 2 == 0 else s2).markdown(f'<div class="yahoo-row"><span class="label">{label}</span><span class="val {c}">{val}</span></div>', unsafe_allow_html=True)

                st.write("---")
                st.markdown(f'<div class="yahoo-row"><span class="label">股利</span><span class="val">{curr_latest.get("Dividends", "--")}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="yahoo-row"><span class="label">分割</span><span class="val">{curr_latest.get("Stock Splits", "--")}</span></div>', unsafe_allow_html=True)

        with st.expander("📋 歷史數據明細"):
            st.dataframe(df_full.rename(columns=COLUMN_MAP).sort_index(ascending=False), use_container_width=True)
    else: st.error("⚠️ 數據獲取異常")

render_stock_ui(target_ticker_input)
