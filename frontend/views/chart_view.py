# frontend/views/chart_view.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

class ChartView:
    @staticmethod
    def inject_css():
        st.markdown("""
        <style>
            .yahoo-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,0.2); font-size: 15px; }
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

    @staticmethod
    def render_statistics_panel(stats: Dict, ref_p: float, perf: Dict, df: pd.DataFrame, show_indicators: List[str]):
        latest_p = stats.get('close', 0)
        diff = latest_p - ref_p
        pct = (diff / ref_p * 100) if ref_p != 0 else 0
        cls = "red" if diff > 0 else "green"

        st.markdown(f'<div class="price-large {cls}">{latest_p:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 18px; margin-bottom: 15px;" class="{cls}">{"▲" if diff > 0 else "▼"} {abs(diff):.2f} ({pct:.2f}%) 今日</div>', unsafe_allow_html=True)

        st.write("🔍 **區間量化績效**")
        p1, p2 = st.columns(2)
        cr_c = "red" if perf.get('cum_ret', 0) > 0 else "green"
        p1.markdown(f'<div class="perf-box"><div class="perf-label">累積報酬</div><div class="perf-value {cr_c}">{perf.get("cum_ret")}%</div></div>', unsafe_allow_html=True)
        p1.markdown(f'<div class="perf-box"><div class="perf-label">夏普值</div><div class="perf-value">{perf.get("sharpe")}</div></div>', unsafe_allow_html=True)
        p2.markdown(f'<div class="perf-box"><div class="perf-label">最大回撤</div><div class="perf-value green">{perf.get("mdd")}%</div></div>', unsafe_allow_html=True)
        
        # 計算信號次數
        s_count = 0
        if "SMA 交叉策略" in show_indicators and "sma_pos" in df.columns:
            s_count += int(abs(df["sma_pos"]).sum())
        if "EMA 交叉策略" in show_indicators and "ema_pos" in df.columns:
            s_count += int(abs(df["ema_pos"]).sum())
        p2.markdown(f'<div class="perf-box"><div class="perf-label">信號次數</div><div class="perf-value">{s_count}</div></div>', unsafe_allow_html=True)

        h_data = [
            ("成交", f"{latest_p:.2f}", cls), ("昨收", f"{ref_p:.2f}", ""),
            ("開盤", f"{stats.get('open', 0):.2f}", ""), ("漲跌幅", f"{pct:.2f}%", cls),
            ("最高", f"{stats.get('high', 0):.2f}", "red"), ("漲跌", f"{diff:.2f}", cls),
            ("最低", f"{stats.get('low', 0):.2f}", "green"), ("總量(張)", f"{int(stats.get('volume', 0)/1000):,}", ""), 
            ("均價", f"{stats.get('avg_price', 0):.2f}", ""), ("金額(億)", f"{stats.get('amount_100m', 0):.2f}", ""),
            ("昨量(張)", f"{int(stats.get('yesterday_volume', 0)/1000):,}", ""), 
            ("振幅", f"{((stats.get('high', 0) - stats.get('low', 0)) / ref_p * 100 if ref_p != 0 else 0):.2f}%", "")
        ]
        s1, s2 = st.columns(2)
        for i, (label, val, c) in enumerate(h_data):
            (s1 if i % 2 == 0 else s2).markdown(f'<div class="yahoo-row"><span class="label">{label}</span><span class="val {c}">{val}</span></div>', unsafe_allow_html=True)

    @staticmethod
    def render_main_chart(df: pd.DataFrame, ref_p: float, chart_mode: str, show_indicators: List[str], show_perf_indicators: List[str], period_label: str, interval_code: str, forecast_data=None, predict_modes=[]):
        # 1. 判斷需要顯示哪些子圖
        h_rsi = "RSI" in show_indicators
        h_macd = "MACD" in show_indicators
        h_vol = "波動率" in show_indicators
        h_cum = "累積報酬" in show_perf_indicators
        h_dd = "水下回撤圖" in show_perf_indicators
        
        # 動態計算子圖列數與高度
        rows = 1 + h_rsi + h_macd + h_vol + h_cum + h_dd
        main_h = 0.45
        sub_h = (1 - main_h) / (rows - 1) if rows > 1 else 0
        row_heights = [main_h] + [sub_h] * (rows - 1)
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)
        curr_r = 2

        # 2. X 軸範圍與 Y 軸對稱縮放 (針對 1天 模式)
        if period_label == "1天":
            max_diff = max(abs(df['close'].max() - ref_p), abs(df['close'].min() - ref_p))
            padding = max(max_diff, ref_p * 0.005)
            y_range = [ref_p - padding, ref_p + padding]
            # 確保交易時間軸正確
            start_t = df.index[-1].replace(hour=9, minute=0, second=0)
            end_t = df.index[-1].replace(hour=13, minute=30, second=0)
            fig.update_xaxes(range=[start_t, end_t], tickformat="%H:%M")
        else:
            y_margin = (df['close'].max() - df['close'].min()) * 0.1
            y_range = [df['close'].min() - y_margin, df['close'].max() + y_margin]
            fig.update_xaxes(tickformat="%Y/%m/%d")

        # 3. 渲染主圖 (走勢圖 / K線圖)
        if chart_mode == "走勢圖":
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], fill='tozeroy',
                fillgradient=dict(type="vertical", colorscale=[[0, 'rgba(217, 48, 37, 0)'], [1, 'rgba(217, 48, 37, 0.4)']]),
                line=dict(color='#d93025', width=2.5), name="價格", customdata=df['volume'],
                hovertemplate="價格: %{y:.2f}<br>量: %{customdata:,.0f}<extra></extra>"), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(
            x=df.index, 
            open=df['open'], 
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            name="K線",
            increasing_line_color='#eb0f29', # 漲：紅
            decreasing_line_color='#008d41', # 跌：綠
            increasing_fillcolor='#eb0f29',
            decreasing_fillcolor='#008d41',
            # 設定 Hover 資訊
            hovertemplate="開: %{open:.2f}<br>高: %{high:.2f}<br>低: %{low:.2f}<br>收: %{close:.2f}<extra></extra>"
        ), row=1, col=1)

        # 4. 指標疊加 (均線, EMA, 策略訊號)
        if "均線" in show_indicators and "sma_s" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_s'], line=dict(color='#FF9500', width=1.5), name="SMA"), row=1, col=1)
        if "EMA" in show_indicators and "ema_s" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_s'], line=dict(color='#007AFF', width=1.5, dash='dot'), name="EMA"), row=1, col=1)
        
        if "SMA 交叉策略" in show_indicators and "sma_pos" in df.columns:
            b = df[df['sma_pos'] == 1]; s = df[df['sma_pos'] == -1]
            fig.add_trace(go.Scatter(x=b.index, y=b['close'], mode='markers', marker=dict(symbol='triangle-up', size=11, color='#00FF00'), name='SMA買'), row=1, col=1)
            fig.add_trace(go.Scatter(x=s.index, y=s['close'], mode='markers', marker=dict(symbol='triangle-down', size=11, color='#FF3333'), name='SMA賣'), row=1, col=1)
        
        if "EMA 交叉策略" in show_indicators and "ema_pos" in df.columns:
            eb = df[df['ema_pos'] == 1]; es = df[df['ema_pos'] == -1]
            fig.add_trace(go.Scatter(x=eb.index, y=eb['close'], mode='markers', marker=dict(symbol='star', size=10, color='#00FFCC'), name='EMA買'), row=1, col=1)
            fig.add_trace(go.Scatter(x=es.index, y=es['close'], mode='markers', marker=dict(symbol='star-triangle-down', size=10, color='#FFCC00'), name='EMA賣'), row=1, col=1)

        fig.add_hline(y=ref_p, line_dash="dash", line_color="gray", line_width=1, opacity=0.5, row=1, col=1)
        fig.update_yaxes(range=y_range, tickformat=".2f", row=1, col=1, gridcolor='rgba(128,128,128,0.1)')

        # 5. 子圖渲染 (RSI, MACD, 波動率, 績效圖)
        if h_rsi and "rsi" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#AF52DE', width=1.5), name="RSI"), row=curr_r, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=curr_r, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=curr_r, col=1)
            fig.update_yaxes(title_text="RSI", row=curr_r, col=1)
            curr_r += 1

        if h_macd and "macd_12_26_9" in df.columns:
            m_hist = df['macdh_12_26_9']
            m_colors = ['#eb0f29' if v >= 0 else '#008d41' for v in m_hist]
            fig.add_trace(go.Scatter(x=df.index, y=df['macd_12_26_9'], line=dict(color='#007AFF', width=1), name="MACD"), row=curr_r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['macds_12_26_9'], line=dict(color='#FF9500', width=1), name="Signal"), row=curr_r, col=1)
            fig.add_trace(go.Bar(x=df.index, y=m_hist, marker_color=m_colors, name="MACD柱"), row=curr_r, col=1)
            fig.update_yaxes(title_text="MACD", row=curr_r, col=1)
            curr_r += 1

        if h_vol and "volatility" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], line=dict(color='#FFD700'), fill='tozeroy', name="波動率"), row=curr_r, col=1)
            fig.update_yaxes(title_text="波動率", tickformat=".1%", row=curr_r, col=1)
            curr_r += 1

        if h_cum and "cum_ret_series" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['cum_ret_series'], line=dict(color='#00FF00', width=2), fill='tozeroy', name="累積報酬"), row=curr_r, col=1)
            fig.update_yaxes(title_text="累積報酬", tickformat=".1%", row=curr_r, col=1)
            curr_r += 1

        if h_dd and "drawdown_series" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['drawdown_series'], fill='tozeroy', line=dict(color='red', width=1), name="回撤"), row=curr_r, col=1)
            fig.update_yaxes(title_text="回撤%", tickformat=".1%", row=curr_r, col=1)
            curr_r += 1

        # 繪製 Prophet 預測線
        if "Prophet 預測" in predict_modes and forecast_data and "prophet" in forecast_data:
            p_df = pd.DataFrame(forecast_data["prophet"])
            if not p_df.empty and 'ds' in p_df.columns:
                p_df['ds'] = pd.to_datetime(p_df['ds'])
            
            # 畫預測中值
            fig.add_trace(go.Scatter(
                x=p_df['ds'], y=p_df['yhat'],
                line=dict(color='rgba(255, 0, 255, 0.8)', width=2, dash='dot'),
                name="Prophet 趨勢"
            ), row=1, col=1)
            
            # 畫信賴區間 (陰影)
            fig.add_trace(go.Scatter(
                x=pd.concat([p_df['ds'], p_df['ds'][::-1]]),
                y=pd.concat([p_df['yhat_upper'], p_df['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="Prophet 區間"
            ), row=1, col=1)

        # 繪製 ARIMA 預測線
        if "ARIMA 預測" in predict_modes and forecast_data and "arima" in forecast_data:
            a_df = pd.DataFrame(forecast_data["arima"])
            if not a_df.empty and 'ds' in a_df.columns:
                a_df['ds'] = pd.to_datetime(a_df['ds'])
            fig.add_trace(go.Scatter(
                x=a_df['ds'], y=a_df['yhat'],
                line=dict(color='rgba(0, 255, 255, 0.8)', width=2, dash='dash'),
                name="ARIMA 預測"
            ), row=1, col=1)

        # 休市時間處理
        breaks = [dict(bounds=["sat", "mon"])]
        if "m" in interval_code: breaks.append(dict(bounds=[13.5, 9], pattern="hour"))
        fig.update_xaxes(rangebreaks=breaks)

        fig.update_layout(height=500+(rows*100), template="plotly_white", hovermode='x unified', showlegend=False, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
