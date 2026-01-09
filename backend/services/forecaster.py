import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

class StockForecaster:
    @staticmethod
    def predict_prophet(df: pd.DataFrame, days: int = 30):
        """ 使用 Meta Prophet 進行預測 """
        try:
            # 準備 Prophet 格式
            data = df.reset_index()
            # 強健性修正：動態獲取第一欄(日期)與 close 欄
            date_col = data.columns[0] 
            data = data[[date_col, 'close']].rename(columns={date_col: 'ds', 'close': 'y'})
            
            # 移除時區資訊，確保與前端及 Plotly 對齊
            data['ds'] = data['ds'].dt.tz_localize(None) 

            model = Prophet(
                daily_seasonality=False, 
                weekly_seasonality=True, 
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(data)

            # 建立未來日期並預測
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # 只取未來的段落
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
            return result.to_dict(orient="records")
        except Exception as e:
            print(f"Prophet Error: {str(e)}") # 建議列印出具體錯誤
            return []

    @staticmethod
    def predict_arima(df: pd.DataFrame, days: int = 30):
        """ 使用優化後的 SARIMAX 進行預測 """
        try:
            series = df['close'].values
            # 關鍵修正：增加 trend='c' 以捕捉長期趨勢，避免水平線
            # order=(1,1,1) 或 (5,1,0) 皆可，重點在於 trend 參數
            model = SARIMAX(series, order=(1, 1, 1), trend='c', enforce_stationarity=False)
            model_fit = model.fit(disp=False)
            
            forecast_obj = model_fit.get_forecast(steps=days)
            forecast = forecast_obj.predicted_mean
            
            # 產生未來日期
            last_date = df.index[-1]
            if hasattr(last_date, 'tz_localize'):
                last_date = last_date.tz_localize(None)
                
            future_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
            
            result = []
            for date, val in zip(future_dates, forecast):
                result.append({
                    "ds": date.strftime('%Y-%m-%d'),
                    "yhat": float(val)
                })
            return result
        except Exception as e:
            print(f"ARIMA Error: {str(e)}")
            return []
        
    # --- 監督式學習：回歸 (預測價格) ---
    @staticmethod
    def predict_regression(df: pd.DataFrame, model_type: str, days: int = 30):
        """ 實作監督式學習：回歸模型 (預測價格) """
        try:
            # --- 1. 進階特徵工程：引入滯後特徵 (Lag) ---
            ml_df = df.copy()
            for i in range(1, 4):  # 加入過去 3 天的收盤價作為參考
                ml_df[f'close_lag_{i}'] = ml_df['close'].shift(i)
            
            # 增加時間索引
            ml_df['time_index'] = np.arange(len(ml_df))
            
            # 定義輸入特徵 (移除可能導致漏題的當日特徵)
            features = ['time_index', 'close_lag_1', 'close_lag_2', 'rsi', 'volatility']
            ml_df = ml_df[features + ['close']].dropna()
            
            # --- 2. 目標轉化：預測漲跌幅 (避免絕對值過擬合) ---
            # 我們預測 (明日收盤 - 今日收盤) / 今日收盤
            ml_df['target_ret'] = (ml_df['close'].shift(-1) - ml_df['close']) / ml_df['close']
            train_data = ml_df.dropna()
            
            X = train_data[features]
            y = train_data['target_ret']
            
            # --- 3. 防止過擬合的模型設定 ---
            if model_type == "線性回歸":
                model = Ridge(alpha=1.0) 
            elif model_type == "決策樹回歸":
                # 限制深度 (max_depth) 是防止決策樹過擬合的關鍵
                model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
            else:
                # 隨機森林：限制樹深並增加森林規模
                model = RandomForestRegressor(n_estimators=200, max_depth=7, n_jobs=-1, random_state=42)
            
            # 4. 數據標準化與訓練
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            # --- 5. 滾動未來預測 ---
            history_list = []
            # 產生歷史驗證數據 (將預測的漲跌幅轉回價格)
            y_fit_ret = model.predict(X_scaled)
            # X 包含了訓練時的特徵，我們需要對齊到「目標日」
            for i in range(len(y_fit_ret)):
                # 找出特徵日 (X.index[i]) 的下一個交易日作為「預測目標日」
                # 這樣虛線才會呈現「預測明天」的規律，而不是「背昨天答案」
                try:
                    target_date_idx = ml_df.index.get_loc(X.index[i]) + 1
                    target_date = ml_df.index[target_date_idx]
                except (IndexError, KeyError):
                    continue

                # 基礎價格應該是「特徵日當天」的收盤價
                current_close = ml_df.loc[X.index[i], 'close']
                pred_price = current_close * (1 + y_fit_ret[i]) 
                
                history_list.append({
                    "ds": target_date.strftime('%Y-%m-%d'), 
                    "yhat": float(pred_price)
                })

            # 產生未來預測
            future_list = []
            curr_close = df['close'].iloc[-1]
            last_feats = X.iloc[-1].copy()
            
            for _ in range(days):
                scaled_input = scaler.transform(last_feats.values.reshape(1, -1))
                pred_ret = model.predict(scaled_input)[0]
                next_price = curr_close * (1 + pred_ret)
                
                future_list.append(next_price)
                # 更新下一輪特徵 (簡單模擬)
                curr_close = next_price
                last_feats['time_index'] += 1
                last_feats['close_lag_1'] = next_price # 滾動更新滯後項
            
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
            future_final = [{"ds": d.strftime('%Y-%m-%d'), "yhat": float(p)} for d, p in zip(future_dates, future_list)]

            return {"history": history_list, "future": future_final}
        except Exception as e:
            print(f"Anti-Overfit ML Error: {e}")
            return {"history": [], "future": []}

    # --- 監督式學習：分類 (預測漲跌方向) ---
    @staticmethod
    def predict_classification(df: pd.DataFrame, model_type: str):
        """ 實作監督式學習：分類模型 (預測漲跌訊號) """
        try:
            # 1. 特徵工程
            ml_df = df.copy()
            for i in range(1, 4):
                ml_df[f'close_lag_{i}'] = ml_df['close'].shift(i)
            ml_df['time_index'] = np.arange(len(ml_df))
            features = ['time_index', 'close_lag_1', 'close_lag_2', 'rsi', 'volatility']
            ml_df = ml_df[features + ['close']].dropna()
            
            # 2. 定義標籤 (y)：漲(1), 跌(0)
            ml_df['target_dir'] = (ml_df['close'].shift(-1) > ml_df['close']).astype(int)
            train_data = ml_df.dropna()
            
            X = train_data[features]
            y = train_data['target_dir']
            
            # 3. 訓練模型
            if model_type == "邏輯回歸":
                model = LogisticRegression(max_iter=1000)
            else: # SVM 分類
                model = SVC(probability=True, kernel='rbf', random_state=42)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            # 4. 產生歷史訊號
            y_pred = model.predict(X_scaled)
            y_probs = model.predict_proba(X_scaled)[:, 1] # 看漲機率
            
            up_signals = []
            down_signals = []
            history_probs = []
            
            for i in range(len(y_pred)):
                date_str = X.index[i].strftime('%Y-%m-%d')
                history_probs.append({"ds": date_str, "prob": float(y_probs[i])})
                
                # --- 核心修正處：將預測結果分別存入對應清單 ---
                if y_pred[i] == 1:
                    up_signals.append(date_str)   # 存入漲
                else:
                    down_signals.append(date_str) # 存入跌
            
            # 5. 預測「明日」
            last_feat_scaled = scaler.transform(ml_df[features].tail(1))
            future_prob = float(model.predict_proba(last_feat_scaled)[0, 1])
            future_signal = bool(future_prob > 0.5)

            return {
                "up_signals": up_signals,
                "down_signals": down_signals,
                "history_probs": history_probs,
                "future_signal": future_signal,
                "future_prob": future_prob
            }
        except Exception as e:
            print(f"Classification Error ({model_type}): {e}")
            return {"up_signals": [], "down_signals": [], "history_probs": [], "future_signal": False, "future_prob": 0.5}

    # --- 非監督式學習：K-Means & PCA ---
    @staticmethod
    def analyze_unsupervised(df: pd.DataFrame, mode: str):
        """ 實作非監督式學習：K-Means 市場分群 & PCA 特徵提取 """
        try:
            # 準備特徵數據
            features = ['close', 'rsi', 'volatility']
            data = df[features].dropna()
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            if mode == "K-Means 聚類":
                # 將市場分為 4 個狀態（例如：牛、熊、震盪、極端波動）
                model = KMeans(n_clusters=4, random_state=42, n_init=10)
                clusters = model.fit_predict(data_scaled)
                return {"ds": data.index.strftime('%Y-%m-%d').tolist(), "labels": clusters.tolist()}
            
            elif mode == "PCA 降維分析":
                # 提取第一主成分（代表市場最強的單一綜合指標）
                pca = PCA(n_components=1)
                pca_result = pca.fit_transform(data_scaled)
                return {"ds": data.index.strftime('%Y-%m-%d').tolist(), "values": pca_result.flatten().tolist()}
            
            return None
        except Exception as e:
            print(f"Unsupervised Error: {e}")
            return None