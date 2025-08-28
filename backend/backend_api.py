from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timezone, timedelta
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- API KEYS ---
FIXER_API_KEY = os.getenv("FIXER_API_KEY", "YOUR_FIXER_API_KEY")

# --- MODEL PATHS ---
BTC_MODEL_PATH = os.getenv("BTC_MODEL_PATH", "rf_btc_model.pkl")
BTC_SCALER_PATH = os.getenv("BTC_SCALER_PATH", "scaler.pkl")
BTC_ACCURACY_NOTE = "Approximately 60.87%"

USDILS_MODEL_PATH = os.getenv("USDILS_MODEL_PATH", "USDILS_weekly_model_step33.pkl")
USDILS_SCALER_PATH = os.getenv("USDILS_SCALER_PATH", "USDILS_weekly_scaler_step33.pkl")
USDILS_ACCURACY_NOTE = "Approximately 58.49%"

JODILS_MODEL_PATH = os.getenv("JODILS_MODEL_PATH", "USDILS_weekly_model_step33.pkl")
JODILS_SCALER_PATH = os.getenv("JODILS_SCALER_PATH", "USDILS_weekly_scaler_step33.pkl")
JODILS_ACCURACY_NOTE = "Approximately 56.60%"

# --- LOAD MODELS ---
btc_model = joblib.load(BTC_MODEL_PATH)
btc_scaler = joblib.load(BTC_SCALER_PATH)
usdils_model = joblib.load(USDILS_MODEL_PATH)
usdils_scaler = joblib.load(USDILS_SCALER_PATH)

features_list = ['RSI', 'Momentum', 'ATR', 'MACD_Signal', 'Support_Strength', 'Distance_Support']

# =====================
# BTC FUNCTIONS
# =====================
def fetch_btc(days=90):
    try:
        df = yf.download("BTC-USD", period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
        df = df.rename(columns=str.title)[["Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        logging.warning(f"Yahoo Finance failed for BTC-USD: {e}. Falling back to CoinGecko.")
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        dfp = pd.DataFrame(data["prices"], columns=["ts", "Close"])
        dfv = pd.DataFrame(data["total_volumes"], columns=["ts", "Volume"])
        df = pd.merge(dfp, dfv, on="ts", how="inner")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df.set_index("ts")[["Close", "Volume"]].dropna()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def btc_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["RSI14"] = compute_rsi(df["Close"], period=14)
    df["Volatility21"] = df["Return"].rolling(window=21).std()
    latest = df.dropna().iloc[-1]
    return np.array([latest["Close"], latest["RSI14"], latest["Volume"], latest["Volatility21"]], dtype=float)

# =====================
# FOREX FUNCTIONS
# =====================
def calculate_rsi(data, periods=12):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_momentum(data, periods=4):
    return data.pct_change(periods=periods)

def calculate_atr(df, periods=12):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=periods).mean()

def calculate_macd_signal(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd.ewm(span=signal, adjust=False).mean()

def calculate_support_levels(df, timeframe='weekly'):
    window = 20
    threshold = 0.025
    supports = []
    if isinstance(df['Close'], pd.DataFrame):
        close_series = df['Close'].iloc[:, 0]
    else:
        close_series = df['Close']
    prices = pd.to_numeric(close_series, errors='coerce').dropna()
    for i in range(window, len(prices)-window):
        price = float(prices.iloc[i])
        past_prices = prices.iloc[i-window:i+window]
        past_min = float(past_prices.min())
        if price <= past_min*(1+threshold):
            supports.append(price)
    supports = sorted(set([round(s, 4) for s in supports]))
    support_strengths = {
        s: sum(1 for p in prices if abs(p - s)/s <= threshold) / len(prices)
        for s in supports
    }
    return supports, support_strengths

def compute_features(df, timeframe='weekly'):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = calculate_rsi(df['Close'], 12)
    df['Momentum'] = calculate_momentum(df['Close'], 4)
    df['ATR'] = calculate_atr(df, 12)
    df['MACD_Signal'] = calculate_macd_signal(df['Close'])

    supports, support_strengths = calculate_support_levels(df, timeframe)
    if supports:
        support_list = list(supports)
        
        def calculate_distance_support(x):
            try:
                x_val = float(x)
                if x_val != 0:
                    return min((abs(x_val - s) / x_val) for s in support_list)
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        def calculate_support_strength(x):
            try:
                x_val = float(x)
                closest_support = min(support_list, key=lambda s: abs(x_val - s) / x_val if x_val != 0 else float('inf'))
                return support_strengths.get(closest_support, 0.0)
            except (ValueError, TypeError, ZeroDivisionError):
                return 0.0
        
        df['Distance_Support'] = df['Close'].apply(calculate_distance_support)
        df['Support_Strength'] = df['Close'].apply(calculate_support_strength)
    else:
        df['Distance_Support'] = 0.0
        df['Support_Strength'] = 0.0
    
    return df.ffill().bfill()

def forex_features(symbol="USDILS=X"):
    try:
        df = yf.download(symbol, period="5y", interval="1d", auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError(f"No data for {symbol} from Yahoo Finance")
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        df.index = pd.to_datetime(df.index)
        df = compute_features(df)
        latest = df.iloc[-1]
        features = pd.DataFrame([[
            float(latest["RSI"].iloc[0]) if isinstance(latest["RSI"], pd.Series) else float(latest["RSI"]),
            float(latest["Momentum"].iloc[0]) if isinstance(latest["Momentum"], pd.Series) else float(latest["Momentum"]),
            float(latest["ATR"].iloc[0]) if isinstance(latest["ATR"], pd.Series) else float(latest["ATR"]),
            float(latest["MACD_Signal"].iloc[0]) if isinstance(latest["MACD_Signal"], pd.Series) else float(latest["MACD_Signal"]),
            float(latest["Support_Strength"].iloc[0]) if isinstance(latest["Support_Strength"], pd.Series) else float(latest["Support_Strength"]),
            float(latest["Distance_Support"].iloc[0]) if isinstance(latest["Distance_Support"], pd.Series) else float(latest["Distance_Support"])
        ]], columns=features_list)
        return features, df
    except Exception as e:
        logging.warning(f"Yahoo Finance failed for {symbol}: {e}. Falling back to Fixer API.")
        try:
            symbol_clean = symbol.replace("=X", "")
            url = "http://data.fixer.io/api/timeseries"
            params = {
                "access_key": FIXER_API_KEY,
                "start_date": (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "base": symbol_clean[:3],
                "symbols": symbol_clean[3:]
            }
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data.get("success") or "rates" not in data:
                raise ValueError(f"No data for {symbol}")
            rates = data["rates"]
            df = pd.DataFrame({
                "Close": [rates[date][symbol_clean[3:]] for date in rates],
                "Open": [rates[date][symbol_clean[3:]] for date in rates],
                "High": [rates[date][symbol_clean[3:]] for date in rates],
                "Low": [rates[date][symbol_clean[3:]] for date in rates]
            }, index=pd.to_datetime(list(rates.keys())))
            df = df.sort_index().tail(1825)
            df = compute_features(df)
            latest = df.iloc[-1]
            features = pd.DataFrame([[
                float(latest["RSI"].iloc[0]) if isinstance(latest["RSI"], pd.Series) else float(latest["RSI"]),
                float(latest["Momentum"].iloc[0]) if isinstance(latest["Momentum"], pd.Series) else float(latest["Momentum"]),
                float(latest["ATR"].iloc[0]) if isinstance(latest["ATR"], pd.Series) else float(latest["ATR"]),
                float(latest["MACD_Signal"].iloc[0]) if isinstance(latest["MACD_Signal"], pd.Series) else float(latest["MACD_Signal"]),
                float(latest["Support_Strength"].iloc[0]) if isinstance(latest["Support_Strength"], pd.Series) else float(latest["Support_Strength"]),
                float(latest["Distance_Support"].iloc[0]) if isinstance(latest["Distance_Support"], pd.Series) else float(latest["Distance_Support"])
            ]], columns=features_list)
            return features, df
        except Exception as e:
            logging.error(f"Fixer API failed for {symbol}: {e}")
            raise RuntimeError(f"No data for {symbol}")

# =====================
# ROUTES
# =====================
@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/predict")
def predict_btc():
    try:
        features = btc_features(fetch_btc(90))
        features_scaled = btc_scaler.transform(features.reshape(1, -1))
        pred = btc_model.predict(features_scaled)[0]
        label = "Up" if int(pred) == 1 else "Down"
        df = fetch_btc(30)
        current_price = float(df["Close"].iloc[-1])
        predicted_price = current_price * 1.02 if label == "Up" else current_price * 0.98
        return jsonify({
            "status": "success",
            "asset": "BTCUSD",
            "prediction": label,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "accuracy": BTC_ACCURACY_NOTE,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logging.exception("BTC prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_usdils")
def predict_usdils():
    try:
        features, df = forex_features("USDILS=X")
        features_scaled = usdils_scaler.transform(features)
        pred = usdils_model.predict(features_scaled)[0]
        label = "Up" if int(pred) == 1 else "Down"
        current_price = float(df["Close"].iloc[-1])
        predicted_price = current_price * 1.005 if label == "Up" else current_price * 0.995
        return jsonify({
            "status": "success",
            "asset": "USDILS",
            "prediction": label,
            "current_price": round(current_price, 4),
            "predicted_price": round(predicted_price, 4),
            "accuracy": USDILS_ACCURACY_NOTE,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logging.exception("USDILS prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/forex/JODILS", methods=["GET"])
def predict_jodils():
    try:
        usdils_raw = yf.download("USDILS=X", period="6mo", interval="1d", auto_adjust=False, progress=False)
        usdjod_raw = yf.download("USDJOD=X", period="6mo", interval="1d", auto_adjust=False, progress=False)
        
        if isinstance(usdils_raw.columns, pd.MultiIndex):
            usdils_df = usdils_raw.droplevel(1, axis=1)[['Close','High','Low']]
        else:
            usdils_df = usdils_raw[['Close','High','Low']]
            
        if isinstance(usdjod_raw.columns, pd.MultiIndex):
            usdjod_df = usdjod_raw.droplevel(1, axis=1)[['Close','High','Low']]
        else:
            usdjod_df = usdjod_raw[['Close','High','Low']]
        
        if usdils_df.empty or usdjod_df.empty:
            return jsonify({"error": "Data unavailable"}), 500

        common_index = usdils_df.index.intersection(usdjod_df.index)
        usdils_df = usdils_df.loc[common_index]
        usdjod_df = usdjod_df.loc[common_index]
        
        if len(common_index) == 0:
            return jsonify({"error": "No overlapping dates found"}), 500

        jodils_df = pd.DataFrame({
            'Close': usdils_df['Close'].values / usdjod_df['Close'].values,
            'High': usdils_df['High'].values / usdjod_df['High'].values,
            'Low': usdils_df['Low'].values / usdjod_df['Low'].values
        }, index=common_index).dropna()

        if jodils_df.empty:
            return jsonify({"error": "No valid JODILS data after calculations"}), 500

        jodils_weekly = jodils_df.resample('W').agg({'Close':'last','High':'max','Low':'min'}).dropna()
        
        if jodils_weekly.empty:
            return jsonify({"error": "No weekly data available"}), 500
            
        features_df = compute_features(jodils_weekly)
        features_df['ATR'] *= 0.709

        X = features_df[features_list].tail(1)
        
        if X.isnull().any().any():
            return jsonify({"error": "Features contain NaN values"}), 500
            
        X_scaled = usdils_scaler.transform(X)
        pred = int(usdils_model.predict(X_scaled)[0])
        confidence = float(usdils_model.predict_proba(X_scaled)[0].max())
        label = "Up" if pred == 1 else "Down"

        current_price = float(jodils_weekly["Close"].iloc[-1])
        predicted_price = current_price * 1.005 if label == "Up" else current_price * 0.995

        return jsonify({
            'status': 'success',
            'pair': 'JODILS',
            'prediction': label,
            'current_price': round(current_price, 4),
            'predicted_price': round(predicted_price, 4),
            'confidence': round(confidence * 100, 1),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logging.exception("JODILS prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/historical")
def historical_btc():
    try:
        df = fetch_btc(30).tail(10)
        
        # Safe conversion of prices and volumes
        prices = []
        volumes = []
        
        for price in df["Close"].values:
            try:
                prices.append(round(float(price), 2))
            except (ValueError, TypeError, OverflowError):
                prices.append(0.0)
                
        for vol in df["Volume"].values:
            try:
                volumes.append(int(float(vol)))
            except (ValueError, TypeError, OverflowError):
                volumes.append(0)
        
        timestamps = [ts.isoformat() for ts in df.index.to_pydatetime()]
        
        return jsonify({
            "status": "success",
            "asset": "BTCUSD",
            "prices": prices,
            "volumes": volumes,
            "timestamps": timestamps,
            "count": len(prices)
        })
    except Exception as e:
        logging.exception("Historical BTC failed")
        return jsonify({"error": str(e)}), 500

@app.route("/historical_usdils")
def historical_usdils():
    try:
        _, df = forex_features("USDILS=X")
        recent = df.tail(10)
        
        # Safe conversion of prices only (no volume for forex)
        prices = []
        for price in recent["Close"].values:
            try:
                prices.append(round(float(price), 4))
            except (ValueError, TypeError, OverflowError):
                prices.append(0.0)
                
        timestamps = [ts.isoformat() for ts in recent.index.to_pydatetime()]
        
        return jsonify({
            "status": "success",
            "asset": "USDILS",
            "prices": prices,
            "timestamps": timestamps,
            "count": len(prices)
        })
    except Exception as e:
        logging.exception("Historical USDILS failed")
        return jsonify({"error": str(e)}), 500

@app.route("/historical_jodils")
def historical_jodils():
    try:
        usdils_raw = yf.download("USDILS=X", period="1mo", interval="1d", auto_adjust=False, progress=False)
        usdjod_raw = yf.download("USDJOD=X", period="1mo", interval="1d", auto_adjust=False, progress=False)
        
        if isinstance(usdils_raw.columns, pd.MultiIndex):
            usdils_df = usdils_raw.droplevel(1, axis=1)[['Close']]
        else:
            usdils_df = usdils_raw[['Close']]
            
        if isinstance(usdjod_raw.columns, pd.MultiIndex):
            usdjod_df = usdjod_raw.droplevel(1, axis=1)[['Close']]
        else:
            usdjod_df = usdjod_raw[['Close']]
        
        if usdils_df.empty or usdjod_df.empty:
            return jsonify({"error": "Data unavailable"}), 500

        common_index = usdils_df.index.intersection(usdjod_df.index)
        usdils_df = usdils_df.loc[common_index]
        usdjod_df = usdjod_df.loc[common_index]
        
        if len(common_index) == 0:
            return jsonify({"error": "No overlapping dates found"}), 500

        jodils_df = pd.DataFrame({
            'Close': usdils_df['Close'].values / usdjod_df['Close'].values
        }, index=common_index).dropna()

        recent = jodils_df.tail(10)
        
        # Safe conversion of prices only (no volume for forex)
        prices = []
        for price in recent["Close"].values:
            try:
                prices.append(round(float(price), 4))
            except (ValueError, TypeError, OverflowError):
                prices.append(0.0)
        
        timestamps = [ts.isoformat() for ts in recent.index.to_pydatetime()]
        
        return jsonify({
            "status": "success",
            "asset": "JODILS",
            "prices": prices,
            "timestamps": timestamps,
            "count": len(prices)
        })
    except Exception as e:
        logging.exception("Historical JODILS failed")
        return jsonify({"error": str(e)}), 500

@app.route("/info", methods=["GET"])
def market_info():
    try:
        def get_info(symbol, round_digits=4):
            t = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if hist.empty or len(hist) < 1:
                return {"error": f"No data for {symbol}"}

            current = hist["Close"].iloc[-1]
            if len(hist) > 1:
                prev = hist["Close"].iloc[-2]
                change = ((current - prev) / prev) * 100
            else:
                prev = None
                change = None

            info_data = {
                "current_price": round(current, round_digits),
                "24h_change_pct": round(change, 2) if change is not None else 0.0,
                "high": round(hist["High"].iloc[-1], round_digits),
                "low": round(hist["Low"].iloc[-1], round_digits)
            }
            
            # Only add volume for BTC, not for forex pairs
            if symbol == "BTC-USD" and "Volume" in hist and not hist["Volume"].isna().all():
                info_data["volume"] = int(hist["Volume"].iloc[-1])
                
            return info_data

        def get_jodils_info():
            usdils = yf.download("USDILS=X", period="1mo", interval="1d", auto_adjust=False, progress=False)
            usdjod = yf.download("USDJOD=X", period="1mo", interval="1d", auto_adjust=False, progress=False)

            if usdils.empty or usdjod.empty:
                return {"error": "Data unavailable for JODILS"}

            if isinstance(usdils.columns, pd.MultiIndex):
                usdils = usdils.droplevel(1, axis=1)
            if isinstance(usdjod.columns, pd.MultiIndex):
                usdjod = usdjod.droplevel(1, axis=1)

            usdils = usdils.tail(2)
            usdjod = usdjod.tail(2)

            jodils = pd.DataFrame({
                "Close": usdils["Close"].values / usdjod["Close"].values,
                "High": usdils["High"].values / usdjod["High"].values,
                "Low": usdils["Low"].values / usdjod["Low"].values
            }, index=usdils.index)

            current = jodils["Close"].iloc[-1]
            prev = jodils["Close"].iloc[-2] if len(jodils) > 1 else current
            change = ((current - prev) / prev) * 100 if prev != current else 0.0
            high = jodils["High"].max()
            low = jodils["Low"].min()

            return {
                "current_price": round(current, 4),
                "24h_change_pct": round(change, 2),
                "high": round(high, 4),
                "low": round(low, 4)
            }

        return jsonify({
            "BTCUSD": get_info("BTC-USD", round_digits=2),
            "USDILS": get_info("ILS=X", round_digits=4),
            "JODILS": get_jodils_info()
        })

    except Exception as e:
        logging.exception("Market info failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)