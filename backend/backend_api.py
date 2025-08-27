from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timezone
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- MODEL PATHS ---
BTC_MODEL_PATH = os.getenv("BTC_MODEL_PATH", "rf_btc_model.pkl")
BTC_SCALER_PATH = os.getenv("BTC_SCALER_PATH", "scaler.pkl")
BTC_ACCURACY_NOTE = "Approximately 60.87%"

USDILS_MODEL_PATH = os.getenv("USDILS_MODEL_PATH", "USDILS_weekly_model_step33.pkl")
USDILS_SCALER_PATH = os.getenv("USDILS_SCALER_PATH", "USDILS_weekly_scaler_step33.pkl")
USDILS_ACCURACY_NOTE = "Approximately 62.26% (JOD/ILS)"

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
    except Exception:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        r = requests.get(url, params=params, timeout=15)
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
    prices = df['Close']
    for i in range(window, len(prices)-window):
        price = prices.iloc[i]
        past_prices = prices.iloc[i-window:i+window]
        if price <= past_prices.min()*(1+threshold):
            supports.append(price)
    supports = sorted(set([round(s,4) for s in supports]))
    support_strengths = {s: sum(1 for p in prices if abs(p-s)/s <= threshold)/len(prices) for s in supports}
    return supports, support_strengths

def compute_features(df, timeframe='weekly'):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = calculate_rsi(df['Close'], 12)
    df['Momentum'] = calculate_momentum(df['Close'], 4)
    df['ATR'] = calculate_atr(df, 12)
    df['MACD_Signal'] = calculate_macd_signal(df['Close'])
    supports, support_strengths = calculate_support_levels(df, timeframe)
    df['Distance_Support'] = df['Close'].apply(lambda x: min([abs(x-s)/x for s in supports]) if supports else 0.0)
    df['Support_Strength'] = df['Close'].apply(lambda x: support_strengths.get(min(supports, key=lambda s: abs(x-s)/x),0.0) if supports else 0.0)
    return df.ffill().bfill()

def forex_features(symbol="USDILS=X"):
    df = yf.download(symbol, period="5y", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")

    df = compute_features(df)
    latest = df.iloc[-1]
    features = np.array([
        float(latest["RSI"]),
        float(latest["Momentum"]),
        float(latest["ATR"]),
        float(latest["MACD_Signal"]),
        float(latest["Support_Strength"]),
        float(latest["Distance_Support"])
    ], dtype=float)
    return features, df

# =====================
# ROUTES
# =====================
@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/predict")
def predict_btc():
    features = btc_features(fetch_btc(90))
    features_scaled = btc_scaler.transform(features.reshape(1, -1))
    pred = btc_model.predict(features_scaled)[0]
    label = "Up" if int(pred)==1 else "Down"
    return jsonify({"status":"success","asset":"BTCUSD","prediction":label,
                    "timestamp":datetime.now(timezone.utc).isoformat(),
                    "accuracy":BTC_ACCURACY_NOTE})

@app.route("/predict_usdils")
def predict_usdils():
    features, _ = forex_features("USDILS=X")
    features_scaled = usdils_scaler.transform(features.reshape(1, -1))
    pred = usdils_model.predict(features_scaled)[0]
    label = "Up" if int(pred)==1 else "Down"
    return jsonify({"status":"success","asset":"USDILS","prediction":label,
                    "timestamp":datetime.now(timezone.utc).isoformat(),
                    "accuracy":USDILS_ACCURACY_NOTE})

@app.route("/predict/forex/JODILS", methods=["GET"])
def predict_jodils():
    try:
        usdils_df = yf.download("USDILS=X", period="6mo", interval="1d")[['Close','High','Low']]
        usdjod_df = yf.download("USDJOD=X", period="6mo", interval="1d")[['Close','High','Low']]
        if usdils_df.empty or usdjod_df.empty:
            return jsonify({"error":"Data unavailable"}),500

        df = pd.concat([usdils_df.add_suffix('_USDILS'),
                        usdjod_df.add_suffix('_USDJOD')], axis=1)
        jodils_df = pd.DataFrame({
            'Close': df['Close_USDILS']/df['Close_USDJOD'],
            'High': df['High_USDILS']/df['High_USDJOD'],
            'Low': df['Low_USDILS']/df['Low_USDJOD']
        }).dropna()

        jodils_weekly = jodils_df.resample('W').agg({'Close':'last','High':'max','Low':'min'}).dropna()
        features_df = compute_features(jodils_weekly)
        features_df['ATR'] *= 0.709  # scaling for JODILS

        X = features_df[features_list].tail(1)
        X_scaled = usdils_scaler.transform(X)
        pred = int(usdils_model.predict(X_scaled)[0])
        confidence = float(usdils_model.predict_proba(X_scaled)[0].max())
        label = "Up" if pred==1 else "Down"

        return jsonify({'status':'success','pair':'JODILS','prediction':label,'confidence':confidence})

    except Exception as e:
        logging.exception("JODILS prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/historical")
def historical_btc():
    df = fetch_btc(30).tail(10)
    prices = df["Close"].values.tolist()
    timestamps = [ts.isoformat() for ts in df.index.to_pydatetime()]
    return jsonify({"status": "success","asset":"BTCUSD","prices":prices,"timestamps":timestamps,"count":len(prices)})

@app.route("/historical_usdils")
def historical_usdils():
    _, df = forex_features("USDILS=X")
    recent = df.tail(10)
    prices = recent["Close"].values.tolist()
    timestamps = [ts.isoformat() for ts in recent.index.to_pydatetime()]
    return jsonify({"status": "success","asset":"USDILS","prices":prices,"timestamps":timestamps,"count":len(prices)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)