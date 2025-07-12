# app.py
# Streamlit GUI for Crypto Year‑End Price Predictor
import streamlit as st
import pandas as pd
import pickle, os
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

################################################################################
# ── CONSTANTS ──────────────────────────────────────────────────────────────────
################################################################################
DATA_DIR   = Path("data")                  # where your CSVs live
PKL_FILE   = Path("crypto_price_models.pkl")
DEFAULT_COINS = {
    "BTC": "coin_Bitcoin.csv",
    "ETH": "coin_Ethereum.csv",
    "BNB": "coin_BinanceCoin.csv",
}

################################################################################
# ── HELPERS ────────────────────────────────────────────────────────────────────
################################################################################
def load_models():
    """Load trained models if pickle exists, else return empty dict."""
    if PKL_FILE.exists():
        with open(PKL_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_models(models: dict):
    with open(PKL_FILE, "wb") as f:
        pickle.dump(models, f)

def clean_series(df: pd.DataFrame) -> pd.Series:
    df = df[['Date', 'Close']].drop_duplicates().dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    ser = df.set_index('Date').sort_index()['Close'].resample('D').ffill()
    return ser

def train_single_series(series: pd.Series):
    return ExponentialSmoothing(series, trend='add', seasonal=None).fit()

def retrain_models(files_map: dict) -> dict:
    """files_map = {'BTC': Path(...), ...}"""
    new_models = {}
    for coin, csv_path in files_map.items():
        ser = clean_series(pd.read_csv(csv_path))
        model = train_single_series(ser)
        new_models[coin] = {"model": model, "last_date": ser.index[-1]}
    save_models(new_models)
    return new_models

def predict_price(models_dict, coin, year):
    if coin not in models_dict:
        raise ValueError("Coin not found. Train or pick another.")
    info  = models_dict[coin]
    model, last = info["model"], info["last_date"]
    target_date = pd.to_datetime(f"{year}-12-31")
    if target_date <= last:
        raise ValueError(f"Year must be after training data ({last.year}).")
    days = (target_date - last).days
    return model.forecast(days).iloc[-1]

################################################################################
# ── SIDEBAR ────────────────────────────────────────────────────────────────────
################################################################################
st.sidebar.title("⚙️ Options")
page = st.sidebar.radio("Choose screen", ["Predict", "Retrain"])

################################################################################
# ── MAIN PAGES ────────────────────────────────────────────────────────────────
################################################################################
if page == "Predict":
    st.title("🪙 Crypto Price Predictor")
    st.write("Forecast **year‑end closing price** for Bitcoin (BTC), Ethereum (ETH),"
             " or Binance Coin (BNB) using a Holt‑Winters trend model.")

    models = load_models()
    if not models:
        st.warning("No models found. Please visit the **Retrain** tab first.")
        st.stop()

    coin = st.selectbox("Select Coin", list(models.keys()))
    year = st.number_input("Target Year", min_value=datetime.now().year + 1,
                           max_value=2035, value=datetime.now().year + 1, step=1)

    if st.button("Predict"):
        try:
            price = predict_price(models, coin, int(year))
            st.success(f"**{coin}** expected close on **{year}-12-31** → "
                       f"**${price:,.2f}**")
        except Exception as e:
            st.error(str(e))

        # Optional: quick glimpse of historical data
        hist = models[coin]["model"].fittedvalues
        fig, ax = plt.subplots()
        hist.plot(ax=ax, label="Historical")
        ax.set_title(f"{coin} Historical Close (fitted)")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

# ──────────────────────────────────────────────────────────────────────────────
elif page == "Retrain":
    st.title("🔄 Retrain / Update Models")
    st.write("Upload **CSV files** with `Date` and `Close` columns. "
             "You can replace one or all coins. "
             "Leave blank to keep the default sample datasets.")

    uploaded = {}
    for coin in DEFAULT_COINS:
        file = st.file_uploader(f"Upload {coin} CSV", type="csv", key=coin)
        if file:
            path = DATA_DIR / f"user_{coin}.csv"
            path.parent.mkdir(exist_ok=True)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            uploaded[coin] = path
    # merge defaults with any user uploads
    file_map = {c: (uploaded.get(c) or DATA_DIR/DEFAULT_COINS[c]) for c in DEFAULT_COINS}

    if st.button("Start Training"):
        with st.spinner("Training models..."):
            new_models = retrain_models(file_map)
        st.success("✅ Training complete. You may now switch to **Predict**.")
