# train_predictor.py

import pandas as pd
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path

DATA_FILES = {
    'BTC': 'coin_Bitcoin.csv',
    'ETH': 'coin_Ethereum.csv',
    'BNB': 'coin_BinanceCoin.csv'
}

models = {}

for coin, file in DATA_FILES.items():
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']].drop_duplicates().dropna()
    df = df.set_index('Date').resample('D').ffill()

    model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None).fit()
    models[coin] = {
        'model': model,
        'last_date': df.index[-1]
    }

with open('crypto_price_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("✅ Models trained and saved to crypto_price_models.pkl")
