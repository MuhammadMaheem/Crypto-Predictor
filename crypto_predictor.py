import pickle
from crypto_predictor import predict_price

with open('crypto_price_models.pkl', 'rb') as f:
    models = pickle.load(f)

# user input
target_year = 2026          # any year > 2021, the data cut‑off
coin        = 'ETH'         # 'BTC', 'ETH', or 'BNB'

price = predict_price(models, coin, target_year)
print(f"Expected {coin} close on {target_year}-12-31 ≈ ${price:,.2f}")
