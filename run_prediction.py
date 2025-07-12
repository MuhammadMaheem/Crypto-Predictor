# run_prediction.py

import argparse
import pickle
from crypto_utils import predict_price

def main():
    parser = argparse.ArgumentParser(description="Forecast year-end crypto prices (BTC, ETH, BNB)")
    parser.add_argument("coin", choices=["BTC", "ETH", "BNB"], help="Coin symbol")
    parser.add_argument("year", type=int, help="Target year (e.g. 2026)")

    args = parser.parse_args()

    with open("crypto_price_models.pkl", "rb") as f:
        models = pickle.load(f)

    price = predict_price(models, args.coin, args.year)
    print(f"{args.coin} forecast for {args.year}-12-31: ${price:,.2f}")

if __name__ == "__main__":
    main()
