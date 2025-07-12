# crypto_utils.py

import pandas as pd

def predict_price(models_dict, coin, target_year):
    from datetime import datetime

    if coin not in models_dict:
        raise ValueError("Invalid coin. Choose from: " + ", ".join(models_dict.keys()))

    model_info = models_dict[coin]
    model = model_info['model']
    last_date = model_info['last_date']

    target_date = pd.to_datetime(f"{target_year}-12-31")
    if target_date <= last_date:
        raise ValueError(f"Target year must be after {last_date.year}")

    days_ahead = (target_date - last_date).days
    forecast = model.forecast(steps=days_ahead)

    return forecast.iloc[-1]
