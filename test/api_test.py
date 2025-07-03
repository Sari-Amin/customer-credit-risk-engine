import requests

sample_data = {
    "total_amount": 10000,
    "avg_amount": 1000,
    "std_amount": 500,
    "txn_count": 10,
    "fraud_txn_count": 0,
    "fraud_ratio": 0.0,
    "txn_hour": 13,
    "txn_day": 15,
    "txn_month": 6,
    "txn_weekday": 2,
    "ChannelId_3": 1,
    "ChannelId_2": 0,
    "ProductCategory_airtime": 1,
    "ProductCategory_financial_services": 0,
    "ProductCategory_other": 0,
    "ProductCategory_ticket": 0,
    "ProductCategory_tv": 0,
    "ProductCategory_utility_bill": 0,
    "ProductCategory_data_bundles": 0,
    "ProductCategory_movies": 0,
    "ProductCategory_transport": 0
}

response = requests.post("http://127.0.0.1:8000/predict", json=sample_data)

print("Prediction result:", response.json())
