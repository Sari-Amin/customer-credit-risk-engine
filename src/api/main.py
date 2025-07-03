from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/random_forest_best_model.pkl")

class CustomerFeatures(BaseModel):
    total_amount: float
    avg_amount: float
    std_amount: float
    txn_count: int
    fraud_txn_count: int
    fraud_ratio: float
    txn_hour: int
    txn_day: int
    txn_month: int
    txn_weekday: int
    ChannelId_3: float
    ChannelId_2: float
    ProductCategory_airtime: float
    ProductCategory_financial_services: float
    ProductCategory_other: float
    ProductCategory_ticket: float
    ProductCategory_tv: float
    ProductCategory_utility_bill: float
    ProductCategory_data_bundles: float
    ProductCategory_movies: float
    ProductCategory_transport: float

@app.post("/predict")
def predict(customer: CustomerFeatures):
    data = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(data)[0]
    return {"is_high_risk": int(prediction)}
