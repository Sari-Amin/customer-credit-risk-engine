from sklearn.base import BaseEstimator, TransformerMixin


class AggregateFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col


    def fit(self, X, y = None):
        return self
    

    def transform(self, X):
        df = X.copy()

        # Basic Amount aggregations
        agg_amounts = df.groupby(self.customer_id_col)[self.amount_col].agg(
            total_amount='sum',
            avg_amount='mean',
            std_amount='std',
            txn_count='count',
            min_amount='min',
            max_amount='max'
        )

        # Fraud aggregations
        agg_fraud = df.groupby(self.customer_id_col)['FraudResult'].agg(
            fraud_txn_count='sum'
        )

        # Join and compute fraud_ratio
        agg = agg_amounts.join(agg_fraud)
        agg['fraud_ratio'] = agg['fraud_txn_count'] / agg['txn_count']

        # Reset index to merge back
        agg = agg.reset_index()

        # Merge with original data
        df = df.merge(agg, on=self.customer_id_col, how='left')

        return df




def build_pipeline():

    return None

import pandas as pd
import sys

sys.path.append("../")
df = pd.read_csv("data/raw/data.csv")

agg = AggregateFeatures()
df_transformed = agg.fit_transform(df)
print(df_transformed.head())