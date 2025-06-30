import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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


class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if not np.issubdtype(df[self.time_col].dtype, np.datetime64):
            df[self.time_col] = pd.to_datetime(df[self.time_col])

        df['txn_hour'] = df[self.time_col].dt.hour
        df['txn_day'] = df[self.time_col].dt.day
        df['txn_month'] = df[self.time_col].dt.month
        df['txn_weekday'] = df[self.time_col].dt.weekday

        # Optional: drop original time column
        df = df.drop(columns=[self.time_col])
        return df
    

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, encoding='onehot'):
        self.columns = columns
        self.encoding = encoding
        self.encoders = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include='object').columns.tolist()

        for col in self.columns:
            if self.encoding == 'label':
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders[col] = le
            elif self.encoding == 'onehot':
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                ohe.fit(X[[col]])
                self.encoders[col] = ohe

        return self

    def transform(self, X):
        df = X.copy()

        for col in self.columns:
            encoder = self.encoders[col]
            if self.encoding == 'label':
                df[col] = encoder.transform(df[col])
            elif self.encoding == 'onehot':
                encoded = encoder.transform(df[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                    index=df.index
                )
                df = df.drop(columns=[col])
                df = pd.concat([df, encoded_df], axis=1)

        return df


class DropUninformativeCols(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None, id_threshold=0.95):
        self.drop_cols = drop_cols
        self.id_threshold = id_threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        df = X.copy()

        # Drop manually specified columns
        if self.drop_cols:
            self.to_drop_.extend(self.drop_cols)

        # Drop columns with 1 unique value
        self.to_drop_.extend(df.columns[df.nunique() <= 1].tolist())

        # Drop columns with too many unique values 
        for col in df.columns:
            if df[col].nunique() / len(df) > self.id_threshold:
                self.to_drop_.append(col)

        # Drop duplicates
        self.to_drop_ = list(set(self.to_drop_))

        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')




def build_pipeline():
    pipeline = Pipeline(steps=[
        ('date', DateFeatures()),
        ('drop_cols', DropUninformativeCols(
            drop_cols=['TransactionId', 'CurrencyCode', 'CountryCode', 'SubscriptionId', 'BatchId']
        )),
        ('aggregate', AggregateFeatures()),
        ('categorical', CategoricalEncoder(
            columns=['ProductCategory', 'ChannelId'], encoding='onehot'
        )),
        # Optional: Add scaling
        # ('scaler', StandardScaler())  # for linear models
    ])
    return pipeline

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    pipe = build_pipeline()
    return pipe.fit_transform(df)
