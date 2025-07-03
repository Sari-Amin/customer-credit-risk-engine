import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class ProxyTargetCreator:

    def __init__(self, date_col='TransactionStartTime', amount_col='Amount', customer_id_col='CustomerId', n_clusters=3):
        self.date_col = date_col
        self.amount_col = amount_col
        self.customer_id_col = customer_id_col
        self.n_clusters = n_clusters
        self.kmeans = None
        self.high_risk_cluster = None


    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        current_date = df[self.date_col].max()

        rfm = df.groupby(self.customer_id_col).agg({
            self.date_col: 'max',
            self.customer_id_col: 'count',
            self.amount_col: 'sum'
        }).rename(columns={
            self.date_col: 'last_txn',
            self.customer_id_col: 'frequency',
            self.amount_col: 'monetary'
        })

        rfm['recency_days'] = (current_date - rfm['last_txn']).dt.days
        rfm = rfm[['recency_days', 'frequency', 'monetary']]
        return rfm


    def cluster_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        rfm_for_clustering = rfm_df.copy()
        rfm_for_clustering = (rfm_for_clustering - rfm_for_clustering.mean()) / rfm_for_clustering.std()

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        rfm_df['cluster'] = self.kmeans.fit_predict(rfm_for_clustering)

        # Find the most risky cluster: highest recency, lowest freq and monetary
        cluster_stats = rfm_df.groupby('cluster').agg({
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        })
        self.high_risk_cluster = cluster_stats.sort_values(['recency_days', 'frequency', 'monetary'], ascending=[False, True, True]).index[0]

        return rfm_df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        rfm = self.compute_rfm(df)
        rfm_clustered = self.cluster_customers(rfm)

        target = rfm_clustered[['recency_days', 'frequency', 'monetary', 'cluster']].copy()
        target['is_high_risk'] = (rfm_clustered['cluster'] == self.high_risk_cluster).astype(int)
        target = target.reset_index()  # bring back CustomerId as column

        return target

