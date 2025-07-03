import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_models(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb_classifier' : GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    }

    best_model = None
    best_score = 0

    mlflow.set_experiment("credit_score_modeling")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining: {name}")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probas = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, probas)
            report = classification_report(y_test, preds, output_dict=True)

            print(f"AUC-ROC: {auc:.4f}")
            print(classification_report(y_test, preds))

            # MLflow logging
            mlflow.log_param("model_type", name)
            mlflow.log_metric("auc_roc", auc)
            mlflow.log_metrics({
                "precision": report['1']['precision'],
                "recall": report['1']['recall'],
                "f1": report['1']['f1-score'],
                "accuracy": report['accuracy']
            })

            # Save model
            model_path = f"../models/{name}_best_model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # Register best
            if auc > best_score:
                best_model = model
                best_score = auc

    return best_model


if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_data.csv")
    df['std_amount'] = df['std_amount'].fillna(0)
    X = df.drop(columns=['AccountId','ProviderId', 'ProductId' ,'CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    train_models(X, y)