"""
Model Training Pipeline
- Data load karo
- Random Forest train karo
- MLflow se track karo
- Model save karo
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import mlflow
import mlflow.sklearn
import pickle
import yaml
import os
import json
import logging

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # MLflow setup
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("fraud-detection")

    logger.info("Loading training data...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    X_test  = test_df.drop("Class", axis=1)
    y_test  = test_df["Class"]

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    with mlflow.start_run(run_name=f"rf-{params['train']['n_estimators']}-trees"):

        # Params log karo
        mlflow.log_params({
            "n_estimators":    params["train"]["n_estimators"],
            "max_depth":       params["train"]["max_depth"],
            "min_samples_leaf": params["train"]["min_samples_leaf"],
            "test_size":       params["train"]["test_size"],
        })

        # Train
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=params["train"]["n_estimators"],
            max_depth=params["train"]["max_depth"],
            min_samples_leaf=params["train"]["min_samples_leaf"],
            random_state=params["train"]["random_state"],
            n_jobs=-1       # All CPU cores use karo
        )
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        scores = {
            "accuracy":  round(accuracy_score(y_test, preds), 4),
            "f1_score":  round(f1_score(y_test, preds, average="weighted"), 4),
            "recall":    round(recall_score(y_test, preds, average="weighted"), 4),
            "precision": round(precision_score(y_test, preds, average="weighted"), 4),
        }

        # Metrics MLflow mein log karo
        mlflow.log_metrics(scores)

        # Model MLflow registry mein save karo
        mlflow.sklearn.log_model(
            model,
            "fraud-detector",
            registered_model_name="FraudDetector"
        )

        # Metrics file mein save karo (DVC ke liye)
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/scores.json", "w") as f:
            json.dump(scores, f, indent=2)

        # Model pickle file mein bhi save karo (serving ke liye)
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        logger.info("=== Training Results ===")
        for k, v in scores.items():
            logger.info(f"  {k:12}: {v:.4f}")
        logger.info("Model saved: models/model.pkl")

if __name__ == "__main__":
    train()
