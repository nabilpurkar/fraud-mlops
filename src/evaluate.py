"""
Model Evaluation
- Detailed metrics calculate karo
- Feature importance nikalo
- Classification report generate karo
"""
import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, classification_report,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate():
    test_df = pd.read_csv("data/processed/test.csv")
    X_test  = test_df.drop("Class", axis=1)
    y_test  = test_df["Class"]

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    scores = {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "f1_score":  round(f1_score(y_test, preds, average="weighted"), 4),
        "recall":    round(recall_score(y_test, preds, average="weighted"), 4),
        "precision": round(precision_score(y_test, preds, average="weighted"), 4),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    # Classification Report
    logger.info("\n" + classification_report(y_test, preds,
        target_names=["Normal", "Fraud"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"False Positives (Normal as Fraud): {cm[0][1]}")
    logger.info(f"False Negatives (Fraud Missed):    {cm[1][0]}")

    # Feature Importance (Top 10)
    feature_imp = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)

    logger.info("\nTop 10 Important Features:")
    for feat, imp in feature_imp.head(10).items():
        logger.info(f"  {feat}: {imp:.4f}")

    logger.info("Evaluation complete!")
    return scores

if __name__ == "__main__":
    evaluate()
