"""
Data Preparation Pipeline
- CSV load karo
- SMOTE se imbalanced data handle karo
- Train/Test/Reference split karo
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_data():
    # Params load karo
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    logger.info("Loading dataset...")
    df = pd.read_csv("data/raw/creditcard.csv")

    logger.info(f"Total rows:  {len(df)}")
    logger.info(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

    # Features aur label alag karo
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Reference data save karo (drift detection ke liye — SMOTE se pehle)
    ref_sample = df.sample(n=5000, random_state=42)
    os.makedirs("data/reference", exist_ok=True)
    ref_sample.to_csv("data/reference/reference.csv", index=False)
    logger.info("Reference data saved (5000 samples for drift baseline)")

    # SMOTE — fraud cases 0.17% hain, model useless hoga bina iske
    logger.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=params["train"]["random_state"])
    X_res, y_res = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_res, columns=X.columns)
    df_balanced["Class"] = y_res

    logger.info(f"After SMOTE: {len(df_balanced)} rows, balanced classes")

    # Train/Test Split
    train_df, test_df = train_test_split(
        df_balanced,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
        stratify=df_balanced["Class"]
    )

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
