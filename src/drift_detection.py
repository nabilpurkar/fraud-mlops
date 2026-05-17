"""
Data Drift Detection using Evidently AI
- Reference data (training time) vs Current data compare karo
- 30% se zyada drift? Auto-retrain trigger karo
"""
import pandas as pd
import json
import sys
import os
import yaml
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_drift():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    threshold = params["drift"]["threshold"]

    logger.info("Loading reference and current data...")
    ref  = pd.read_csv("data/reference/reference.csv").drop("Class", axis=1)
    curr = pd.read_csv("data/processed/test.csv").drop("Class", axis=1)

    logger.info(f"Reference: {ref.shape} | Current: {curr.shape}")

    # Evidently Report
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
    ])
    report.run(reference_data=ref, current_data=curr)

    # HTML report save karo
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/drift_report.html")
    logger.info("Drift report saved: reports/drift_report.html")

    # JSON se summary nikalo
    result      = report.as_dict()
    drift_score = result["metrics"][0]["result"]["share_of_drifted_columns"]
    drifted     = result["metrics"][0]["result"]["dataset_drift"]

    summary = {
        "drift_score":         round(drift_score, 4),
        "drifted_columns_pct": round(drift_score * 100, 1),
        "dataset_drifted":     drifted,
        "threshold":           threshold,
        "action":              "RETRAIN" if drifted else "OK"
    }

    with open("reports/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Results print karo
    logger.info("=" * 50)
    logger.info("DRIFT DETECTION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Drifted columns: {drift_score*100:.1f}%")
    logger.info(f"  Threshold:       {threshold*100:.0f}%")
    logger.info(f"  Dataset drifted: {drifted}")
    logger.info(f"  Action:          {summary['action']}")
    logger.info("=" * 50)

    if drift_score > threshold:
        logger.warning("HIGH DRIFT DETECTED! Triggering retrain...")
        sys.exit(1)    # GitHub Actions mein retrain workflow trigger
    else:
        logger.info("Drift within acceptable range. No action needed.")
        sys.exit(0)

if __name__ == "__main__":
    detect_drift()
