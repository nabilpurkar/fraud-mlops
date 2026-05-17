"""
Quality Gate — Bad model ko deploy hone se roko
CI/CD mein fail hone pe sys.exit(1) se pipeline rukegi
"""
import json
import sys
import yaml
import logging

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def quality_gate():
    # Thresholds params.yaml se lo
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    gates = params["quality_gates"]

    # Actual metrics load karo
    try:
        with open("metrics/scores.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        logger.error("metrics/scores.json not found! Run train.py first.")
        sys.exit(1)

    # Check karo
    logger.info("=" * 50)
    logger.info("QUALITY GATE CHECK")
    logger.info("=" * 50)

    failed = []
    for metric, threshold in gates.items():
        actual = metrics.get(metric, 0)
        passed = actual >= threshold
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {metric:12}: {actual:.4f}  (need >= {threshold})  [{status}]")
        if not passed:
            failed.append(f"{metric}: {actual:.4f} < {threshold}")

    logger.info("=" * 50)

    if failed:
        logger.error(f"BLOCKED! {len(failed)} quality gate(s) failed:")
        for f in failed:
            logger.error(f"  - {f}")
        logger.error("Model will NOT be deployed.")
        sys.exit(1)    # Pipeline rok do
    else:
        logger.info("ALL QUALITY GATES PASSED!")
        logger.info("Model is ready to deploy.")
        sys.exit(0)

if __name__ == "__main__":
    quality_gate()
