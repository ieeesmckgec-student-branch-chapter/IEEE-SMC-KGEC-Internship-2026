#!/usr/bin/env python3
"""
Main pipeline runner (robust defaults).

- If no --data is provided, the script looks for data/raw/*.xlsx and data/raw/*.csv.
- If no files are found, it runs a synthetic demo dataset so you can verify the pipeline.
- Use --use-ml with --model if you have trained the ML categorizer.

Usage examples:
  python main.py
  python main.py --data "data/raw/*.xlsx"
  python main.py --data "data/raw/statement1.xlsx"
  python main.py --use-ml --model models/ml_model.pkl
"""
import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd

from preprocessing.loader import load_transaction_files
from preprocessing.categorizer import categorize_transactions
from feature_engineering.feature_engineering import compute_all_features
from scoring.credit_score import compute_final_score
from scoring.loan_eligibility import evaluate_loan_eligibility
from scoring.risk_analysis import risk_bucket_from_score
from scoring.explainability import generate_explanations

# optional ML categorizer
try:
    from preprocessing.ml_categorizer import load_model, predict_categories  # type: ignore
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GLOBS = ["data/raw/*.xlsx", "data/raw/*.csv"]


def expand_paths(pattern_or_paths: List[str]) -> List[str]:
    out = []
    for p in pattern_or_paths:
        p = str(p)
        # If contains wildcard, expand by glob
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(glob(p)))
        else:
            # if explicit path exists, accept it; otherwise try glob on parent
            if Path(p).exists():
                out.append(p)
            else:
                out.extend(sorted(glob(p)))
    # dedupe preserving order
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def synthetic_dataset() -> pd.DataFrame:
    """
    Build a small synthetic transaction dataset (6 months) for demo/testing.
    """
    import numpy as np

    rows = []
    # 6 months salary on 1st
    for m in pd.date_range("2026-01-01", periods=6, freq="MS"):
        rows.append({"date": m, "amount": 50000.0, "payee": "ACME Payroll", "description": "Salary credit"})
    # daily random expenses
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    rng = np.random.default_rng(42)
    for d in dates:
        amt = -float(rng.integers(50, 2000))
        rows.append({"date": d, "amount": amt, "description": "Daily expense"})
    # monthly EMI
    for m in pd.date_range("2026-01-05", periods=6, freq="MS"):
        rows.append({"date": m, "amount": -8000.0, "description": "EMI payment"})
    # one bounce
    rows.append({"date": pd.Timestamp("2026-03-15"), "amount": -500.0, "description": "Bounced transaction"})
    df = pd.DataFrame(rows)
    # ensure date dtype
    df["date"] = pd.to_datetime(df["date"])
    return df


def pretty_print_summary(features, scoring, eligibility, risk):
    print("\n=== Component scores (0..100) ===")
    for k, v in features["component_scores"].items():
        print(f"  {k:8s}: {v:6.2f}")
    final = scoring.get("final_score")
    credit = scoring.get("credit_score")
    print(f"\nFinal score (0..100): {final:.2f}    Credit score (300..900): {credit}")
    print(f"Decision: {eligibility.get('decision')}   Risk bucket: {risk.get('bucket')}")
    if eligibility.get("reasons"):
        print("Reasons:", ", ".join(eligibility.get("reasons")))
    print("\n=== Raw feature summaries ===")
    print("Income:", features["raw"]["income_raw"])
    print("Expense:", features["raw"]["expense_raw"])
    print("Repayment:", features["raw"]["repayment_raw"])
    print("\n(See saved JSON for full details.)\n")


def main():
    p = argparse.ArgumentParser(description="Run AI Credit Scoring pipeline on transaction files")
    p.add_argument("--data", "-d", nargs="*", default=DEFAULT_GLOBS, help="File path(s) or glob(s) for transaction files (csv or xlsx). Default: data/raw/*.xlsx and data/raw/*.csv")
    p.add_argument("--use-ml", action="store_true", help="Use ML categorizer if model present (requires --model)")
    p.add_argument("--model", default="models/ml_model.pkl", help="Path to trained ML model (joblib) for categorization")
    p.add_argument("--out", default="data/output/dashboard.json", help="Path to save output JSON summary")
    p.add_argument("--preview", type=int, default=5, help="Number of transaction rows to preview")
    args = p.parse_args()

    # Expand globs and verify files exist
    paths = expand_paths(args.data or DEFAULT_GLOBS)
    if paths:
        logger.info("Found %d file(s) to process", len(paths))
        try:
            df = load_transaction_files(paths)
        except Exception as e:
            logger.exception("Failed to load transaction files: %s", e)
            raise SystemExit(1)
        logger.info("Loaded %d transactions", len(df))
        used_synthetic = False
    else:
        logger.warning("No input files found for patterns %s. Falling back to a synthetic demo dataset.", args.data)
        df = synthetic_dataset()
        used_synthetic = True

    # Show preview
    print("\n=== Sample transactions (after normalization) ===")
    # if loader returned data with normalized columns, print them; otherwise df from synthetic already normalized
    try:
        print(df.head(args.preview).to_string(index=False))
    except Exception:
        print(df.head(args.preview))

    # Categorize
    if args.use_ml:
        if not ML_AVAILABLE:
            logger.warning("ML categorizer module not available; falling back to rule-based categorizer.")
            df = categorize_transactions(df, text_cols=("description", "payee"))
        else:
            try:
                model = load_model(args.model)
                df = predict_categories(df, text_cols=("description", "payee"), model=model, out_col="category")
                logger.info("Categorized transactions using ML model: %s", args.model)
            except Exception as e:
                logger.exception("ML categorizer failed, falling back to rule-based: %s", e)
                df = categorize_transactions(df, text_cols=("description", "payee"))
    else:
        df = categorize_transactions(df, text_cols=("description", "payee"))
        logger.info("Categorized transactions using rule-based rules")

    # Compute features
    try:
        features = compute_all_features(df)
    except Exception as e:
        logger.exception("Feature computation failed: %s", e)
        raise SystemExit(1)

    # Compute final score & credit score
    scoring = compute_final_score(features["component_scores"])
    eligibility = evaluate_loan_eligibility(features, scoring["final_score"])
    risk = risk_bucket_from_score(scoring["final_score"])
    explanations = generate_explanations(features["raw"])

    # Pretty print summary
    pretty_print_summary(features, scoring, eligibility, risk)

    # Build JSON output
    out_obj = {
        "component_scores": features["component_scores"],
        "final_score": scoring,
        "eligibility": eligibility,
        "risk": risk,
        "explanations": explanations,
        "raw": {
            "income": getattr(features["raw"]["income_raw"], "__dict__", str(features["raw"]["income_raw"])),
            "expense": getattr(features["raw"]["expense_raw"], "__dict__", str(features["raw"]["expense_raw"])),
            "cashflow": getattr(features["raw"]["cashflow_raw"], "__dict__", str(features["raw"]["cashflow_raw"])),
            "repayment": getattr(features["raw"]["repayment_raw"], "__dict__", str(features["raw"]["repayment_raw"])),
            "behaviour": getattr(features["raw"]["behaviour_raw"], "__dict__", str(features["raw"]["behaviour_raw"])),
        },
        "used_synthetic_data": used_synthetic,
    }

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, default=str)
    logger.info("Saved output JSON to %s", out_path)


if __name__ == "__main__":
    main()