import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _read_labeled_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(p)
    else:
        # try utf-8 then fallback to latin1 for files with different encodings
        try:
            df = pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("utf-8 decode failed, falling back to latin1 for %s", p)
            df = pd.read_csv(p, encoding="latin1")
    return df

def train_categorizer(csv_path: str, out_model_path: str, min_df: int = 1, bootstrap_rules: bool = False):
    """
    csv_path may be a CSV or XLSX. Expects columns:
      - 'text' and 'category'
    If 'text' missing but 'description'/'payee' are present, builds 'text' as description + payee.
    If 'category' missing and bootstrap_rules=True, will auto-label using your rule-based classifier (for bootstrapping).
    """
    df = _read_labeled_table(csv_path)
    # build text if necessary
    if "text" not in df.columns:
        if "description" in df.columns:
            payee_col = df.columns[df.columns.str.lower() == "payee"]
            if "payee" in df.columns:
                df["text"] = df["description"].astype(str).fillna("") + " " + df["payee"].astype(str).fillna("")
            else:
                df["text"] = df["description"].astype(str).fillna("")
            logger.info("Built 'text' column from description/payee")
        else:
            raise ValueError("Input file must contain a 'text' column or 'description' (and optionally 'payee') to build text.")

    # if category missing, either bootstrap or fail
    if "category" not in df.columns:
        if bootstrap_rules:
            try:
                from models.rule_based import batch_classify as rule_batch  # type: ignore
            except Exception:
                from preprocessing.categorizer import categorize_transactions  # fallback
                # create temp DF with text in 'description' to reuse existing code
                temp_df = pd.DataFrame({"description": df["text"]})
                labeled = categorize_transactions(temp_df, text_cols=("description", ""), out_col="category")
                df["category"] = labeled["category"].values
            else:
                df["category"] = rule_batch(df["text"].tolist())
            logger.info("Auto-labeled data using rule-based classifier (bootstrap).")
            # save bootstrap for manual review if desired
            bootstrap_path = Path("data/bootstrap_labels.csv")
            df[["text","category"]].to_csv(bootstrap_path, index=False)
            logger.info("Bootstrap labels saved to %s — please review & correct before final training", bootstrap_path)
        else:
            raise ValueError("Input file has no 'category' column. Provide labeled data or run with bootstrap_rules=True to auto-label using rules.")

    # proceed with training using df['text'] and df['category'] ...
    # existing training code continues here (GridSearchCV etc.)