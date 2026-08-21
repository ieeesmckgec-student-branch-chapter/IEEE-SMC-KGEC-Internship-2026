"""
Simple rule-based + optional ML transaction categorizer.

- If use_ml=True and a saved ML model is available, the function will try ML first.
- If ML is not confident (handled in predict_categories) it can fall back to the rule classifier.
- If ML loading fails or --use-ml is False, it uses the external rule module when present.
"""
from __future__ import annotations
from typing import Tuple, Iterable, List, Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# prefer external rule module if present
try:
    from models import rule_based as external_rule_mod  # type: ignore
    EXTERNAL_CLASSIFY = external_rule_mod.classify
    EXTERNAL_BATCH = getattr(external_rule_mod, "batch_classify", None)
    logger.debug("Loaded external rule-based module from models.rule_based")
except Exception:
    EXTERNAL_CLASSIFY = None
    EXTERNAL_BATCH = None
    logger.debug("No external rule-based module found at models.rule_based")

# optional ML categorizer helpers
try:
    from preprocessing.ml_categorizer import load_model, predict_categories  # type: ignore
    ML_AVAILABLE = True
    logger.debug("ML categorizer module available")
except Exception:
    load_model = None  # type: ignore
    predict_categories = None  # type: ignore
    ML_AVAILABLE = False
    logger.debug("ML categorizer module NOT available")


def _build_texts(df: pd.DataFrame, text_cols: Tuple[str, str]) -> pd.Series:
    texts = df.get(text_cols[0], "").fillna("").astype(str).str.strip()
    if text_cols[1] in df.columns:
        texts = texts + " " + df.get(text_cols[1], "").fillna("").astype(str).str.strip()
    return texts


def categorize_transactions(
    df: pd.DataFrame,
    text_cols: Tuple[str, str] = ("description", "payee"),
    out_col: str = "category",
    use_ml: bool = False,
    model_path: str = "models/ml_model.pkl",
    confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Return a copy of df with column `out_col` containing predicted categories.

    Args:
      df: normalized DataFrame with description/payee (or equivalent) columns
      text_cols: tuple of (description_col, payee_col)
      out_col: output column name
      use_ml: whether to attempt ML-based classification first
      model_path: path to saved ML model (used if use_ml True)
      confidence_threshold: passed to predict_categories via ml_categorizer (controls fallback)
    """
    df_out = df.copy()
    texts = _build_texts(df_out, text_cols)
    # ML path
    if use_ml:
        if not ML_AVAILABLE:
            logger.warning("Requested ML but preprocessing.ml_categorizer is not available; falling back to rules")
        else:
            try:
                model = load_model(model_path)
                # predict_categories will itself optionally use rule_classifier for low-confidence cases.
                df_ml = predict_categories(
                    df_out,
                    text_cols=text_cols,
                    model=model,
                    model_path=model_path,
                    out_col=out_col,
                    confidence_threshold=confidence_threshold,
                    rule_classifier=EXTERNAL_CLASSIFY if EXTERNAL_CLASSIFY else None,
                )
                logger.info("Categorized using ML model: %s", model_path)
                # ensure column exists and fill NaN
                df_ml[out_col] = df_ml.get(out_col).fillna("Uncategorized")
                return df_ml
            except Exception as e:
                logger.exception("ML categorizer failed, falling back to rules: %s", e)

    # Rule-based fallback (external module preferred)
    preds: List[Optional[str]]
    if EXTERNAL_BATCH is not None:
        try:
            preds = EXTERNAL_BATCH(texts.tolist())
            logger.info("Categorized using external rule-based batch classifier")
        except Exception as e:
            logger.exception("External batch classifier failed: %s", e)
            preds = [None] * len(texts)
    elif EXTERNAL_CLASSIFY is not None:
        try:
            preds = [EXTERNAL_CLASSIFY(t) for t in texts.tolist()]
            logger.info("Categorized using external rule-based classifier")
        except Exception as e:
            logger.exception("External classifier failed: %s", e)
            preds = [None] * len(texts)
    else:
        # no rule engine available: return Unclassified
        logger.warning("No rule-based classifier available; returning 'Uncategorized' for all rows")
        preds = [None] * len(texts)

    df_out[out_col] = preds
    df_out[out_col] = df_out[out_col].fillna("Uncategorized")
    return df_out