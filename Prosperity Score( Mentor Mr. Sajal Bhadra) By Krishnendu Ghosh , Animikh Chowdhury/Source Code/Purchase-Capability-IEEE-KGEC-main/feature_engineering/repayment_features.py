"""
Repayment features:
- count of EMIs and repayment transactions
- on-time payment ratio
- bounce frequency
- credit card payment behaviour (min vs full)
- repayment_component_score (0..100)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass(frozen=True)
class RepaymentFeatures:
    emi_count: int
    on_time_ratio: float        # 0..1
    bounce_count: int
    bounce_rate: float          # bounce_count / total_debits
    credit_card_min_ratio: float # fraction of credit card payments that were min only (higher -> worse)
    repayment_component_score: float  # 0..100

def compute_repayment_features(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
    category_col: Optional[str] = "category",
    event_col: Optional[str] = "event",  # optional: contains 'bounce' or flags
    due_date_col: Optional[str] = "due_date",
    payment_date_col: Optional[str] = "payment_date",
) -> RepaymentFeatures:
    """
    Expects:
    - EMI / Loan repayment transactions categorized as 'Loan Repayments' or 'EMI' in category_col
    - Bounce events indicated by category 'Bounces' or event_col containing 'bounce'
    - Credit card payment rows categorized under 'Loan Repayments' or 'Credit Card'
    - For on-time ratio, optional due_date/payment_date can be used (if not present we infer by description presence 'late' etc.)
    """
    if df is None or df.empty:
        return RepaymentFeatures(0, 0.0, 0, 0.0, 0.0, 0.0)

    df_local = df.copy()
    df_local[amount_col] = pd.to_numeric(df_local[amount_col], errors="coerce").fillna(0.0)
    debits = df_local[df_local[amount_col] < 0].copy()
    total_debits = len(debits)
    # bounces
    bounce_mask = pd.Series(False, index=df_local.index)
    if category_col and category_col in df_local.columns:
        bounce_mask = df_local[category_col].astype(str).str.lower().str.contains("bounce")
    if event_col and event_col in df_local.columns:
        bounce_mask = bounce_mask | df_local[event_col].astype(str).str.lower().str.contains("bounce")
    bounce_count = int(bounce_mask.sum())

    # EMI / loan repayments
    emi_mask = pd.Series(False, index=df_local.index)
    if category_col and category_col in df_local.columns:
        emi_mask = df_local[category_col].astype(str).str.lower().str.contains("emi|loan|credit card")
    emi_count = int(emi_mask.sum())

    # On-time ratio: if due_date/payment_date present, compute directly
    on_time_ratio = 0.0
    if due_date_col and payment_date_col and due_date_col in df_local.columns and payment_date_col in df_local.columns:
        mask = emi_mask & df_local[due_date_col].notna() & df_local[payment_date_col].notna()
        if mask.any():
            due = pd.to_datetime(df_local.loc[mask, due_date_col], errors="coerce")
            paid = pd.to_datetime(df_local.loc[mask, payment_date_col], errors="coerce")
            on_time = (paid <= due).sum()
            on_time_ratio = float(on_time / mask.sum())
    else:
        # fallback heuristic: assume most EMIs are on-time; if description mentions 'late' mark as late.
        if emi_count == 0:
            on_time_ratio = 1.0
        else:
            late_mask = df_local.loc[emi_mask, :].apply(lambda r: "late" in str(r.get("description", "")).lower() or "delayed" in str(r.get("description", "")).lower(), axis=1)
            on_time_ratio = float(max(0.0, 1.0 - late_mask.sum() / max(1, emi_count)))

    # Credit card behaviour: detect payments labeled credit card and whether description includes 'minimum' or 'min'
    cc_mask = pd.Series(False, index=df_local.index)
    if category_col and category_col in df_local.columns:
        cc_mask = df_local[category_col].astype(str).str.lower().str.contains("credit card")
    cc_rows = df_local.loc[cc_mask]
    cc_min_count = 0
    if not cc_rows.empty:
        cc_min_count = int(cc_rows["description"].astype(str).str.lower().str.contains("min").sum())
    credit_card_min_ratio = float(cc_min_count / max(1, len(cc_rows))) if len(cc_rows) > 0 else 0.0

    bounce_rate = float(bounce_count / max(1, total_debits)) if total_debits > 0 else 0.0

    # Scoring composition:
    # on_time_ratio (0..1) good -> high score
    # bounce_rate bad -> penalize
    # credit_card_min_ratio bad -> penalize
    w_on_time = 0.6
    w_bounce = 0.25
    w_cc = 0.15

    score = w_on_time * on_time_ratio + w_bounce * (1.0 - min(1.0, bounce_rate / 0.05)) + w_cc * (1.0 - credit_card_min_ratio)
    # Ensure in 0..1
    score = float(np.clip(score, 0.0, 1.0))
    component_score = float(round(score * 100.0, 2))

    return RepaymentFeatures(
        emi_count=emi_count,
        on_time_ratio=on_time_ratio,
        bounce_count=bounce_count,
        bounce_rate=bounce_rate,
        credit_card_min_ratio=credit_card_min_ratio,
        repayment_component_score=component_score,
    )