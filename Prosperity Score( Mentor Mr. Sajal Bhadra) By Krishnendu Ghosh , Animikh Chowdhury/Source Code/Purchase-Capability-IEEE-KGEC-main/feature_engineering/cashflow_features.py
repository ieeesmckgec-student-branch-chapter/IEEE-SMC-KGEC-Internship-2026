"""
Cashflow features:
- average monthly balance
- minimum monthly balance
- negative balance occurrences
- cashflow stability score
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass(frozen=True)
class CashflowFeatures:
    avg_monthly_balance: float
    min_monthly_balance: float
    negative_balance_count: int
    min_balance_trend_slope: float   # slope of monthly minima: currency/month
    cashflow_stability_score: float  # 0..1 (higher = better)
    cashflow_component_score: float  # 0..100

def compute_monthly_balances(
    df: pd.DataFrame,
    date_col: str = "date",
    balance_col: Optional[str] = "running_balance",
) -> pd.Series:
    """
    If a running_balance column is present, use that aggregated per month (mean).
    Otherwise, attempt to construct by cumulatively summing amounts per account per date.
    Input assumption: df has 'amount' and 'date'.
    """
    if balance_col and balance_col in df.columns:
        s = pd.to_datetime(df[date_col], errors="coerce")
        temp = df[[date_col, balance_col]].copy()
        temp[date_col] = s
        temp = temp.dropna(subset=[date_col])
        # Coerce balance column to numeric (force invalid strings to NaN)
        temp[balance_col] = pd.to_numeric(temp[balance_col], errors="coerce")
        # Drop rows where balance is not numeric
        temp = temp.dropna(subset=[balance_col])
        if temp.empty:
            return pd.Series(dtype=float)
        # Convert to monthly period -> timestamp in a pandas-compatible way
        temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
        monthly = temp.groupby("period")[balance_col].mean().sort_index()
        return monthly.astype(float)

    # fallback: build running balance by sorting by date and cumulatively summing amounts
    if "amount" not in df.columns or date_col not in df.columns:
        return pd.Series(dtype=float)
    temp = df[[date_col, "amount"]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).sort_values(date_col)
    temp["running"] = temp["amount"].cumsum()
    temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = temp.groupby("period")["running"].mean().sort_index()
    return monthly.astype(float)

def compute_cashflow_features(
    df: pd.DataFrame,
    date_col: str = "date",
    balance_col: Optional[str] = "running_balance",
    min_months_for_trend: int = 3,
) -> CashflowFeatures:
    monthly = compute_monthly_balances(df, date_col=date_col, balance_col=balance_col)
    if monthly.empty:
        return CashflowFeatures(0.0, 0.0, 0, 0.0, 0.0, 0.0)

    avg_monthly = float(monthly.mean())
    min_monthly = float(monthly.min())
    # count months where monthly mean balance <= a threshold near zero (end-of-month stress)
    negative_count = int((monthly < 0).sum())

    # compute trend of minimum balances: here consider monthly minima by month (if available)
    # if balance_col present compute min per month (approx same as monthly)
    min_balance_trend_slope = 0.0
    trend_score = 0.5
    if len(monthly) >= min_months_for_trend:
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly.values.reshape(-1, 1)
        reg = LinearRegression()
        try:
            reg.fit(X, y)
            slope = float(reg.coef_[0][0])
            min_balance_trend_slope = slope
            # if slope increasing or stable => good
            mean_val = abs(avg_monthly) if abs(avg_monthly) > 1 else 1.0
            bound = mean_val / 10.0
            if slope >= bound:
                trend_score = 1.0
            elif slope <= -bound:
                trend_score = 0.0
            else:
                trend_score = float(np.clip((slope + bound) / (2 * bound), 0.0, 1.0))
        except Exception as e:
            logger.exception("Cashflow trend regression failed: %s", e)
            min_balance_trend_slope = 0.0
            trend_score = 0.5

    # Stability score composition:
    # Start from normalized average balance (log scale), penalize negatives and many negative occurrences.
    avg_norm = float(np.clip(np.tanh(avg_monthly / (abs(avg_monthly) + 1.0)), 0.0, 1.0)) if avg_monthly > 0 else 0.0
    neg_penalty = min(1.0, negative_count / max(1, len(monthly)))
    stability_raw = 0.6 * avg_norm + 0.3 * trend_score - 0.3 * neg_penalty
    stability = float(np.clip(stability_raw, 0.0, 1.0))

    # final component score 0..100
    component_score = float(np.clip(stability * 100.0, 0.0, 100.0))

    return CashflowFeatures(
        avg_monthly_balance=avg_monthly,
        min_monthly_balance=min_monthly,
        negative_balance_count=negative_count,
        min_balance_trend_slope=min_balance_trend_slope,
        cashflow_stability_score=stability,
        cashflow_component_score=component_score,
    )