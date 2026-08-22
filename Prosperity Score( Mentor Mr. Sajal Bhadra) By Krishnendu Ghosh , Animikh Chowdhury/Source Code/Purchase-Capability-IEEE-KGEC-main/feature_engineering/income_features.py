"""
Income features module.

API:
- compute_income_features(df, date_col='date', amount_col='amount', source_col='payee')
  -> IncomeFeatures dataclass

Expected input:
- df: pandas.DataFrame including at minimum a date column and an amount column.
  Income transactions are identified as positive amounts by default. If your categorizer tags 'category' you can filter externally.
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
class IncomeFeatures:
    avg_monthly_income: float
    income_stability_score: float        # 0..1
    income_consistency_score: float     # 0..1 (alias of stability by default)
    income_source_diversity_score: float # 0..1 (higher = safer per your spec)
    income_growth_slope: float          # raw slope (currency per month)
    income_growth_score: float          # 0..1 normalized
    income_component_score: float       # 0..100 final normalized income score for pipeline use


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.Series:
    s = df[date_col]
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_datetime(s, errors="coerce")
    return s


def compute_monthly_income_series(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
) -> pd.Series:
    """
    Return a pandas Series indexed by month timestamp (month-end) with aggregated gross income (sum of positive amounts).
    This implementation groups by month PeriodIndex to be robust across pandas versions.
    """
    if date_col not in df.columns or amount_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns {date_col!r} and {amount_col!r}")

    dates = _ensure_datetime(df, date_col)
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

    temp = pd.DataFrame({"date": dates, "amount": amounts})
    temp = temp.dropna(subset=["date"])
    # consider incomes as positive amounts
    incomes = temp[temp["amount"] > 0].copy()
    if incomes.empty:
        # return empty series with monthly timestamp index
        return pd.Series(dtype=float)

    incomes.set_index("date", inplace=True)
    # Group by month PeriodIndex (robust across pandas versions), then convert periods to month-end timestamps
    monthly = incomes["amount"].groupby(incomes.index.to_period("M")).sum().sort_index()
    monthly.index = monthly.index.to_timestamp(how="end")
    return monthly

def _normalize_0_1(x: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    v = (x - min_val) / (max_val - min_val)
    return float(np.clip(v, 0.0, 1.0))


def compute_income_features(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
    source_col: Optional[str] = "payee",
    effective_fraction: float = 0.85,
    min_months_for_growth: int = 3,
) -> IncomeFeatures:
    """
    Compute income-related features from transactions DataFrame.

    Args:
      df: transactions DataFrame with at least date_col and amount_col.
      date_col: name of date column
      amount_col: name of amount column (credits positive)
      source_col: optional column name used to compute source diversity (payee)
      effective_fraction: fraction of gross credit income considered as effective/net income (e.g., 0.85)
      min_months_for_growth: minimum months to run growth regression

    Returns:
      IncomeFeatures dataclass
    """
    # Defensive checks
    if date_col not in df.columns or amount_col not in df.columns:
        logger.debug("compute_income_features: missing required columns")
        return IncomeFeatures(
            avg_monthly_income=0.0,
            income_stability_score=0.0,
            income_consistency_score=0.0,
            income_source_diversity_score=0.0,
            income_growth_slope=0.0,
            income_growth_score=0.0,
            income_component_score=0.0,
        )

    # Compute gross monthly income series (sum of credits)
    monthly_gross = compute_monthly_income_series(df, date_col=date_col, amount_col=amount_col)

    if monthly_gross.empty:
        logger.debug("compute_income_features: no positive income transactions found")
        return IncomeFeatures(
            avg_monthly_income=0.0,
            income_stability_score=0.0,
            income_consistency_score=0.0,
            income_source_diversity_score=0.0,
            income_growth_slope=0.0,
            income_growth_score=0.0,
            income_component_score=0.0,
        )

    # Apply effective fraction to each month's gross to get effective monthly income
    effective_fraction = float(effective_fraction)
    if effective_fraction < 0 or effective_fraction > 1:
        logger.warning("effective_fraction out of bounds, clamping to [0,1]: %s", effective_fraction)
        effective_fraction = float(np.clip(effective_fraction, 0.0, 1.0))

    monthly_effective = monthly_gross * effective_fraction

    # Average monthly effective income (use mean of available months)
    avg_monthly_effective = float(monthly_effective.mean())

    # Stability: 1 - (std/mean) on effective monthly amounts
    std_effective = float(monthly_effective.std(ddof=0)) if len(monthly_effective) > 0 else 0.0
    if avg_monthly_effective > 0:
        stability = 1.0 - (std_effective / avg_monthly_effective)
    else:
        stability = 0.0
    stability = float(np.clip(stability, 0.0, 1.0))

    # Consistency alias
    consistency = stability

    # Source diversity: evaluate unique payees for positive (credit) transactions
    if source_col and source_col in df.columns:
        credit_mask = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) > 0
        sources = df.loc[credit_mask, source_col].dropna().astype(str).str.strip()
        unique_sources = sources.replace("", np.nan).dropna().unique()
        n_sources = max(1, len(unique_sources))
    else:
        # fallback to description if payee not available
        if "description" in df.columns:
            credit_mask = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) > 0
            sources = df.loc[credit_mask, "description"].dropna().astype(str).str.strip()
            unique_sources = sources.replace("", np.nan).dropna().unique()
            n_sources = max(1, len(unique_sources))
        else:
            n_sources = 1

    # Map number of sources to diversity score: single source -> 1.0, many sources (>=6) -> 0.0, linear in-between
    if n_sources <= 1:
        diversity_score = 1.0
    elif n_sources >= 6:
        diversity_score = 0.0
    else:
        diversity_score = float(np.clip(1.0 - (n_sources - 1) / 5.0, 0.0, 1.0))

    # Income growth: linear regression on monthly_effective series
    income_growth_slope = 0.0
    growth_score = 0.0
    if len(monthly_effective) >= min_months_for_growth:
        y = monthly_effective.values.reshape(-1, 1)
        X = (np.arange(len(y))).reshape(-1, 1)
        reg = LinearRegression()
        try:
            reg.fit(X, y)
            slope = float(reg.coef_[0][0])  # currency per month (effective)
            income_growth_slope = slope
            # Normalize slope relative to mean monthly effective income to a 0..1 score
            if avg_monthly_effective > 0:
                # map slope/avg to [-inf..inf] then to [0..1] centered at 0 (0.5 neutral)
                growth_score = float(np.clip((slope / avg_monthly_effective) * 0.5 + 0.5, 0.0, 1.0))
            else:
                growth_score = 0.5 if slope == 0 else (1.0 if slope > 0 else 0.0)
        except Exception as e:
            logger.exception("compute_income_features: growth regression failed: %s", e)
            income_growth_slope = 0.0
            growth_score = 0.0
    else:
        # insufficient data -> neutral growth
        growth_score = 0.5
        income_growth_slope = 0.0

    # Map average monthly effective income to a normalized value in 0..1 using log scaling to reduce outlier sensitivity
    try:
        avg_norm = float(np.clip(np.log1p(avg_monthly_effective) / (np.log1p(avg_monthly_effective) + 1.0), 0.0, 1.0))
    except Exception:
        avg_norm = 0.0

    # Combine into final income component score (0..100)
    # Weights tuned to emphasize stability and average income reasonably; tweak to match your business rules
    w_avg = 0.3
    w_stab = 0.4
    w_div = 0.15
    w_growth = 0.15

    combined = w_avg * avg_norm + w_stab * stability + w_div * diversity_score + w_growth * growth_score
    income_component_score = float(np.clip(combined * 100.0, 0.0, 100.0))

    return IncomeFeatures(
        avg_monthly_income=avg_monthly_effective,
        income_stability_score=stability,
        income_consistency_score=consistency,
        income_source_diversity_score=diversity_score,
        income_growth_slope=income_growth_slope,
        income_growth_score=growth_score,
        income_component_score=income_component_score,
    )
