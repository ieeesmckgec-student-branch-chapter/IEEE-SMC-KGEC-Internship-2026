"""
Expense features module.

API:
- compute_expense_features(df, date_col='date', amount_col='amount', category_col='category', income_total=None)
  -> ExpenseFeatures dataclass

Expected input:
- df: pandas.DataFrame including at minimum a date column and an amount column.
  Expense transactions are identified as negative amounts by default (debits).
- category_col (optional): if present should contain pre-computed categories like 'Fixed Expenses', 'Variable Expenses', 'Discretionary', etc.
- income_total (optional): total income over the same period; if not provided, the caller should compute or supply it (used for EIR).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class ExpenseFeatures:
    total_expenses: float
    eir: float                       # Expense-to-Income Ratio (0..inf)
    eir_score: float                 # 0..1 normalized (higher = better)
    fixed_ratio: float               # fixed / total_expenses (0..1)
    variable_ratio: float            # variable / total_expenses (0..1)
    discretionary_ratio: float       # discretionary / total_expenses (0..1)
    discretionary_score: float       # 0..1 (lower discretionary = better)
    expense_trend_slope: float       # currency per month (positive = increasing)
    expense_trend_score: float       # 0..1 (decreasing expenses => higher score)
    expense_component_score: float   # 0..100 final normalized expense score for pipeline use


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.Series:
    s = df[date_col]
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_datetime(s, errors="coerce")
    return s


def compute_monthly_expense_series(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
) -> pd.Series:
    """
    Return a pandas Series indexed by month start (Timestamp) with aggregated expenses (sum of absolute negative amounts).
    """
    if date_col not in df.columns or amount_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns {date_col!r} and {amount_col!r}")

    dates = _ensure_datetime(df, date_col)
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

    temp = pd.DataFrame({"date": dates, "amount": amounts})
    temp = temp.dropna(subset=["date"])
    # consider expenses as negative amounts (debits)
    expenses = temp[temp["amount"] < 0].copy()
    if expenses.empty:
        return pd.Series(dtype=float)

    expenses["period"] = expenses["date"].dt.to_period("M")
    # sum absolute values of debits
    monthly = expenses.groupby("period")["amount"].sum().abs().sort_index()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def _safe_div(n: float, d: float) -> float:
    if d == 0 or np.isnan(d):
        return 0.0
    return float(n / d)


def _eir_to_score(eir: float) -> float:
    """
    Map EIR (expense/income) to a score in 0..1 where lower EIR -> higher score.
    Thresholds from spec:
    <50% -> Excellent
    50-70% -> Good
    70-90% -> Risky
    >=90% -> High Risk

    We map:
    eir <= 0.5 -> 1.0
    0.5 < eir <= 0.7 -> linear 1.0->0.8
    0.7 < eir <= 0.9 -> linear 0.8->0.4
    eir > 0.9 -> linear 0.4->0.0 (capping at 0)
    """
    if eir <= 0.5:
        return 1.0
    if eir <= 0.7:
        # map 0.5..0.7 -> 1.0..0.8
        return float(1.0 - (eir - 0.5) * (0.2 / 0.2))
    if eir <= 0.9:
        # map 0.7..0.9 -> 0.8..0.4
        return float(0.8 - (eir - 0.7) * (0.4 / 0.2))
    # eir > 0.9, map to 0.4..0.0 over some reasonable range, say up to 2.0 (200% expense)
    max_eir = 2.0
    capped = min(eir, max_eir)
    # map 0.9..max_eir -> 0.4..0.0
    return float(max(0.0, 0.4 * (1.0 - (capped - 0.9) / (max_eir - 0.9))))


def compute_expense_features(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
    category_col: Optional[str] = "category",
    income_total: Optional[float] = None,
    min_months_for_trend: int = 3,
    discretionary_labels: Tuple[str, ...] = ("Discretionary", "Luxury", "Travel"),
    fixed_labels: Tuple[str, ...] = ("Fixed Expenses", "Rent", "EMI", "Insurance"),
    variable_labels: Tuple[str, ...] = ("Variable Expenses", "Food", "Shopping"),
) -> ExpenseFeatures:
    """
    Compute expense-related features.

    Returns an ExpenseFeatures dataclass:
    - eir: expense / income (caller can pass income_total; otherwise attempts to compute from df if positive incomes exist)
    - scores normalized to 0..1 and a final 0..100 component score.
    """
    monthly = compute_monthly_expense_series(df, date_col, amount_col)
    # total expenses (absolute)
    total_expenses = float(monthly.sum()) if not monthly.empty else 0.0

    # Determine income_total if not provided: sum positive amounts in df
    income_total_local = income_total
    if income_total_local is None:
        if date_col in df.columns and amount_col in df.columns:
            amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
            income_total_local = float(amounts[amounts > 0].sum())
        else:
            income_total_local = 0.0

    # EIR
    eir = _safe_div(total_expenses, income_total_local)  # can be inf if income_total_local==0 -> handled by _safe_div -> 0
    # If income_total_local == 0, set eir large to penalize
    if income_total_local == 0 and total_expenses > 0:
        # treat as very high EIR
        eir = float("inf")

    eir_score = 0.0
    if np.isfinite(eir):
        eir_score = float(_eir_to_score(eir))
    else:
        eir_score = 0.0

    # If category_col exists, compute category sums; otherwise use heuristics (not available)
    fixed = variable = discretionary = 0.0
    if category_col and category_col in df.columns:
        df_local = df.copy()
        df_local[amount_col] = pd.to_numeric(df_local[amount_col], errors="coerce").fillna(0.0)
        debits = df_local[df_local[amount_col] < 0].copy()
        if not debits.empty:
            # make category strings normalized
            cats = debits[category_col].fillna("").astype(str).str.strip()
            amounts_abs = debits[amount_col].abs()
            fixed_mask = cats.isin(fixed_labels)
            variable_mask = cats.isin(variable_labels)
            discretionary_mask = cats.isin(discretionary_labels)
            fixed = float(amounts_abs[fixed_mask].sum())
            variable = float(amounts_abs[variable_mask].sum())
            discretionary = float(amounts_abs[discretionary_mask].sum())
            # any uncategorized debits go to variable by default
            uncategorized = float(amounts_abs[~(fixed_mask | variable_mask | discretionary_mask)].sum())
            variable += uncategorized
    else:
        # No category column: estimate discretionary as fraction of variable-ish descriptions if present
        fixed = 0.0
        variable = total_expenses
        discretionary = 0.0

    fixed_ratio = _safe_div(fixed, total_expenses)
    variable_ratio = _safe_div(variable, total_expenses)
    discretionary_ratio = _safe_div(discretionary, total_expenses)

    # Discretionary score: lower discretionary share => higher score
    # Map 0% -> 1.0, 0..30% -> linear 1..0.7, 30..60% -> 0.7..0.4, >60% -> 0.4..0
    d = discretionary_ratio
    if d <= 0.0:
        discretionary_score = 1.0
    elif d <= 0.3:
        discretionary_score = float(1.0 - (d / 0.3) * 0.3)  # 1 -> 0.7 at 30%
    elif d <= 0.6:
        discretionary_score = float(0.7 - ((d - 0.3) / 0.3) * 0.3)  # 0.7 -> 0.4
    else:
        # >0.6 maps to 0.4 down to 0 at 1.0
        discretionary_score = float(max(0.0, 0.4 - ((d - 0.6) / 0.4) * 0.4))

    # Expense trend: regression on monthly totals (currency per month). Decreasing expense => positive score.
    expense_trend_slope = 0.0
    expense_trend_score = 0.5  # neutral
    if len(monthly) >= min_months_for_trend:
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly.values.reshape(-1, 1)
        reg = LinearRegression()
        try:
            reg.fit(X, y)
            slope = float(reg.coef_[0][0])  # currency per month, positive = increasing expense
            expense_trend_slope = slope
            # Map slope to score:
            # If slope <= -mean/10 => very good (expenses falling fast) -> ~1.0
            # slope around 0 -> 0.5 neutral
            # slope >= mean/10 -> bad -> ~0.0
            mean_monthly = float(np.mean(monthly))
            if mean_monthly <= 0:
                expense_trend_score = 0.5
            else:
                bound = mean_monthly / 10.0  # reasonable normalization
                if slope <= -bound:
                    expense_trend_score = 1.0
                elif slope >= bound:
                    expense_trend_score = 0.0
                else:
                    # linear between -bound..bound -> 1..0
                    expense_trend_score = float(np.clip(1.0 - (slope + bound) / (2 * bound), 0.0, 1.0))
        except Exception as e:
            logger.exception("Expense trend regression failed: %s", e)
            expense_trend_slope = 0.0
            expense_trend_score = 0.5

    # Compose final expense component score 0..100
    # Weights: EIR (0.6), discretionary (0.25), trend (0.15)
    w_eir = 0.6
    w_disc = 0.25
    w_trend = 0.15

    combined = w_eir * eir_score + w_disc * discretionary_score + w_trend * expense_trend_score
    expense_component_score = float(np.clip(combined * 100.0, 0.0, 100.0))

    return ExpenseFeatures(
        total_expenses=total_expenses,
        eir=eir,
        eir_score=eir_score,
        fixed_ratio=fixed_ratio,
        variable_ratio=variable_ratio,
        discretionary_ratio=discretionary_ratio,
        discretionary_score=discretionary_score,
        expense_trend_slope=expense_trend_slope,
        expense_trend_score=expense_trend_score,
        expense_component_score=expense_component_score,
    )