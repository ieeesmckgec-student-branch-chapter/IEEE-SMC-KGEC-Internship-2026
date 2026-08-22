"""
Behavioral features:
- financial discipline (savings behavior)
- spending pattern stability (spikes)
- end-of-month stress (balance drops)
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
class BehaviourFeatures:
    savings_rate: float              # savings / income (0..1)
    financial_discipline_score: float # 0..1
    spending_spike_score: float       # 0..1 (higher = stable)
    end_of_month_stress_score: float  # 0..1 (higher = less stress)
    behaviour_component_score: float  # 0..100

def compute_behaviour_features(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
    balance_col: Optional[str] = "running_balance",
    income_total: Optional[float] = None,
    window_months: int = 6,
) -> BehaviourFeatures:
    if df is None or df.empty:
        return BehaviourFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    df_local = df.copy()
    df_local[amount_col] = pd.to_numeric(df_local[amount_col], errors="coerce").fillna(0.0)
    df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce")
    df_local = df_local.dropna(subset=[date_col])
    # Compute monthly totals using a pandas-safe conversion
    df_local["period"] = df_local[date_col].dt.to_period("M").dt.to_timestamp()
    monthly_net = df_local.groupby("period")[amount_col].sum().sort_index()  # positive if net inflow
    # savings estimate: positive net amounts per month (net inflow saved)
    positive_savings = monthly_net[monthly_net > 0].sum() if not monthly_net.empty else 0.0

    if income_total is None:
        # approximate income_total as sum of positive amounts
        income_total = float(df_local[df_local[amount_col] > 0][amount_col].sum())

    savings_rate = float(positive_savings / max(1.0, income_total)) if income_total > 0 else 0.0

    # financial discipline: combine consistent savings and consistent recurring payments like rent/emi
    if not monthly_net.empty:
        pos_monthly = monthly_net[monthly_net > 0]
        if not pos_monthly.empty:
            mean_pos = float(pos_monthly.mean())
            std_pos = float(pos_monthly.std(ddof=0))
            consistency = float(np.clip(1.0 - (std_pos / (mean_pos + 1e-9)), 0.0, 1.0))
        else:
            consistency = 0.5
    else:
        consistency = 0.5

    financial_discipline_score = float(np.clip(0.6 * savings_rate + 0.4 * consistency, 0.0, 1.0))

    # spending spike detection
    spend = df_local[df_local[amount_col] < 0].groupby("period")[amount_col].sum().abs().sort_index()
    if len(spend) >= 2:
        rel_changes = spend.pct_change().abs().dropna()
        median_change = float(rel_changes.median()) if not rel_changes.empty else 0.0
        spike_ratio = float((rel_changes > 2.0 * max(1e-6, median_change)).sum() / max(1, len(rel_changes)))
        spending_spike_score = float(np.clip(1.0 - spike_ratio, 0.0, 1.0))
    else:
        spending_spike_score = 1.0

    # end-of-month stress: robust numeric handling for running_balance
    stress_score = 1.0
    if balance_col and balance_col in df_local.columns:
        # compute monthly min balances then coerce to numeric and drop non-numeric values
        monthly_balance = df_local.groupby("period")[balance_col].min().sort_index()
        monthly_balance = pd.to_numeric(monthly_balance, errors="coerce").dropna()
        if not monthly_balance.empty:
            # use max positive balance as reference; if all balances <=0, use absolute max
            ref_max = monthly_balance.max()
            # if ref_max is 0 -> avoid division by zero; treat small positive values sensibly
            threshold = ref_max * 0.05 if ref_max != 0 else 0.0
            stress_count = int((monthly_balance < threshold).sum())
            stress_score = float(np.clip(1.0 - stress_count / max(1, len(monthly_balance)), 0.0, 1.0))
        else:
            stress_score = 1.0
    else:
        # fallback: check if monthly net is frequently near zero or negative
        if not monthly_net.empty:
            stress_count = int((monthly_net < 0).sum())
            stress_score = float(np.clip(1.0 - stress_count / max(1, len(monthly_net)), 0.0, 1.0))
        else:
            stress_score = 1.0

    # final behaviour component score 0..100
    combined = 0.4 * financial_discipline_score + 0.35 * spending_spike_score + 0.25 * stress_score
    behaviour_component_score = float(np.clip(combined * 100.0, 0.0, 100.0))

    return BehaviourFeatures(
        savings_rate=savings_rate,
        financial_discipline_score=financial_discipline_score,
        spending_spike_score=spending_spike_score,
        end_of_month_stress_score=stress_score,
        behaviour_component_score=behaviour_component_score,
    )