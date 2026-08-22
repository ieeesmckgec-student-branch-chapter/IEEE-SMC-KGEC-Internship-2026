"""
Combine all feature modules into a single pipeline call.

Usage:
  features = compute_all_features(df)
  features is a dict containing component scores and raw subfeatures for explainability.
"""
from __future__ import annotations
from typing import Dict, Any

from .income_features import compute_income_features
from .expense_features import compute_expense_features
from .cashflow_features import compute_cashflow_features
from .repayment_features import compute_repayment_features
from .behaviour_features import compute_behaviour_features

def compute_all_features(df, date_col="date", amount_col="amount", category_col="category", balance_col="running_balance") -> Dict[str, Any]:
    # Income features
    income = compute_income_features(df, date_col=date_col, amount_col=amount_col, source_col="payee")
    # Expense features (supply income_total as sum of positives)
    income_total = float(df[amount_col].where(df[amount_col] > 0).sum()) if amount_col in df.columns else 0.0
    expense = compute_expense_features(df, date_col=date_col, amount_col=amount_col, category_col=category_col, income_total=income_total)
    # Cashflow
    cashflow = compute_cashflow_features(df, date_col=date_col, balance_col=balance_col)
    # Repayment
    repayment = compute_repayment_features(df, date_col=date_col, amount_col=amount_col, category_col=category_col)
    # Behaviour
    behaviour = compute_behaviour_features(df, date_col=date_col, amount_col=amount_col, balance_col=balance_col, income_total=income_total)

    # Compose a flat dictionary of scores and raw features for downstream consumption & explainability
    result = {
        "income": income,
        "expense": expense,
        "cashflow": cashflow,
        "repayment": repayment,
        "behaviour": behaviour,
        # component scores ready to combine
        "component_scores": {
            "income": income.income_component_score,
            "expense": expense.expense_component_score,
            "cashflow": cashflow.cashflow_component_score,
            "repayment": repayment.repayment_component_score,
            # behaviour can be additional signal; keep but lower weight by default
            "behaviour": behaviour.behaviour_component_score,
        },
        "raw": {
            "income_raw": income,
            "expense_raw": expense,
            "cashflow_raw": cashflow,
            "repayment_raw": repayment,
            "behaviour_raw": behaviour,
        }
    }
    return result