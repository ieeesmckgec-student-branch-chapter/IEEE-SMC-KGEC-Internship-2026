"""
Loan eligibility rules and loan amount calculation.
"""
from __future__ import annotations
from typing import Dict, Any

def evaluate_loan_eligibility(features: Dict[str, Any], final_score: float, config: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Returns a dict with decision, reasons, eligible_emi, approx_max_loan.
    features: should contain income.avg_monthly_income and repayment.bounce_rate and expense.eir etc.
    config: thresholds: min_income, bounce_rate_threshold, eir_threshold, score_threshold
    """
    if config is None:
        config = {
            "min_income": 10000.0,
            "bounce_rate_threshold": 0.05,
            "eir_threshold": 0.7,
            "score_threshold": 70.0,
            "emi_fraction": 0.30,
            "tenure_months": 36,
        }
    income_val = features.get("income", {}).get("avg_monthly_income", 0.0) if isinstance(features.get("income"), dict) else getattr(features.get("income"), "avg_monthly_income", 0.0)
    bounce_rate = features.get("repayment", {}).get("bounce_rate", 0.0) if isinstance(features.get("repayment"), dict) else getattr(features.get("repayment"), "bounce_rate", 0.0)
    eir = features.get("expense", {}).get("eir", 1.0) if isinstance(features.get("expense"), dict) else getattr(features.get("expense"), "eir", 1.0)

    reasons = []
    decision = "UNKNOWN"
    # Reject conditions
    if income_val < config["min_income"]:
        reasons.append("Income below minimum threshold")
    if bounce_rate > config["bounce_rate_threshold"]:
        reasons.append("High bounce rate")
    if eir > config["eir_threshold"]:
        reasons.append("High expense-to-income ratio")

    if reasons:
        decision = "REJECT"
    else:
        # score-based decisions
        if final_score >= config["score_threshold"]:
            decision = "APPROVE"
        elif final_score >= 60.0:
            decision = "CONDITIONAL"
            reasons.append("Score in conditional range")
        else:
            decision = "REJECT"
            reasons.append("Low credit score")

    # EMI capacity and loan amount calculation
    emi_capacity = income_val * config["emi_fraction"]
    approx_loan = emi_capacity * config["tenure_months"]

    return {
        "decision": decision,
        "reasons": reasons,
        "eligible_emi": round(emi_capacity, 2),
        "approx_max_loan": round(approx_loan, 2),
    }