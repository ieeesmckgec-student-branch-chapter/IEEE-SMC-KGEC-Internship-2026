"""
Produce prioritized, human-readable reasons (3-4) explaining score & decision.
The function builds candidate reasons with impact scores, ranks them, and returns
the top max_reasons messages (prefer negative/important signals first, then strengths).
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


def _get(f: Optional[object], key: str, default=None):
    """Support both dicts and objects with attributes."""
    if f is None:
        return default
    if isinstance(f, dict):
        return f.get(key, default)
    return getattr(f, key, default)

def generate_explanations(features: Dict[str, Any], max_reasons: int = 4) -> List[str]:
    """
    Produce 2-4 professional, prioritized explanations for the score:
     - 1-2 primary reasons explaining the current score (negatives first)
     - 1-2 'why not higher' / remediation suggestions describing what prevented a better score

    The function accepts the same features dict shape as before (components may be dicts or objects).
    """
    def _get(f: Optional[object], key: str, default=None):
        if f is None:
            return default
        if isinstance(f, dict):
            return f.get(key, default)
        return getattr(f, key, default)

    candidates: List[Tuple[float, str, str]] = []  # (severity, reason, improvement)

    income = features.get("income", {}) or {}
    expense = features.get("expense", {}) or {}
    cashflow = features.get("cashflow", {}) or {}
    repayment = features.get("repayment", {}) or {}
    behaviour = features.get("behaviour", {}) or {}

    # shortcuts
    inc_comp = _get(income, "income_component_score")
    inc_growth = _get(income, "income_growth_score") or 0.0
    avg_income = _get(income, "avg_monthly_income")

    eir = _get(expense, "eir")
    disc_ratio = _get(expense, "discretionary_ratio") or 0.0
    total_expenses = _get(expense, "total_expenses")

    neg_bal_count = _get(cashflow, "negative_balance_count") or 0
    min_balance = _get(cashflow, "min_monthly_balance")
    cash_comp = _get(cashflow, "cashflow_component_score")

    bounce_count = _get(repayment, "bounce_count") or 0
    on_time_ratio = _get(repayment, "on_time_ratio")
    emi_count = _get(repayment, "emi_count") or 0
    rep_comp = _get(repayment, "repayment_component_score")

    savings_rate = _get(behaviour, "savings_rate")
    discipline = _get(behaviour, "financial_discipline_score")
    spike = _get(behaviour, "spending_spike_score")
    eom_stress = _get(behaviour, "end_of_month_stress_score")

    # Negative / risk signals (high severity first)
    if bounce_count and bounce_count > 0:
        candidates.append((10.0,
            f"Transaction bounces detected ({int(bounce_count)}). Bounces strongly reduce repayment reliability.",
            "Resolve bounced transactions and maintain a positive balance; contact your bank to reverse/rectify bounces."))

    if on_time_ratio is not None and on_time_ratio < 0.9:
        candidates.append((9.0,
            f"Repayment timeliness below 90% (on-time ratio {on_time_ratio:.2f}). This increases default risk.",
            "Improve on-time payments (aim >95%): set up autopay, reminders, or consolidate payments to reduce missed dates."))

    if neg_bal_count and neg_bal_count > 0:
        candidates.append((8.5,
            f"Negative balance occurrences detected ({int(neg_bal_count)} times) indicating month-end liquidity stress.",
            "Build a buffer or smooth cash flows (move salary earlier, reduce variable outflows) to avoid negative balances."))

    if inc_comp is not None and inc_comp < 40:
        candidates.append((8.0,
            f"Low income stability (income score {inc_comp:.0f}/100) — income may be irregular or insufficient.",
            "Provide longer income history, add a co-borrower, or supply additional income documentation to demonstrate stability."))

    if eir is not None and eir >= 0.6:
        candidates.append((8.0,
            f"Very high expense-to-income ratio ({eir:.2f}); limited spare capacity for new debt.",
            "Reduce discretionary spending or consolidate recurring expenses to lower your expense-to-income ratio."))

    if savings_rate is not None and savings_rate < 0.1:
        candidates.append((7.5,
            f"Low savings rate ({savings_rate:.2f}) — limited buffer against shocks.",
            "Increase savings to build a contingency buffer (target at least 1–3 months of essential expenses)."))

    if emi_count and emi_count >= 4:
        candidates.append((7.0,
            f"Multiple active EMIs ({int(emi_count)}) constraining borrowing capacity.",
            "Consider consolidating loans or paying down high-cost EMIs to free up capacity for new credit."))

    if disc_ratio and disc_ratio > 0.4:
        candidates.append((6.5,
            f"High discretionary spending (ratio {disc_ratio:.2f}) reducing affordability.",
            "Identify and reduce non-essential expenses to improve affordability metrics."))

    if cash_comp is not None and cash_comp < 50:
        candidates.append((6.0,
            f"Cashflow stability is low (cashflow score {cash_comp:.0f}/100).",
            "Stabilize cash inflows and avoid large, irregular outflows to improve cashflow stability."))

    if discipline is not None and discipline < 0.4:
        candidates.append((6.0,
            "Low financial discipline score — inconsistent savings or spending patterns observed.",
            "Adopt a budgeting routine, automate savings, and reduce impulsive spending to raise discipline metrics."))

    if spike is not None and spike > 0.7:
        candidates.append((5.5,
            "Recent large spending spikes detected — potential volatility in monthly budget.",
            "Limit irregular large expenses or spread them across months; document one-off expenses if necessary."))

    if eom_stress is not None and eom_stress < 0.4:
        candidates.append((5.0,
            "End-of-month stress indicator low — months may end with tight cash positions.",
            "Improve month-end liquidity through savings or adjusting timing of receipts/payments."))

    # Positive / strength signals (lower severity values so negatives chosen first)
    if rep_comp is not None and rep_comp >= 95:
        candidates.append((2.0,
            "Excellent repayment history (no recent bounces; consistent on-time payments).",
            "Maintain timely repayments to preserve credit profile."))

    if inc_comp is not None and inc_comp >= 80:
        candidates.append((2.5,
            f"Stable and sufficient income (income score {inc_comp:.0f}/100).",
            "Continue steady employment/income and provide documentation if applying for higher limits."))

    if cash_comp is not None and cash_comp >= 80:
        candidates.append((2.5,
            f"Strong cashflow stability (cashflow score {cash_comp:.0f}/100).",
            "Keep cashflow stable and avoid voluntary overdrafts."))

    # Sort by severity descending
    candidates_sorted = sorted(candidates, key=lambda x: -x[0])

    primary_reasons: List[str] = []
    improvements: List[str] = []

    # choose up to 2 primary reasons (negatives first due to sorting)
    for severity, reason, improvement in candidates_sorted:
        if len(primary_reasons) >= 2:
            break
        # pick high severity first (>=5) as primary; but if no high severity, allow strengths
        if severity >= 5 or not any(s >= 5 for s, _, _ in candidates_sorted):
            primary_reasons.append(reason)

    # if no primary reasons found (unlikely), add a neutral positive
    if not primary_reasons:
        primary_reasons.append("No major risk signals detected — core affordability and repayment indicators are acceptable.")

    # Build improvements list: pick top 1-2 distinct remediation messages (avoid duplicates)
    seen_impr = set()
    for _, reason, improvement in candidates_sorted:
        if len(improvements) >= 2:
            break
        if improvement and improvement not in seen_impr:
            improvements.append(improvement)
            seen_impr.add(improvement)

    # If no improvements (very strong profile), add maintenance guidance
    if not improvements:
        improvements.append("To improve further, maintain steady income, on-time payments, and a healthy savings buffer.")

    # Compose final messages: professional phrasing
    final: List[str] = []
    # Add primary reasons (1-2)
    for r in primary_reasons:
        final.append(f"Reason: {r}")
    # Add why-not/higher (1-2)
    for imp in improvements:
        final.append(f"To improve your score: {imp}")

    # Ensure at least 2 items
    if len(final) < 2:
        final.append("No additional details available. Please provide more transaction history or documentation for a fuller assessment.")

    # Truncate to max_reasons
    return final[:max_reasons]
