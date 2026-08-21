"""
Credit score engine: weighted combination of component scores.

Final Score (0..100) =
  0.30 * income +
  0.20 * expense +
  0.20 * cashflow +
  0.30 * repayment

Then convert to 300..900:
Credit Score = 300 + FinalScore * 6
"""
from __future__ import annotations
from typing import Dict

def compute_final_score(component_scores: Dict[str, float], weights: Dict[str, float] = None) -> Dict[str, float]:
    if weights is None:
        weights = {"income": 0.30, "expense": 0.20, "cashflow": 0.20, "repayment": 0.30}
    # ensure keys exist
    for k in weights.keys():
        if k not in component_scores:
            component_scores[k] = 0.0
    final = 0.0
    for k, w in weights.items():
        final += component_scores[k] * w
    # final is 0..100
    final = max(0.0, min(100.0, final))
    credit_score = 300 + final * 6.0
    return {"final_score": final, "credit_score": round(credit_score, 2)}