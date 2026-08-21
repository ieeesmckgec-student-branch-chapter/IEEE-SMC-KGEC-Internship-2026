def risk_bucket_from_score(final_score: float) -> dict:
    if final_score >= 80:
        return {"bucket": "Low Risk"}
    if final_score >= 60:
        return {"bucket": "Medium Risk"}
    if final_score >= 40:
        return {"bucket": "High Risk"}
    return {"bucket": "Very High Risk"}