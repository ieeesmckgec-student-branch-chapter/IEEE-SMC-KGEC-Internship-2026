#!/usr/bin/env python3
# scripts/api.py
from __future__ import annotations
import os
import tempfile
import uuid
import logging
import traceback
import dataclasses
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from numbers import Number
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Cookie, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Pipeline imports (must be importable from repo root)
from preprocessing.loader import load_transaction_files
from preprocessing.categorizer import categorize_transactions
from feature_engineering.feature_engineering import compute_all_features
from scoring.credit_score import compute_final_score
from scoring.loan_eligibility import evaluate_loan_eligibility
from scoring.risk_analysis import risk_bucket_from_score
from scoring.explainability import generate_explanations

# Optional ML helpers
try:
    from preprocessing.ml_categorizer import load_model as ml_load_model, predict_categories  # type: ignore
    ML_HELPERS_AVAILABLE = True
except Exception:
    ml_load_model = None  # type: ignore
    predict_categories = None  # type: ignore
    ML_HELPERS_AVAILABLE = False

# joblib fallback loader
try:
    from joblib import load as joblib_load  # type: ignore
except Exception:
    joblib_load = None  # type: ignore

logger = logging.getLogger("prosperity_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="ProsperityScore API (with summary store)")

# CORS configuration: must be placed immediately after app creation.
# Set FRONTEND_ORIGINS in your environment to the exact origin(s) of your frontend,
# e.g. "https://app.example.com" or "http://localhost:3000".
# Example: hard-coded allowed origins (three example origins)
FRONTEND_ORIGINS = [
    "https://prosperityscore.vercel.app",
    "http://localhost:8080",
    "https://prosperityscore.krishnendughosh.in",
]
logger.info("CORS allow_origins (hard-coded): %s", FRONTEND_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Config / Model
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/ml_model.pkl"))
USE_ML_ENV = os.environ.get("USE_ML", "false").lower()
REQUEST_USE_ML = USE_ML_ENV in ("1", "true", "yes")

ML_MODEL = None
ML_MODEL_LOADED = False


def try_load_ml_model():
    """Attempt to load ML model into ML_MODEL if REQUEST_USE_ML is true."""
    global ML_MODEL, ML_MODEL_LOADED
    if not REQUEST_USE_ML:
        logger.info("USE_ML not enabled; ML disabled.")
        ML_MODEL = None
        ML_MODEL_LOADED = False
        return

    if not MODEL_PATH.exists():
        logger.warning("USE_ML enabled but MODEL_PATH not found: %s", MODEL_PATH)
        ML_MODEL = None
        ML_MODEL_LOADED = False
        return

    if ML_HELPERS_AVAILABLE and ml_load_model is not None:
        try:
            ML_MODEL = ml_load_model(str(MODEL_PATH))
            ML_MODEL_LOADED = True
            logger.info("Loaded ML model via preprocessing.ml_categorizer.load_model from %s", MODEL_PATH)
            return
        except Exception:
            logger.exception("preprocessing.ml_categorizer.load_model failed; falling back to joblib.load if available")

    if joblib_load is not None:
        try:
            ML_MODEL = joblib_load(str(MODEL_PATH))
            ML_MODEL_LOADED = True
            logger.info("Loaded ML model via joblib from %s", MODEL_PATH)
            return
        except Exception:
            logger.exception("joblib.load failed to load model at %s", MODEL_PATH)

    logger.warning("ML model could not be loaded despite REQUEST_USE_ML; ML disabled.")
    ML_MODEL = None
    ML_MODEL_LOADED = False

try_load_ml_model()

# In-memory summary store for quick frontend autofill (dev/test use only)
SUMMARIES: Dict[str, Dict[str, Any]] = {}
SUMMARY_TTL_SECONDS = int(os.environ.get("SUMMARY_TTL", "600"))  # TTL not enforced in this simple store


def _to_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable primitives.
    Handles dataclasses, pandas (Series, Timestamp), numpy scalars/arrays, dicts, lists, tuples, sets.
    Falls back to str(obj) for unknown types.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Number):
        try:
            # numpy scalars respond to .item()
            return obj.item()
        except Exception:
            return obj
    if dataclasses.is_dataclass(obj):
        try:
            return _to_json_serializable(dataclasses.asdict(obj))
        except Exception:
            return _to_json_serializable(dict(obj))
    if isinstance(obj, pd.Series):
        try:
            return {str(k): _to_json_serializable(v) for k, v in obj.to_dict().items()}
        except Exception:
            return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        try:
            return [_to_json_serializable(dict(row)) for _, row in obj.iterrows()]
        except Exception:
            return obj.to_dict()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        try:
            return _to_json_serializable(obj.tolist())
        except Exception:
            return [ _to_json_serializable(x) for x in obj ]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[str(k)] = _to_json_serializable(v)
            except Exception:
                out[str(k)] = str(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return _to_json_serializable(obj.tolist())
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return None


@app.get("/health", response_class=JSONResponse)
def health():
    return JSONResponse(
        content={
            "status": "ok",
            "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
            "request_use_ml": REQUEST_USE_ML,
            "ml_helpers_available": ML_HELPERS_AVAILABLE,
            "ml_model_loaded": ML_MODEL_LOADED,
            "cwd": str(Path.cwd()),
            "frontend_origins": FRONTEND_ORIGINS,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Build the two dynamic values separately
    health_data = health().body
    try:
        health_json = health_data.decode() if isinstance(health_data, (bytes, bytearray)) else str(health_data)
    except Exception:
        health_json = str(health_data)

    # Ensure score_url is a plain string (URL object -> str)
    score_url = str(request.url_for("score_file"))

    # Use a plain triple-quoted string (no f-string) and inject the dynamic parts by concatenation.
    html = """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"/><title>ProsperityScore API</title></head>
      <body style="font-family:Arial,Helvetica,sans-serif; margin:30px;">
        <h1>ProsperityScore API</h1>
        <p>Health: <pre>""" + health_json + """</pre></p>
        <h3>Upload a CSV / XLS / XLSX bank statement</h3>
        <input id="file" type="file" accept=".csv,.xls,.xlsx"/>
        <button id="send">Upload & Score</button>
        <pre id="out"></pre>
        <script>
          document.getElementById('send').onclick = async function () {
            const f = document.getElementById('file').files[0];
            if (!f) { alert('Select a file'); return; }
            const fd = new FormData();
            fd.append('file', f);
            document.getElementById('out').textContent = 'Uploading...';
            try {
              const resp = await fetch('""" + score_url + """', { method: 'POST', body: fd });
              const j = await resp.json();
              document.getElementById('out').textContent = JSON.stringify(j, null, 2);
            } catch (e) {
              document.getElementById('out').textContent = 'Error: ' + e;
            }
          };
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

def _pick_top_features(component_scores: Dict[str, float], raw: Dict[str, Any], top_k: int = 2) -> List[Dict[str, Any]]:
    comps_sorted = sorted(component_scores.items(), key=lambda kv: abs(kv[1] - 50), reverse=True)
    selected = []
    for comp_name, _ in comps_sorted[:top_k]:
        comp_raw = raw.get(f"{comp_name}_raw") or raw.get(comp_name) or {}
        if not isinstance(comp_raw, dict):
            comp_raw = getattr(comp_raw, "__dict__", {}) or {}
        score_keys = [k for k in comp_raw.keys() if k.endswith("_score")]
        if score_keys:
            worst_key = min(score_keys, key=lambda k: float(comp_raw.get(k) or 0.0))
            selected.append({
                "component": comp_name,
                "metric": worst_key,
                "value": float(comp_raw.get(worst_key) or 0.0),
                "note": f"Lowest subscore: {worst_key}"
            })
        else:
            numeric = {k: v for k, v in comp_raw.items() if isinstance(v, (int, float, np.integer, np.floating))}
            if numeric:
                key = max(numeric.keys(), key=lambda kk: abs(numeric[kk]))
                selected.append({
                    "component": comp_name,
                    "metric": key,
                    "value": numeric[key],
                    "note": f"Strong numeric signal: {key}"
                })
            else:
                selected.append({"component": comp_name, "metric": None, "value": None, "note": "No numeric detail"})
    return selected


@app.post("/score", response_class=JSONResponse)
async def score_file(file: UploadFile = File(...)):
    """
    Accept one uploaded CSV/XLS/XLSX file, parse using preprocessing.loader.load_transaction_files,
    (optionally) categorize with ML, else rule-based, compute features & scoring and return JSON.
    Also stores a compact parsed summary in-memory and sets an HttpOnly cookie so the frontend can fetch it.
    """
    fname = (file.filename or "upload").strip()
    ext = Path(fname).suffix.lower()
    if ext not in (".csv", ".xls", ".xlsx"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv, .xls, or .xlsx")

    tmp_path = None
    try:
        # write uploaded content to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
            tmp.flush()

        # parse
        try:
            df = load_transaction_files([tmp_path])
        except Exception as e:
            logger.exception("Loader failed to parse uploaded file: %s", e)
            raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {e}")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file parsed but returned no transactions")

        # categorize
        df_cat = None
        if REQUEST_USE_ML and ML_MODEL_LOADED and predict_categories is not None:
            try:
                df_cat = predict_categories(df, text_cols=("description", "payee"), model=ML_MODEL, out_col="category")
                logger.info("Categorized transactions using ML predict_categories")
            except Exception as e:
                logger.exception("ML predict_categories failed, falling back to rule-based: %s", e)
                df_cat = None

        if df_cat is None:
            try:
                df_cat = categorize_transactions(df, text_cols=("description", "payee"), use_ml=False)
                logger.info("Categorized transactions using rule-based categorize_transactions")
            except Exception as e:
                logger.exception("Rule-based categorizer failed; proceeding with uncategorized: %s", e)
                df_cat = df.copy()
                df_cat["category"] = "Uncategorized"

        # compute features
        try:
            features = compute_all_features(df_cat)
        except Exception as e:
            logger.exception("Feature computation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

        # scoring pipeline
        try:
            scoring = compute_final_score(features["component_scores"])
            eligibility = evaluate_loan_eligibility(features, scoring["final_score"])
            risk = risk_bucket_from_score(scoring["final_score"])
            explanations = generate_explanations(features["raw"])
        except Exception as e:
            logger.exception("Scoring pipeline failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Scoring pipeline failed: {e}")

        # pick top features
        try:
            top_features = _pick_top_features(features["component_scores"], features["raw"], top_k=2)
        except Exception:
            top_features = []

        resp = {
            "component_scores": features["component_scores"],
            "final_score": scoring,
            "eligibility": eligibility,
            "risk": risk,
            "explanations": explanations,
            "top_features": top_features,
        }

        # Build a compact summary for quick frontend autofill (UI-friendly; numeric fields default to 0)
        summary_id = str(uuid.uuid4())

        def _get_raw(path: List[str], default=None):
            try:
                obj = features.get("raw", {})
                for p in path:
                    if obj is None:
                        return default
                    if isinstance(obj, dict):
                        obj = obj.get(p)
                    else:
                        obj = getattr(obj, p, None)
                return obj if obj is not None else default
            except Exception:
                return default

        def _round_or_none(v):
            try:
                if v is None:
                    return None
                if isinstance(v, (float, np.floating)) and np.isnan(v):
                    return None
                return int(round(float(v)))
            except Exception:
                return None

        def infer_employment_type_from_features():
            emp = _get_raw(["income", "inferred_employment_type"], None) or _get_raw(["income_raw", "inferred_employment_type"], None)
            if emp and isinstance(emp, str):
                emp = emp.strip()
                if emp in ("Salaried", "Self-employed", "Business owner", "Gig / freelance"):
                    return emp
            avg_inc = _get_raw(["income", "avg_monthly_income"], None) or _get_raw(["income_raw", "avg_monthly_income"], None)
            stability = _get_raw(["income", "income_stability_score"], None) or _get_raw(["income_raw", "income_stability_score"], None)
            diversity = _get_raw(["income", "income_source_diversity_score"], None) or _get_raw(["income_raw", "income_source_diversity_score"], None)
            try:
                diversity = float(diversity) if diversity is not None else None
            except Exception:
                diversity = None
            if diversity is not None:
                if diversity >= 0.9:
                    return "Salaried"
                if diversity >= 0.6:
                    return "Self-employed"
                if diversity < 0.4:
                    if stability is not None and float(stability) < 0.4:
                        return "Gig / freelance"
                    return "Business owner"
            if stability is not None and float(stability) >= 0.8 and avg_inc and float(avg_inc) > 0:
                return "Salaried"
            return None

        def detect_monthly_rent():
            rent = _get_raw(["expense", "monthly_rent_estimate"], None) or _get_raw(["expense", "rent_monthly"], None)
            if rent is not None:
                return _round_or_none(rent)
            cat_totals = _get_raw(["expense", "category_totals"], None) or _get_raw(["expense_raw", "category_totals"], None)
            if isinstance(cat_totals, dict):
                for k in ("rent", "house rent", "rental"):
                    if k in cat_totals:
                        return _round_or_none(cat_totals[k])
            return None

        def detect_existing_emi():
            emi_amt = _get_raw(["repayment", "emi_monthly_total"], None) or _get_raw(["repayment", "monthly_emi_total"], None)
            if emi_amt is not None:
                return _round_or_none(emi_amt)
            repayment_cat = _get_raw(["repayment", "category_totals"], None)
            if isinstance(repayment_cat, dict):
                emi_like = repayment_cat.get("emi") or repayment_cat.get("loan_repayment") or repayment_cat.get("loan")
                if emi_like is not None:
                    return _round_or_none(emi_like)
            return None

        def detect_other_loans_count():
            c = _get_raw(["repayment", "emi_count"], None) or _get_raw(["repayment", "loan_count"], None)
            try:
                return int(c) if c is not None else None
            except Exception:
                return None

        def detect_utility_bills_on_time():
            ub = _get_raw(["behaviour", "utility_bills_on_time"], None)
            if isinstance(ub, bool):
                return ub
            eom = _get_raw(["behaviour", "end_of_month_stress_score"], None)
            neg_bal = _get_raw(["cashflow", "negative_balance_count"], None)
            try:
                if eom is not None:
                    if float(eom) >= 0.6 and (neg_bal is None or int(neg_bal) == 0):
                        return True
                    if float(eom) < 0.4:
                        return False
            except Exception:
                pass
            return None

        monthly_income_val = _get_raw(["income", "avg_monthly_income"], None) \
            or _get_raw(["income_raw", "avg_monthly_income"], None) \
            or _get_raw(["income", "effective_total_income"], None) \
            or _get_raw(["income_raw", "effective_total_income"], None)

        # robust avg balance lookup (extra candidate keys)
        try:
            logger.debug("features.raw keys: %s", list(features.get("raw", {}).keys()))
            logger.debug("raw cashflow object: %s", features.get("raw", {}).get("cashflow"))
        except Exception:
            pass

        avg_bank_balance_val = (
            _get_raw(["cashflow", "avg_monthly_balance"], None)
            or _get_raw(["cashflow_raw", "avg_monthly_balance"], None)
            or _get_raw(["cashflow", "avg_balance"], None)
            or _get_raw(["cashflow_raw", "avg_balance"], None)
            or _get_raw(["cashflow", "avg_monthly_balance_effective"], None)
            or _get_raw(["cashflow", "avg_balance_monthly"], None)
            or _get_raw(["cashflow", "avg_balance_inr"], None)
        )

        monthly_rent_val = detect_monthly_rent()
        existing_emi_val = detect_existing_emi()
        other_loans_val = detect_other_loans_count()
        utility_bills_on_time_val = detect_utility_bills_on_time()
        employment_type_val = infer_employment_type_from_features()

        # UI-friendly: return 0 for numeric missing values (per your request)
        summary_obj = {
            "id": summary_id,
            "component_scores": features.get("component_scores"),
            "features": {
                "income": features.get("raw", {}).get("income") or features.get("raw", {}).get("income_raw"),
                "cashflow": features.get("raw", {}).get("cashflow"),
                "repayment": features.get("raw", {}).get("repayment"),
                "expense": features.get("raw", {}).get("expense"),
                "behaviour": features.get("raw", {}).get("behaviour"),
            },
            "top_features": top_features,
            "final_score": scoring,
            "created_at": str(pd.Timestamp.now()),

            # UI-specific keys for frontend autofill — use 0 defaults to show 0 in UI
            "monthlyIncome": _round_or_none(monthly_income_val) if _round_or_none(monthly_income_val) is not None else 0,
            "avgBankBalance": _round_or_none(avg_bank_balance_val) if _round_or_none(avg_bank_balance_val) is not None else 0,
            "monthlyRent": _round_or_none(monthly_rent_val) if _round_or_none(monthly_rent_val) is not None else 0,
            "existingEmi": _round_or_none(existing_emi_val) if _round_or_none(existing_emi_val) is not None else 0,
            "otherLoans": int(other_loans_val) if other_loans_val is not None else 0,
            "utilityBillsOnTime": bool(utility_bills_on_time_val) if utility_bills_on_time_val is not None else False,
            "employmentType": employment_type_val or "Salaried",
        }

        # Convert summary to JSON-serializable form before storing/returning
        try:
            serializable_summary = _to_json_serializable(summary_obj)
            SUMMARIES[summary_id] = serializable_summary
        except Exception:
            logger.exception("Failed to serialize statement summary; saving fallback string")
            serializable_summary = {"id": summary_id, "note": "serialization_failed"}
            SUMMARIES[summary_id] = serializable_summary

        # include the id and the parsed (serializable) summary directly in the /score response
        resp["_statement_id"] = summary_id
        resp["_statement_summary"] = serializable_summary

        # Convert entire response to serializable form too (protects against numpy/pandas inside scoring result)
        try:
            serializable_resp = _to_json_serializable(resp)
        except Exception:
            logger.exception("Failed to serialize full response; falling back to minimal response")
            serializable_resp = {
                "component_scores": resp.get("component_scores"),
                "final_score": str(resp.get("final_score")),
                "_statement_id": summary_id,
                "_statement_summary": serializable_summary,
            }

        # set cookie for fallback flows (use secure + samesite=None when FORCE_SECURE_COOKIES=true)
        cookie_secure = os.environ.get("FORCE_SECURE_COOKIES", "false").lower() in ("1", "true", "yes")
        samesite_val = "None" if cookie_secure else "Lax"
        jr = JSONResponse(content=serializable_resp)
        jr.set_cookie("statement_id", summary_id, max_age=SUMMARY_TTL_SECONDS, httponly=True,
                      secure=cookie_secure, samesite=samesite_val, path="/")
        return jr

    finally:
        # cleanup temp file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.exception("Failed to remove temporary file %s", tmp_path)


@app.get("/api/statement/summary", response_class=JSONResponse)
def get_statement_summary(id: str | None = Query(None, alias="id"), statement_id_cookie: str | None = Cookie(None)):
    """
    Return the parsed statement summary for the current client.
    Accepts either ?id=<uuid> or the HttpOnly cookie 'statement_id'.
    """
    sid = id or statement_id_cookie
    if not sid:
        raise HTTPException(status_code=404, detail="No statement id provided")

    summary = SUMMARIES.get(sid)
    if not summary:
        raise HTTPException(status_code=404, detail="No parsed statement summary available for this id")
    return JSONResponse(content=summary)
