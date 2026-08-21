from pathlib import Path
from typing import Any

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "heart_risk_model_bundle.joblib"
)


_model_bundle: dict[str, Any] | None = None


def load_model_bundle() -> dict[str, Any]:
    """
    Load the trained model bundle once and reuse it.
    """

    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle was not found at: {MODEL_PATH}"
        )

    loaded_bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "logistic_model",
        "random_forest_model",
        "xgboost_model",
        "feature_names",
        "ensemble_weights",
        "thresholds",
        "model_version",
    }

    missing_keys = required_keys.difference(
        loaded_bundle.keys()
    )

    if missing_keys:
        raise ValueError(
            "Model bundle is missing required values: "
            f"{sorted(missing_keys)}"
        )

    _model_bundle = loaded_bundle

    return _model_bundle


def get_model_information() -> dict[str, Any]:
    """
    Return safe information about the loaded models.
    """

    bundle = load_model_bundle()

    return {
        "status": "loaded",
        "model_version": bundle["model_version"],
        "feature_count": len(
            bundle["feature_names"]
        ),
        "feature_names": bundle["feature_names"],
        "ensemble_weights": bundle[
            "ensemble_weights"
        ],
        "thresholds": bundle["thresholds"],
        "study_population": bundle.get(
            "study_population"
        ),
        "models": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ],
    }


import numpy as np
import pandas as pd


def predict_heart_risk(
    input_features: dict[str, float],
) -> dict[str, Any]:
    """
    Generate a real ensemble prediction using the
    trained Logistic Regression, Random Forest and XGBoost models.
    """

    bundle = load_model_bundle()

    feature_names = bundle["feature_names"]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in input_features
    ]

    extra_features = [
        feature
        for feature in input_features
        if feature not in feature_names
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    if extra_features:
        raise ValueError(
            "Unexpected model features: "
            f"{extra_features}"
        )

    ordered_values = {
        feature: float(input_features[feature])
        for feature in feature_names
    }

    patient_df = pd.DataFrame(
        [ordered_values],
        columns=feature_names,
    )

    logistic_model = bundle["logistic_model"]
    random_forest_model = bundle[
        "random_forest_model"
    ]
    xgboost_model = bundle["xgboost_model"]

    weights = bundle["ensemble_weights"]
    thresholds = bundle["thresholds"]

    logistic_probability = float(
        logistic_model.predict_proba(
            patient_df
        )[0, 1]
    )

    random_forest_probability = float(
        random_forest_model.predict_proba(
            patient_df
        )[0, 1]
    )

    xgboost_probability = float(
        xgboost_model.predict_proba(
            patient_df
        )[0, 1]
    )

    ensemble_probability = float(
        weights["logistic"] * logistic_probability
        + weights["random_forest"]
        * random_forest_probability
        + weights["xgboost"] * xgboost_probability
    )

    screening_threshold = float(
        thresholds["screening"]
    )

    high_specificity_threshold = float(
        thresholds["high_specificity"]
    )

    if ensemble_probability >= high_specificity_threshold:
        risk_category = "high predicted risk"

    elif ensemble_probability >= screening_threshold:
        risk_category = "elevated predicted risk"

    else:
        risk_category = "lower predicted risk"

    return {
        "status": "success",
        "model_version": bundle["model_version"],
        "probability": round(
            ensemble_probability,
            6,
        ),
        "probability_percent": round(
            ensemble_probability * 100,
            2,
        ),
        "risk_category": risk_category,
        "screening_prediction": bool(
            ensemble_probability
            >= screening_threshold
        ),
        "high_specificity_prediction": bool(
            ensemble_probability
            >= high_specificity_threshold
        ),
        "thresholds": {
            "screening": round(
                screening_threshold,
                3,
            ),
            "high_specificity": round(
                high_specificity_threshold,
                3,
            ),
        },
        "component_probabilities": {
            "logistic_regression": round(
                logistic_probability,
                6,
            ),
            "random_forest": round(
                random_forest_probability,
                6,
            ),
            "xgboost": round(
                xgboost_probability,
                6,
            ),
        },
    }