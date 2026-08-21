import base64
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from app.services.model_service import load_model_bundle


_shap_explainer: Any | None = None


def get_shap_explainer():
    """
    Create the XGBoost SHAP explainer once and reuse it.
    """

    global _shap_explainer

    if _shap_explainer is not None:
        return _shap_explainer

    bundle = load_model_bundle()
    xgboost_model = bundle["xgboost_model"]

    _shap_explainer = shap.TreeExplainer(
        xgboost_model
    )

    return _shap_explainer


def create_patient_dataframe(
    input_features: dict[str, float],
) -> pd.DataFrame:
    """
    Create a one-row DataFrame using the model's
    original feature order.
    """

    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in input_features
    ]

    if missing_features:
        raise ValueError(
            "Missing SHAP input features: "
            f"{missing_features}"
        )

    ordered_features = {
        feature: float(input_features[feature])
        for feature in feature_names
    }

    return pd.DataFrame(
        [ordered_features],
        columns=feature_names,
    )


def extract_patient_shap_data(
    input_features: dict[str, float],
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    float,
]:
    """
    Calculate SHAP values for one user and normalize
    possible binary-class output shapes.
    """

    patient_df = create_patient_dataframe(
        input_features
    )

    explainer = get_shap_explainer()
    shap_result = explainer(patient_df)

    shap_values = np.asarray(
        shap_result.values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    patient_shap_values = np.asarray(
        shap_values[0],
        dtype=float,
    )

    base_values = np.asarray(
        shap_result.base_values
    )

    if base_values.ndim >= 2:
        if base_values.shape[-1] > 1:
            base_value = float(
                base_values.reshape(
                    -1,
                    base_values.shape[-1],
                )[0, 1]
            )
        else:
            base_value = float(
                base_values.reshape(-1)[0]
            )
    else:
        base_value = float(
            base_values.reshape(-1)[0]
        )

    return (
        patient_df,
        patient_shap_values,
        base_value,
    )


def figure_to_base64(
    figure: plt.Figure,
) -> str:
    """
    Convert a Matplotlib figure into a Base64 PNG
    data URL that can be displayed directly by React.
    """

    image_buffer = BytesIO()

    figure.savefig(
        image_buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    image_buffer.seek(0)

    encoded_image = base64.b64encode(
        image_buffer.read()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        f"{encoded_image}"
    )


def create_local_shap_explanation(
    input_features: dict[str, float],
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Explain one respondent's XGBoost prediction using SHAP.
    """

    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]

    (
        patient_df,
        patient_shap_values,
        base_value,
    ) = extract_patient_shap_data(
        input_features
    )

    contributors = []

    for feature, value, shap_value in zip(
        feature_names,
        patient_df.iloc[0].values,
        patient_shap_values,
    ):
        shap_value = float(shap_value)

        contributors.append({
            "feature": feature,
            "value": float(value),
            "shap_value": round(
                shap_value,
                6,
            ),
            "absolute_shap_value": round(
                abs(shap_value),
                6,
            ),
            "effect": (
                "increases predicted risk"
                if shap_value > 0
                else "decreases predicted risk"
                if shap_value < 0
                else "has almost no effect"
            ),
        })

    contributors.sort(
        key=lambda item: item[
            "absolute_shap_value"
        ],
        reverse=True,
    )

    top_contributors = contributors[:top_n]

    increasing_factors = [
        item
        for item in top_contributors
        if item["shap_value"] > 0
    ]

    decreasing_factors = [
        item
        for item in top_contributors
        if item["shap_value"] < 0
    ]

    return {
        "explained_model": "XGBoost",
        "explanation_method": "SHAP TreeExplainer",
        "base_value": round(
            base_value,
            6,
        ),
        "top_contributors": top_contributors,
        "increasing_factors": increasing_factors,
        "decreasing_factors": decreasing_factors,
        "important_note": (
            "SHAP explains the XGBoost component of the "
            "ensemble. It describes model influence, not "
            "medical causation."
        ),
    }


def create_shap_waterfall_image(
    input_features: dict[str, float],
    max_display: int = 10,
) -> str:
    """
    Create a SHAP waterfall plot for one user.
    """

    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]

    (
        patient_df,
        patient_shap_values,
        base_value,
    ) = extract_patient_shap_data(
        input_features
    )

    explanation = shap.Explanation(
        values=patient_shap_values,
        base_values=base_value,
        data=patient_df.iloc[0].values,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6.5))

    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False,
    )

    figure = plt.gcf()

    figure.suptitle(
        "SHAP Waterfall Plot — XGBoost Component",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    figure.text(
        0.5,
        -0.02,
        (
            "Red contributions increase the XGBoost output; "
            "blue contributions decrease it."
        ),
        ha="center",
        fontsize=9,
    )

    return figure_to_base64(figure)


def create_shap_bar_image(
    input_features: dict[str, float],
    max_display: int = 10,
) -> str:
    """
    Create a horizontal SHAP contribution bar chart.
    """

    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]

    (
        patient_df,
        patient_shap_values,
        _,
    ) = extract_patient_shap_data(
        input_features
    )

    contribution_data = []

    for feature, value, shap_value in zip(
        feature_names,
        patient_df.iloc[0].values,
        patient_shap_values,
    ):
        contribution_data.append({
            "feature": feature.replace("_", " "),
            "value": float(value),
            "shap_value": float(shap_value),
            "absolute_value": abs(
                float(shap_value)
            ),
        })

    contribution_data.sort(
        key=lambda item: item["absolute_value"],
        reverse=True,
    )

    top_contributions = contribution_data[
        :max_display
    ]

    top_contributions.reverse()

    labels = [
        (
            f'{item["feature"]} '
            f'= {item["value"]:g}'
        )
        for item in top_contributions
    ]

    values = [
        item["shap_value"]
        for item in top_contributions
    ]

    colors = [
        "#d94f64"
        if value > 0
        else "#2f7dbd"
        for value in values
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6.5)
    )

    axis.barh(
        labels,
        values,
        color=colors,
        alpha=0.9,
    )

    axis.axvline(
        0,
        color="#243247",
        linewidth=1,
    )

    axis.set_title(
        "Top SHAP Feature Contributions",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    axis.set_xlabel(
        "SHAP contribution to XGBoost model output"
    )

    axis.set_ylabel("Feature and submitted value")

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    for index, value in enumerate(values):
        horizontal_alignment = (
            "left"
            if value >= 0
            else "right"
        )

        offset = (
            3
            if value >= 0
            else -3
        )

        axis.annotate(
            f"{value:+.3f}",
            xy=(value, index),
            xytext=(offset, 0),
            textcoords="offset points",
            va="center",
            ha=horizontal_alignment,
            fontsize=8,
        )

    figure.text(
        0.5,
        0.01,
        (
            "Red increases the XGBoost output; "
            "blue decreases the XGBoost output."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    return figure_to_base64(figure)


def create_shap_visualizations(
    input_features: dict[str, float],
    max_display: int = 10,
) -> dict[str, str]:
    """
    Generate all patient-level SHAP images.
    """

    return {
        "waterfall_image": (
            create_shap_waterfall_image(
                input_features=input_features,
                max_display=max_display,
            )
        ),
        "bar_image": create_shap_bar_image(
            input_features=input_features,
            max_display=max_display,
        ),
        "explained_model": "XGBoost",
        "important_note": (
            "These plots explain the XGBoost component, "
            "not the complete weighted ensemble."
        ),
    }