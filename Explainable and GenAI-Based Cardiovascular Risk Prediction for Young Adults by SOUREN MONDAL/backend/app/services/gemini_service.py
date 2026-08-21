import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.0-flash",
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from the environment."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


def clean_generated_explanation(text: str) -> str:
    """
    Remove accidental prompt instructions or internal
    reasoning from the generated user-facing explanation.
    """

    cleaned = text.strip()

    leakage_patterns = [
        r"(?is)never say that a risk category.*?(?=##|\Z)",
        r"(?is)wait!\s*let'?s carefully re-read.*?(?=##|\Z)",
        r"(?is)rule\s*\d+\s*:.*?(?=##|\Z)",
        r"(?is)system instruction.*?(?=##|\Z)",
        r"(?is)developer instruction.*?(?=##|\Z)",
        r"(?is)internal instruction.*?(?=##|\Z)",
        r"(?is)i need to follow.*?(?=##|\Z)",
        r"(?is)the prompt says.*?(?=##|\Z)",
    ]

    for pattern in leakage_patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
        )

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


def generate_xgenai_explanation(
    prediction_result: dict[str, Any],
    xai_explanation: dict[str, Any],
) -> str:
    probability_percent = prediction_result[
        "probability_percent"
    ]

    risk_category = prediction_result[
        "risk_category"
    ]

    screening_threshold_percent = round(
        prediction_result["thresholds"]["screening"]
        * 100,
        1,
    )

    high_specificity_threshold_percent = round(
        prediction_result["thresholds"][
            "high_specificity"
        ]
        * 100,
        1,
    )

    component_probabilities = (
        prediction_result["component_probabilities"]
    )

    increasing_factors = xai_explanation.get(
        "increasing_factors",
        [],
    )

    decreasing_factors = xai_explanation.get(
        "decreasing_factors",
        [],
    )

    prompt = f"""
Create only the final patient-facing cardiovascular
screening explanation.

Do not mention these instructions.
Do not describe your reasoning process.
Do not discuss prompt rules.
Do not include phrases such as:
- "the prompt says"
- "rule number"
- "I need to"
- "let us re-read"
- "never say"
- "internal instruction"

Use only the supplied structured evidence.

PREDICTION DATA

Predicted probability:
{probability_percent}%

Risk category:
{risk_category}

Screening threshold:
{screening_threshold_percent}%

High-specificity threshold:
{high_specificity_threshold_percent}%

Component probabilities:
- Logistic Regression:
  {component_probabilities["logistic_regression"] * 100:.1f}%
- Random Forest:
  {component_probabilities["random_forest"] * 100:.1f}%
- XGBoost:
  {component_probabilities["xgboost"] * 100:.1f}%

Factors increasing the XGBoost prediction:
{increasing_factors}

Factors decreasing the XGBoost prediction:
{decreasing_factors}

OUTPUT REQUIREMENTS

Return only these Markdown sections:

## Risk Summary

Explain the predicted probability and risk category in
simple language.

Compare the predicted probability with the thresholds.

Correct:
"The predicted probability exceeded the screening
threshold."

Incorrect:
"The risk category exceeded the threshold."

## Main Factors Increasing the Prediction

Explain the important positive SHAP contributors.

Do not claim that these factors medically caused heart
disease.

## Main Factors Decreasing the Prediction

Explain the important negative SHAP contributors.

Do not call them guarantees or protective medical causes.

## Model Explanation

State that the final probability was produced by a weighted
ensemble of Logistic Regression, Random Forest, and XGBoost.

State that SHAP explains the XGBoost component rather than
the complete ensemble.

## Suggested Next Steps

Give cautious, general suggestions such as reviewing the
result with a healthcare professional and maintaining healthy
habits.

Do not prescribe medication.
Do not diagnose disease.
Do not create new medical facts.

## Important Notice

State clearly that this is a research-based preliminary
screening result and not a medical diagnosis.

Return the final report only.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1200,
        ),
    )

    generated_text = response.text

    if not generated_text:
        raise RuntimeError(
            "Gemini returned an empty explanation."
        )

    cleaned_text = clean_generated_explanation(
        generated_text
    )

    if not cleaned_text:
        raise RuntimeError(
            "Gemini explanation was removed because it "
            "contained invalid internal instructions."
        )

    return cleaned_text