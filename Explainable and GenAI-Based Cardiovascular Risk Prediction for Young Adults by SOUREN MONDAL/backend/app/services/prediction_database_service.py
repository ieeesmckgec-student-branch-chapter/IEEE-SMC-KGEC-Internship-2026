from datetime import datetime, timezone
from typing import Any

from app.services.firebase_service import (
    get_firestore_client,
)


def save_prediction_result(
    user_id: str,
    user_email: str | None,
    request_data: dict[str, Any],
    prediction_result: dict[str, Any],
) -> str:
    """
    Save one authenticated prediction in Firestore.
    """

    database = get_firestore_client()

    prediction_document = {
        "user_id": user_id,
        "user_email": user_email,

        "probability": prediction_result[
            "probability"
        ],
        "probability_percent": prediction_result[
            "probability_percent"
        ],
        "risk_category": prediction_result[
            "risk_category"
        ],

        "screening_prediction": prediction_result[
            "screening_prediction"
        ],
        "high_specificity_prediction": (
            prediction_result[
                "high_specificity_prediction"
            ]
        ),

        "thresholds": prediction_result[
            "thresholds"
        ],

        "component_probabilities": (
            prediction_result[
                "component_probabilities"
            ]
        ),

        "input_data": request_data,

        "processed_features": prediction_result[
            "processed_features"
        ],

        "xai_explanation": prediction_result[
            "xai_explanation"
        ],

        "xgenai_explanation": prediction_result[
            "xgenai_explanation"
        ],

        "model_version": prediction_result[
            "model_version"
        ],

        "created_at": datetime.now(
            timezone.utc
        ),
    }

    document_reference = (
        database.collection("predictions")
        .document()
    )

    document_reference.set(
        prediction_document
    )

    return document_reference.id
from google.cloud.firestore_v1.base_query import FieldFilter


def get_user_predictions(
    user_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return the authenticated user's latest predictions.
    """

    database = get_firestore_client()

    query = (
        database.collection("predictions")
        .where(
            filter=FieldFilter(
                "user_id",
                "==",
                user_id,
            )
        )
        .order_by(
            "created_at",
            direction="DESCENDING",
        )
        .limit(limit)
    )

    predictions = []

    for document in query.stream():
        data = document.to_dict()

        created_at = data.get("created_at")

        predictions.append({
            "prediction_id": document.id,
            "probability": data.get("probability"),
            "probability_percent": data.get(
                "probability_percent"
            ),
            "risk_category": data.get(
                "risk_category"
            ),
            "model_version": data.get(
                "model_version"
            ),
            "created_at": (
                created_at.isoformat()
                if created_at
                else None
            ),
        })

    return predictions
def get_prediction_by_id(
    user_id: str,
    prediction_id: str,
) -> dict[str, Any] | None:
    """
    Return one prediction only when it belongs to
    the authenticated user.
    """

    database = get_firestore_client()

    document = (
        database.collection("predictions")
        .document(prediction_id)
        .get()
    )

    if not document.exists:
        return None

    data = document.to_dict()

    if data.get("user_id") != user_id:
        return None

    created_at = data.get("created_at")

    return {
        "prediction_id": document.id,
        "probability": data.get("probability"),
        "probability_percent": data.get(
            "probability_percent"
        ),
        "risk_category": data.get("risk_category"),
        "screening_prediction": data.get(
            "screening_prediction"
        ),
        "high_specificity_prediction": data.get(
            "high_specificity_prediction"
        ),
        "thresholds": data.get("thresholds"),
        "component_probabilities": data.get(
            "component_probabilities"
        ),
        "input_data": data.get("input_data"),
        "processed_features": data.get(
            "processed_features"
        ),
        "xai_explanation": data.get(
            "xai_explanation"
        ),
        "xgenai_explanation": data.get(
            "xgenai_explanation"
        ),
        "model_version": data.get("model_version"),
        "created_at": (
            created_at.isoformat()
            if created_at
            else None
        ),
    }
def delete_prediction_by_id(
    user_id: str,
    prediction_id: str,
) -> bool:
    """
    Delete a prediction only when it belongs to
    the authenticated user.
    """

    database = get_firestore_client()

    document_reference = (
        database.collection("predictions")
        .document(prediction_id)
    )

    document = document_reference.get()

    if not document.exists:
        return False

    data = document.to_dict()

    if data.get("user_id") != user_id:
        return False

    document_reference.delete()

    return True