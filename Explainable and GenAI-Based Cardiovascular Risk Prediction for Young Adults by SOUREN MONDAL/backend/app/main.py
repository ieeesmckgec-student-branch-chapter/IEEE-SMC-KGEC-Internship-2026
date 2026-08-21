from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.prediction import PredictionRequest
from app.services.auth_service import verify_firebase_token
from app.services.feature_service import create_model_features
from app.services.firebase_service import get_firebase_status
from app.services.gemini_service import generate_xgenai_explanation
from app.services.model_service import (
    get_model_information,
    predict_heart_risk,
)
from app.services.prediction_database_service import (
    delete_prediction_by_id,
    get_prediction_by_id,
    get_user_predictions,
    save_prediction_result,
)
from app.services.shap_service import (
    create_local_shap_explanation,
    create_shap_visualizations,
)


app = FastAPI(
    title="Heart Risk X-GenAI API",
    description=(
        "Backend API for heart-disease risk prediction, "
        "SHAP explainability and generative explanations."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Heart Risk X-GenAI backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Heart Risk X-GenAI API",
        "version": "1.0.0",
    }


@app.get("/api/firebase-status")
def firebase_status():
    """
    Check whether Firebase Admin and Firestore
    are connected to the backend.
    """

    try:
        return get_firebase_status()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Firebase connection failed: "
                f"{error}"
            ),
        ) from error


@app.get("/api/me")
def get_current_user(
    decoded_token: dict = Depends(
        verify_firebase_token
    ),
):
    """
    Return information about the authenticated
    Firebase user.
    """

    return {
        "status": "authenticated",
        "user": {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get(
                "email_verified",
                False,
            ),
            "name": decoded_token.get("name"),
        },
    }


@app.get("/api/predictions")
def prediction_history(
    decoded_token: dict = Depends(
        verify_firebase_token
    ),
):
    """
    Return the authenticated user's latest predictions.
    """

    try:
        user_id = decoded_token.get("uid")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authenticated user ID is missing.",
            )

        predictions = get_user_predictions(
            user_id=user_id,
            limit=20,
        )

        return {
            "status": "success",
            "count": len(predictions),
            "predictions": predictions,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction history could not be loaded: "
                f"{error}"
            ),
        ) from error


@app.get("/api/predictions/{prediction_id}")
def prediction_details(
    prediction_id: str,
    decoded_token: dict = Depends(
        verify_firebase_token
    ),
):
    """
    Return one saved prediction belonging to
    the authenticated user.
    """

    try:
        user_id = decoded_token.get("uid")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authenticated user ID is missing.",
            )

        prediction = get_prediction_by_id(
            user_id=user_id,
            prediction_id=prediction_id,
        )

        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction was not found.",
            )

        return {
            "status": "success",
            "prediction": prediction,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction details could not be loaded: "
                f"{error}"
            ),
        ) from error

@app.get("/api/predictions/{prediction_id}/shap-images")
def prediction_shap_images(
    prediction_id: str,
    decoded_token: dict = Depends(
        verify_firebase_token
    ),
):
    """
    Regenerate SHAP images for a saved prediction.
    """

    try:
        user_id = decoded_token.get("uid")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authenticated user ID is missing.",
            )

        prediction = get_prediction_by_id(
            user_id=user_id,
            prediction_id=prediction_id,
        )

        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction was not found.",
            )

        processed_features = prediction.get(
            "processed_features"
        )

        if not processed_features:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Processed model features are missing "
                    "from this saved prediction."
                ),
            )

        visualizations = create_shap_visualizations(
            input_features=processed_features,
            max_display=10,
        )

        return {
            "status": "success",
            "visualizations": visualizations,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "SHAP images could not be generated: "
                f"{error}"
            ),
        ) from error


@app.delete("/api/predictions/{prediction_id}")
def delete_prediction(
    prediction_id: str,
    decoded_token: dict = Depends(
        verify_firebase_token
    ),
):
    """
    Delete a saved prediction belonging to
    the authenticated user.
    """

    try:
        user_id = decoded_token.get("uid")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authenticated user ID is missing.",
            )

        deleted = delete_prediction_by_id(
            user_id=user_id,
            prediction_id=prediction_id,
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Prediction was not found.",
            )

        return {
            "status": "success",
            "message": "Prediction deleted successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction could not be deleted: "
                f"{error}"
            ),
        ) from error


@app.get("/api/model-info")
def model_information():
    try:
        return get_model_information()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The trained model could not be loaded: "
                f"{error}"
            ),
        ) from error


@app.post("/api/predict")
def create_prediction(
    request: PredictionRequest,
    decoded_token: dict = Depends(
        verify_firebase_token
    ),
):
    """
    Generate an authenticated ensemble prediction,
    SHAP explanation and X-GenAI explanation.
    """

    try:
        user_id = decoded_token.get("uid")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authenticated user ID is missing.",
            )

        model_features = create_model_features(
            request
        )

        result = predict_heart_risk(
            input_features=model_features
        )

        shap_explanation = (
            create_local_shap_explanation(
                input_features=model_features,
                top_n=10,
            )
        )
        shap_visualizations = (
            create_shap_visualizations(
                input_features=model_features,
                max_display=10,
            )
        )

        generated_explanation = (
            generate_xgenai_explanation(
                prediction_result=result,
                xai_explanation=shap_explanation,
            )
        )

        result["user"] = {
            "uid": user_id,
            "email": decoded_token.get("email"),
        }

        result["processed_features"] = (
            model_features
        )

        result["xai_explanation"] = (
            shap_explanation
        )

        result["xgenai_explanation"] = (
            generated_explanation
        )

        prediction_id = save_prediction_result(
            user_id=user_id,
            user_email=decoded_token.get("email"),
            request_data=request.model_dump(),
            prediction_result=result,
        )

        result["prediction_id"] = prediction_id
        result["shap_visualizations"] = (
            shap_visualizations
        )

        return result

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction could not be generated: "
                f"{error}"
            ),
        ) from error