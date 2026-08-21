from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    # Basic information
    age: int = Field(..., ge=18, le=55)
    sex: int = Field(..., ge=0, le=1)
    bmi: float = Field(..., ge=10, le=70)

    # Medical information
    high_blood_pressure: int = Field(..., ge=0, le=1)
    high_cholesterol: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)

    # Lifestyle information
    smoking_history: int = Field(..., ge=0, le=1)
    current_smoking: int = Field(...,ge=0,le=2,description=("Current smoking category used during model training"),)
    exercise: int = Field(..., ge=0, le=1)
    alcohol_consumption: float = Field(..., ge=0)

    # Mental wellbeing
    mentally_unhealthy_days: int = Field(..., ge=0, le=30)

    # Food-frequency survey values
    fruit_intake: float = Field(..., ge=0)
    fruit_juice_intake: float = Field(..., ge=0)
    other_vegetable_intake: float = Field(..., ge=0)
    green_vegetable_intake: float = Field(..., ge=0)

    # Cholesterol-check category used during training
    time_since_cholesterol_check: float = Field(..., ge=0)