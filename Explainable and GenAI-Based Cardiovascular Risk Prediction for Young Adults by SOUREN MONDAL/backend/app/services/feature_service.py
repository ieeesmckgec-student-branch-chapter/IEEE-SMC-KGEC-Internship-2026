from app.schemas.prediction import PredictionRequest


def create_model_features(
    patient: PredictionRequest,
) -> dict[str, float]:
    """
    Convert normal user inputs into the exact 24 features
    used during model training.
    """

    # BMI-derived features
    overweight = int(patient.bmi >= 25)
    obesity = int(patient.bmi >= 30)
    severe_obesity = int(patient.bmi >= 35)

    # Mental-distress features
    moderate_mental_distress = int(
        patient.mentally_unhealthy_days >= 7
    )

    high_mental_distress = int(
        patient.mentally_unhealthy_days >= 14
    )

    # Combined metabolic risk
    metabolic_risk_count = (
        int(patient.diabetes == 1)
        + int(patient.high_blood_pressure == 1)
        + int(patient.high_cholesterol == 1)
        + obesity
    )

    # In Colab, Current_Smoking values 1 and 2 were treated as smoking
    current_smoking_risk = int(
        patient.current_smoking in [1, 2]
    )

    lifestyle_risk_count = (
        current_smoking_risk
        + int(patient.exercise == 0)
        + high_mental_distress
    )

    # Interaction features
    age_bmi_interaction = (
        patient.age * patient.bmi
    )

    age_smoking_interaction = (
        patient.age * patient.current_smoking
    )

    bp_cholesterol_interaction = (
        int(patient.high_blood_pressure == 1)
        * int(patient.high_cholesterol == 1)
    )

    diabetes_obesity_interaction = (
        int(patient.diabetes == 1)
        * obesity
    )

    # Only include the exact final 24 model features
    features = {
        "High_Blood_Pressure": float(
            patient.high_blood_pressure
        ),
        "Metabolic_Risk_Count": float(
            metabolic_risk_count
        ),
        "BP_Cholesterol_Interaction": float(
            bp_cholesterol_interaction
        ),
        "Obesity": float(obesity),
        "Diabetes": float(patient.diabetes),
        "Lifestyle_Risk_Count": float(
            lifestyle_risk_count
        ),
        "Smoking_History": float(
            patient.smoking_history
        ),
        "Age": float(patient.age),
        "Mentally_Unhealthy_Days": float(
            patient.mentally_unhealthy_days
        ),
        "Age_Smoking_Interaction": float(
            age_smoking_interaction
        ),
        "Moderate_Mental_Distress": float(
            moderate_mental_distress
        ),
        "High_Cholesterol": float(
            patient.high_cholesterol
        ),
        "Time_Since_Cholesterol_Check": float(
            patient.time_since_cholesterol_check
        ),
        "Sex": float(patient.sex),
        "Alcohol_Consumption": float(
            patient.alcohol_consumption
        ),
        "Fruit_Intake": float(
            patient.fruit_intake
        ),
        "Current_Smoking": float(
            patient.current_smoking
        ),
        "BMI": float(patient.bmi),
        "Age_BMI_Interaction": float(
            age_bmi_interaction
        ),
        "Overweight": float(overweight),
        "Exercise": float(patient.exercise),
        "Other_Vegetable_Intake": float(
            patient.other_vegetable_intake
        ),
        "Fruit_Juice_Intake": float(
            patient.fruit_juice_intake
        ),
        "Green_Vegetable_Intake": float(
            patient.green_vegetable_intake
        ),
    }

    return features