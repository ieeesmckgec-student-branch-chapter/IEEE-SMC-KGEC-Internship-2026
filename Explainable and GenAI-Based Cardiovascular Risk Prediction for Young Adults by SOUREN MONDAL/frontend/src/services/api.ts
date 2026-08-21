import { auth } from "../lib/firebase";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

export type PredictionRequest = {
  age: number;
  sex: number;
  bmi: number;
  high_blood_pressure: number;
  high_cholesterol: number;
  diabetes: number;
  smoking_history: number;
  current_smoking: number;
  exercise: number;
  alcohol_consumption: number;
  mentally_unhealthy_days: number;
  fruit_intake: number;
  fruit_juice_intake: number;
  other_vegetable_intake: number;
  green_vegetable_intake: number;
  time_since_cholesterol_check: number;
};

export type ShapFactor = {
  feature: string;
  value: number;
  shap_value: number;
  absolute_shap_value: number;
  effect: string;
};

export type XaiExplanation = {
  explained_model: string;
  explanation_method: string;
  base_value: number;
  top_contributors: ShapFactor[];
  increasing_factors: ShapFactor[];
  decreasing_factors: ShapFactor[];
  important_note: string;
};

export type ShapVisualizations = {
  waterfall_image: string;
  bar_image: string;
  explained_model: string;
  important_note: string;
};

export type PredictionThresholds = {
  screening: number;
  high_specificity: number;
};

export type ComponentProbabilities = {
  logistic_regression: number;
  random_forest: number;
  xgboost: number;
};

export type PredictionResponse = {
  status: string;
  model_version: string;
  probability: number;
  probability_percent: number;
  risk_category: string;
  screening_prediction: boolean;
  high_specificity_prediction: boolean;
  thresholds: PredictionThresholds;
  component_probabilities: ComponentProbabilities;

  user: {
    uid: string;
    email: string | null;
  };

  processed_features: Record<string, number>;
  xai_explanation: XaiExplanation;
  xgenai_explanation: string;
  prediction_id: string;
  shap_visualizations: ShapVisualizations;
};

export type PredictionHistoryItem = {
  prediction_id: string;
  probability: number;
  probability_percent: number;
  risk_category: string;
  screening_prediction: boolean;
  high_specificity_prediction: boolean;
  model_version: string;
  created_at: string | null;
};

export type PredictionDetails = {
  prediction_id: string;
  probability: number;
  probability_percent: number;
  risk_category: string;
  screening_prediction: boolean;
  high_specificity_prediction: boolean;
  thresholds: PredictionThresholds;
  component_probabilities: ComponentProbabilities;
  input_data: PredictionRequest;
  processed_features: Record<string, number>;
  xai_explanation: XaiExplanation;
  xgenai_explanation: string;
  model_version: string;
  created_at: string | null;
};

type AuthenticatedUserResponse = {
  status: string;

  user: {
    uid: string;
    email: string | null;
    email_verified: boolean;
    name: string | null;
  };
};

type PredictionHistoryResponse = {
  status: string;
  count: number;
  predictions: PredictionHistoryItem[];
};

type PredictionDetailsResponse = {
  status: string;
  prediction: PredictionDetails;
};

type DeletePredictionResponse = {
  status: string;
  message: string;
};

type ShapImagesResponse = {
  status: string;
  visualizations: ShapVisualizations;
};

async function getAuthenticationToken() {
  const user = auth.currentUser;

  if (!user) {
    throw new Error(
      "You must be logged in to use this service."
    );
  }

  return user.getIdToken(true);
}

async function readApiError(
  response: Response,
  fallbackMessage: string
) {
  const errorData = await response
    .json()
    .catch(() => null);

  return (
    errorData?.detail ||
    fallbackMessage
  );
}

export async function getAuthenticatedUser(): Promise<AuthenticatedUserResponse> {
  const idToken =
    await getAuthenticationToken();

  const response = await fetch(
    `${API_URL}/api/me`,
    {
      method: "GET",
      headers: {
        Authorization: `Bearer ${idToken}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Backend authentication failed."
      )
    );
  }

  return response.json();
}

export async function createPrediction(
  requestData: PredictionRequest
): Promise<PredictionResponse> {
  const idToken =
    await getAuthenticationToken();

  const response = await fetch(
    `${API_URL}/api/predict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${idToken}`,
      },
      body: JSON.stringify(requestData),
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Prediction could not be generated."
      )
    );
  }

  return response.json();
}

export async function getPredictionHistory(): Promise<PredictionHistoryResponse> {
  const idToken =
    await getAuthenticationToken();

  const response = await fetch(
    `${API_URL}/api/predictions`,
    {
      method: "GET",
      headers: {
        Authorization: `Bearer ${idToken}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Prediction history could not be loaded."
      )
    );
  }

  return response.json();
}

export async function getPredictionDetails(
  predictionId: string
): Promise<PredictionDetailsResponse> {
  const idToken =
    await getAuthenticationToken();

  const response = await fetch(
    `${API_URL}/api/predictions/${predictionId}`,
    {
      method: "GET",
      headers: {
        Authorization: `Bearer ${idToken}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Prediction details could not be loaded."
      )
    );
  }

  return response.json();
}

export async function getPredictionShapImages(
  predictionId: string
): Promise<ShapImagesResponse> {
  const idToken =
    await getAuthenticationToken();

  const response = await fetch(
    `${API_URL}/api/predictions/${predictionId}/shap-images`,
    {
      method: "GET",
      headers: {
        Authorization: `Bearer ${idToken}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "SHAP images could not be loaded."
      )
    );
  }

  return response.json();
}

export async function deletePrediction(
  predictionId: string
): Promise<DeletePredictionResponse> {
  const idToken =
    await getAuthenticationToken();

  const response = await fetch(
    `${API_URL}/api/predictions/${predictionId}`,
    {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${idToken}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Prediction could not be deleted."
      )
    );
  }

  return response.json();
}