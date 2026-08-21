from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.crop_predictor import predict_crop


app = FastAPI(
    title="Crop Recommendation API",
    version="1.0.0"
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Crop Recommendation API is running"
    }


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ------------------------------------------------------------
# CROP PREDICTION
# ------------------------------------------------------------

@app.post("/predict")
def predict(data: dict):

    result = predict_crop(data)

    return result