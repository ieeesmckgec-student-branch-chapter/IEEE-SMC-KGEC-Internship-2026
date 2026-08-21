import torch
import numpy as np
import joblib
import json

from model import SwiFT


MODEL_PATH = "../results/centralized/centralized_swift.pth"
SCALER_PATH = "../results/centralized/scaler.pkl"
CLASS_PATH = "../results/centralized/class_mapping.json"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


scaler = joblib.load(
    SCALER_PATH
)


with open(CLASS_PATH, "r") as f:
    class_mapping = json.load(f)

class_mapping = {
    int(k): v
    for k, v in class_mapping.items()
}


model = SwiFT(
    input_dim=7,
    embedding_dim=64,
    num_heads=4,
    num_layers=2,
    num_classes=22,
    dropout=0.1
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


def predict_crop(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):

    features = np.array([
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    ]).reshape(1, -1)

    features_scaled = scaler.transform(
        features
    )

    tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = output.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    crop = class_mapping[
        predicted_class
    ]

    return {
        "crop": crop,
        "class_id": predicted_class,
        "confidence": confidence
    }