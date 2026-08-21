import os
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "centralized_swift.pth"
)


# ============================================================
# CROP CLASSES
# ============================================================

CLASS_NAMES = [
    "apple",
    "banana",
    "blackgram",
    "chickpea",
    "coconut",
    "coffee",
    "cotton",
    "grapes",
    "jute",
    "kidneybeans",
    "lentil",
    "maize",
    "mango",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "orange",
    "papaya",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon"
]


# ============================================================
# SwiFT MODEL
# ============================================================

class SwiFT(nn.Module):

    def __init__(
        self,
        input_dim=7,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=22,
        dropout=0.1
    ):

        super().__init__()

        self.embedding = nn.Linear(
            input_dim,
            embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

        x = self.embedding(x)

        # Transformer expects sequence dimension
        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x[:, 0, :]

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")


model = SwiFT(
    input_dim=7,
    embedding_dim=64,
    num_heads=4,
    num_layers=2,
    num_classes=22,
    dropout=0.1
)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# Handle normal state_dict
if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]


model.load_state_dict(checkpoint)

model.to(device)

model.eval()


print(
    f"Loaded SwiFT model from: {MODEL_PATH}"
)


# ============================================================
# PREDICTION
# ============================================================

def predict_crop(data):

    required_features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    values = []

    for feature in required_features:

        if feature not in data:

            raise ValueError(
                f"Missing feature: {feature}"
            )

        values.append(
            float(data[feature])
        )


    features = np.array(
        values,
        dtype=np.float32
    )


    tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0)


    tensor = tensor.to(device)


    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()


    crop = CLASS_NAMES[
        predicted_class
    ]


    confidence = probabilities[
        0,
        predicted_class
    ].item()


    return {
        "crop": crop,
        "confidence": round(
            confidence * 100,
            2
        )
    }