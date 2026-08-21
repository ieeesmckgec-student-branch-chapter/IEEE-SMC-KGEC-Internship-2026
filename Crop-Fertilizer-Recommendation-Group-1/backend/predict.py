import os
import torch
import joblib
import numpy as np

from models.swift_model import SwiFT


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "fed_iid",
    "fed_swift_iid.pth"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "preprocessing",
    "scaler.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "preprocessing",
    "label_encoder.pkl"
)

DEVICE = torch.device("cpu")


# ============================================================
# LOAD SCALER
# ============================================================

print("Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded.")


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("Loading label encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print(
    "Label encoder loaded."
)

print(
    "Number of classes:",
    len(label_encoder.classes_)
)


# ============================================================
# CREATE SwiFT MODEL
# ============================================================

print("Creating SwiFT model...")

model = SwiFT(
    input_dim=7,
    embedding_dim=64,
    num_heads=4,
    num_layers=2,
    num_classes=22,
    dropout=0.1
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ------------------------------------------------------------
# Load state dictionary
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    raise ValueError(
        "Unsupported model checkpoint format."
    )


model.to(DEVICE)

model.eval()


print("SwiFT model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    # --------------------------------------------------------
    # Create feature array
    #
    # IMPORTANT:
    # This order MUST remain exactly the same as training.
    #
    # N, P, K, temperature, humidity, ph, rainfall
    # --------------------------------------------------------

    features = np.array(
        [[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Apply the SAME scaler used during training
    # --------------------------------------------------------

    features_scaled = scaler.transform(
        features
    )


    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------------

    tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32
    ).to(DEVICE)


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            tensor
        )


        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Highest probability class
        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()


        # Confidence
        confidence = probabilities[
            0,
            predicted_class
        ].item()


    # --------------------------------------------------------
    # Convert class number to crop name
    # --------------------------------------------------------

    crop = label_encoder.inverse_transform(
        [predicted_class]
    )[0]


    return crop, confidence