import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from ml.preprocessing import load_and_preprocess_data
from ml.model import SwiFT


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

INPUT_DIM = 7
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1


DATASET_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "datasets",
    "Crop_recommendation.csv"
)


MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


print("Device:", DEVICE)
print("Dataset:", DATASET_PATH)


# ============================================================
# LOAD DATA
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    label_encoder
) = load_and_preprocess_data(
    DATASET_PATH
)


# ============================================================
# TORCH DATASETS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


NUM_CLASSES = len(
    label_encoder.classes_
)


# ============================================================
# MODEL
# ============================================================

model = SwiFT(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = total_loss / total

    train_accuracy = correct / total


    # ========================================================
    # TEST
    # ========================================================

    model.eval()

    predictions_all = []
    labels_all = []

    with torch.no_grad():

        for features, labels in test_loader:

            features = features.to(DEVICE)

            outputs = model(features)

            predictions = outputs.argmax(
                dim=1
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

    test_accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Test Acc: {test_accuracy * 100:.2f}%"
    )


# ============================================================
# FINAL METRICS
# ============================================================

precision = precision_score(
    labels_all,
    predictions_all,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    labels_all,
    predictions_all,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    labels_all,
    predictions_all,
    average="weighted",
    zero_division=0
)


print("\n")
print("=" * 60)
print("CENTRALIZED SwiFT RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {test_accuracy * 100:.4f}%"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)


# ============================================================
# SAVE MODEL + PREPROCESSING
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "num_classes": NUM_CLASSES,
        "dropout": DROPOUT,
        "class_names": label_encoder.classes_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    },
    os.path.join(
        MODEL_DIR,
        "centralized_swift.pth"
    )
)


print("\nModel saved successfully.")