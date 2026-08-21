import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

TARGET_COLUMN = "label"


def load_and_preprocess_data(csv_path):

    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("Number of classes:", len(label_encoder.classes_))

    print("\nClasses:")
    print(label_encoder.classes_)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoder
    )