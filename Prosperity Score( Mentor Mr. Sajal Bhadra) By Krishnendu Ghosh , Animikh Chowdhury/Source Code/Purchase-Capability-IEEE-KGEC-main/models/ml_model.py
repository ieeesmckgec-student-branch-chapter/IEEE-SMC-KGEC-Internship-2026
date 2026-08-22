# models/ml_model.py
from __future__ import annotations
from typing import Any, Optional
import logging
from joblib import dump, load

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class MLModel:
    """
    Thin wrapper around a sklearn-like model/pipeline that exposes
    predict / predict_proba and save/load helpers via joblib.
    """
    def __init__(self, model: Optional[Any] = None):
        self.model = model

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # some models may not have predict_proba; handle gracefully
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError("Underlying model has no predict_proba")

    def save(self, path: str):
        """Persist underlying model using joblib (recommended for sklearn)."""
        if self.model is None:
            raise RuntimeError("No model to save")
        dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "MLModel":
        m = load(path)
        return MLModel(m)