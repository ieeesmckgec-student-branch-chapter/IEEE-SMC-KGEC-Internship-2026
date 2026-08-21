import os
from pathlib import Path
from typing import Any

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore


load_dotenv()


BACKEND_DIR = Path(__file__).resolve().parents[2]

FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    "firebase-service-account.json",
)


_firebase_app: Any | None = None
_firestore_client: Any | None = None


def initialize_firebase():
    """
    Initialize Firebase Admin SDK once.
    """

    global _firebase_app

    if _firebase_app is not None:
        return _firebase_app

    # Reuse an already initialized Firebase app
    if firebase_admin._apps:
        _firebase_app = firebase_admin.get_app()
        return _firebase_app

    credentials_path = Path(
        FIREBASE_CREDENTIALS_PATH
    )

    if not credentials_path.is_absolute():
        credentials_path = (
            BACKEND_DIR / credentials_path
        )

    if not credentials_path.exists():
        raise FileNotFoundError(
            "Firebase service-account file was not found at: "
            f"{credentials_path}"
        )

    firebase_credentials = credentials.Certificate(
        str(credentials_path)
    )

    _firebase_app = firebase_admin.initialize_app(
        firebase_credentials
    )

    return _firebase_app


def get_firestore_client():
    """
    Return one reusable Firestore client.
    """

    global _firestore_client

    if _firestore_client is not None:
        return _firestore_client

    initialize_firebase()

    _firestore_client = firestore.client()

    return _firestore_client


def get_firebase_status() -> dict[str, Any]:
    """
    Test Firebase and Firestore connectivity.
    """

    app = initialize_firebase()
    database = get_firestore_client()

    # Access a collection reference without writing data
    database.collection("system_checks")

    project_id = app.project_id

    return {
        "status": "connected",
        "firebase_admin_initialized": True,
        "firestore_initialized": True,
        "project_id": project_id,
    }